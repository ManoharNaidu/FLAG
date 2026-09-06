"""FLAG semantic ego-subgraph sampler (paper section 3.1, Eq. 3 and Eq. 4).

For every target node v we build one ego-subgraph:

  1. take the k-hop neighbourhood of v (candidates = that set minus v itself);
  2. Eq. 3 -- cosine similarity between the frozen-LM text embedding of v and
     of every candidate u:   sim(v,u) = B(t_v).B(t_u) / (||B(t_v)|| ||B(t_u)||);
  3. Eq. 4 -- keep the candidates with sim >= delta, then take the top-N of
     those by similarity;
  4. subset = [v] + selected neighbours, stored as GLOBAL node ids, v first;
  5. edge_index = induced subgraph over `subset`, relabelled to LOCAL ids.

Data contract consumed downstream (train.py / chat.py / encode.py):
  batch.subset      LongTensor of GLOBAL node ids, batch.central is subset[0]
  batch.central     0-dim LongTensor, the GLOBAL id of the target node
  batch.edge_index  LOCAL indices into `subset` (used with data.x[batch.subset])
  batch.raw_texts   list[str] aligned with `subset`

Usage
-----
python sampler.py --data Instagram/instagram_1to10.pt --out-dir Instagram/ss_d0_n10_k2
"""

import argparse
import hashlib
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


LM_NAME = "all-MiniLM-L6-v2"

# selection strategies used by the sampler and by the Figure-3(a) study
STRATEGIES = ("ns", "rs", "fs", "ss_star", "ss")


def safe_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def resolve_device(name):
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _texts_fingerprint(texts):
    h = hashlib.sha1()
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def load_or_build_embeddings(texts, cache_path, device, model_name=LM_NAME,
                             batch_size=128, verbose=True):
    """Frozen-LM sentence embeddings B(t) for every node, with an on-disk cache."""
    fp = _texts_fingerprint(texts)
    if cache_path and os.path.exists(cache_path):
        blob = safe_load(cache_path)
        if (isinstance(blob, dict) and blob.get("fingerprint") == fp
                and blob.get("model") == model_name):
            if verbose:
                print(f"[emb] reusing cache {cache_path} "
                      f"({tuple(blob['embeddings'].shape)})")
            return blob["embeddings"].to(device).float()
        if verbose:
            print(f"[emb] cache {cache_path} does not match this graph/model, "
                  f"recomputing")

    from sentence_transformers import SentenceTransformer

    if verbose:
        print(f"[emb] encoding {len(texts)} texts with frozen LM '{model_name}' ...")
    encoder = SentenceTransformer(model_name, device=str(device))
    emb = encoder.encode(list(texts), batch_size=batch_size,
                         convert_to_numpy=True, show_progress_bar=verbose)
    emb = torch.as_tensor(emb).float()

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save({"embeddings": emb, "fingerprint": fp, "model": model_name},
                   cache_path)
        if verbose:
            print(f"[emb] cached to {cache_path} ({tuple(emb.shape)})")
    return emb.to(device).float()


class EgoSampler:
    """Builds one ego-subgraph per target node under a given selection strategy."""

    def __init__(self, edge_index, num_nodes, lm_emb=None, shallow_x=None,
                 k=2, topn=10, delta=0.0, seed=0, device=None):
        self.device = device or torch.device("cpu")
        self.edge_index = edge_index.to(self.device)
        self.num_nodes = int(num_nodes)
        self.k = int(k)
        self.topn = int(topn)
        self.delta = float(delta)

        # L2-normalised once so Eq. 3 reduces to a dot product
        self.lm_unit = (F.normalize(lm_emb.to(self.device).float(), p=2, dim=1)
                        if lm_emb is not None else None)
        self.x_unit = (F.normalize(shallow_x.to(self.device).float(), p=2, dim=1)
                       if shallow_x is not None else None)

        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        # scratch buffer for local relabelling, reused across calls
        self._map = torch.full((self.num_nodes,), -1, dtype=torch.long,
                               device=self.device)

    # ---- Eq. 3 --------------------------------------------------------
    def similarity(self, v, candidates, space="lm"):
        unit = self.lm_unit if space == "lm" else self.x_unit
        if unit is None:
            raise ValueError(f"no embedding table for space='{space}'")
        return unit[candidates] @ unit[v]

    # ---- Eq. 4 --------------------------------------------------------
    def select(self, v, candidates, strategy):
        """Return the neighbours to keep, ordered as the strategy ranks them."""
        if candidates.numel() == 0 or strategy == "ns":
            return candidates

        if strategy == "rs":
            if candidates.numel() <= self.topn:
                return candidates
            perm = torch.randperm(candidates.numel(), generator=self.generator)
            return candidates[perm[:self.topn].to(candidates.device)]

        space = "x" if strategy == "fs" else "lm"
        sim = self.similarity(v, candidates, space=space)

        if strategy == "ss":  # threshold first, then top-N
            keep = sim >= self.delta
            candidates, sim = candidates[keep], sim[keep]
            if candidates.numel() == 0:
                return candidates

        n = min(self.topn, candidates.numel())
        order = torch.topk(sim, n).indices
        return candidates[order]

    # ---- ego-subgraph -------------------------------------------------
    def build(self, v, strategy="ss"):
        v = int(v)
        nodes, khop_ei, _, _ = k_hop_subgraph(
            v, self.k, self.edge_index, relabel_nodes=False,
            num_nodes=self.num_nodes)
        candidates = nodes[nodes != v]

        selected = self.select(v, candidates, strategy)

        v_t = torch.tensor([v], dtype=torch.long, device=self.device)
        subset = torch.cat([v_t, selected.to(self.device)])

        # induced subgraph over `subset`, relabelled to local ids.
        # k_hop_subgraph (undirected) already returns every edge whose two
        # endpoints lie in the k-hop node set, so filtering it is exactly the
        # induced subgraph over the (smaller) `subset`.
        self._map[subset] = torch.arange(subset.numel(), device=self.device)
        src_local = self._map[khop_ei[0]]
        dst_local = self._map[khop_ei[1]]
        keep = (src_local >= 0) & (dst_local >= 0)
        local_ei = torch.stack([src_local[keep], dst_local[keep]], dim=0)
        self._map[subset] = -1  # reset scratch buffer

        out = Data(edge_index=local_ei.cpu())
        out.subset = subset.cpu()
        out.central = torch.tensor(v, dtype=torch.long)
        out.num_nodes = int(subset.numel())
        return out


def attach_texts(batch, raw_texts):
    batch.raw_texts = [raw_texts[int(i)] for i in batch.subset.tolist()]
    return batch


def main():
    parser = argparse.ArgumentParser(description="FLAG semantic ego-subgraph sampler")
    parser.add_argument("--data", type=str, default="Instagram/instagram_1to10.pt")
    parser.add_argument("--out-dir", type=str, default="Instagram/ss_d0_n10_k2")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--emb-cache", type=str, default="Instagram/lm_embeddings.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="ss", choices=list(STRATEGIES))
    args = parser.parse_args()

    device = resolve_device(args.device)
    data = safe_load(args.data)
    num_nodes = int(data.y.numel())

    print("=" * 72)
    print("FLAG ego-subgraph sampling (paper sec. 3.1, Eq. 3 / Eq. 4)")
    print("=" * 72)
    print(f"data     : {args.data}  (N={num_nodes}, E={data.edge_index.size(1)})")
    print(f"device   : {device}")
    print(f"strategy : {args.strategy}  k={args.k}  top-N={args.topn} "
          f"delta={args.delta}")

    lm_emb = load_or_build_embeddings(data.raw_texts, args.emb_cache, device)

    sampler = EgoSampler(data.edge_index, num_nodes, lm_emb=lm_emb,
                         k=args.k, topn=args.topn, delta=args.delta,
                         seed=args.seed, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }

    grand_isolated = 0
    for name, mask in splits.items():
        targets = mask.nonzero(as_tuple=False).view(-1).tolist()
        batches = []
        isolated = 0
        sizes = 0
        t0 = time.time()
        iterator = (tqdm(targets, desc=f"{name:<5}", unit="node")
                    if tqdm is not None else targets)
        for n_done, v in enumerate(iterator, 1):
            batch = attach_texts(sampler.build(v, strategy=args.strategy),
                                 data.raw_texts)
            if batch.subset.numel() == 1:
                isolated += 1
            sizes += batch.subset.numel()
            batches.append(batch)
            if tqdm is None and n_done % 500 == 0:
                print(f"  {name}: {n_done}/{len(targets)} "
                      f"({time.time() - t0:.1f}s)")

        out_path = os.path.join(args.out_dir, f"{name}_sampler.pt")
        torch.save(batches, out_path)
        grand_isolated += isolated
        print(f"[{name:<5}] {len(batches)} subgraphs -> {out_path}  "
              f"avg |subset|={sizes / max(len(batches), 1):.2f}  "
              f"isolated(no k-hop neighbour)={isolated}  "
              f"({time.time() - t0:.1f}s)")

    print(f"done. total isolated target nodes: {grand_isolated}")


if __name__ == "__main__":
    main()
