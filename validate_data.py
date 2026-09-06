"""Acceptance tests for preprocess.py + sampler.py, plus the Figure-3(a) study.

Runs plain ``assert``s (no pytest) over

  * the downsampled graph produced by preprocess.py, and
  * the ego-subgraph files produced by sampler.py,

and then reproduces the paper's Figure 3(a): average subgraph edge homophily
(Eq. 5) under five neighbour-selection strategies (NS / RS / FS / SS* / SS).

Usage
-----
python validate_data.py
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F

from sampler import (EgoSampler, load_or_build_embeddings, resolve_device,
                     safe_load)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PASSED = []
NOTES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {name}" + (f"  ({detail})" if detail else "")
    print(line)
    PASSED.append((name, bool(condition)))
    assert condition, f"{name} failed. {detail}"


def section(title):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


# =====================================================================
# 1. preprocessing checks
# =====================================================================
def validate_preprocessing(orig, new, ratio, tol=0.05):
    section("1. PREPROCESSING  (preprocess.py output)")

    n = int(new.y.numel())
    n0 = int((new.y == 0).sum())
    n1 = int((new.y == 1).sum())
    achieved = n0 / n1
    rel_err = abs(achieved - ratio) / ratio
    check("minority:majority ratio within 5% of requested",
          rel_err <= tol,
          f"requested 1:{ratio:g}, achieved 1:{achieved:.4f}, "
          f"rel.err={rel_err:.4%}")

    ei = new.edge_index
    check("edge_index.max() < num_nodes", int(ei.max()) < n,
          f"max={int(ei.max())}, N={n}")
    check("edge_index.min() >= 0", int(ei.min()) >= 0, f"min={int(ei.min())}")

    tm, vm, sm = new.train_mask, new.val_mask, new.test_mask
    check("train/val/test masks pairwise disjoint",
          int((tm & vm).sum()) == 0 and int((tm & sm).sum()) == 0
          and int((vm & sm).sum()) == 0)
    check("train/val/test masks cover every node",
          int((tm | vm | sm).sum()) == n,
          f"{int((tm | vm | sm).sum())}/{n}")

    check("len(raw_texts) == N == x.shape[0] == y.shape[0]",
          len(new.raw_texts) == n == new.x.shape[0] == new.y.shape[0],
          f"texts={len(new.raw_texts)} x={tuple(new.x.shape)} "
          f"y={tuple(new.y.shape)}")

    check("label_name preserved",
          list(new.label_name) == list(orig.label_name),
          f"{list(new.label_name)}")

    dropped_present = [k for k in new.keys()
                       if k.startswith(("one_shot", "three_shot", "five_shot"))]
    check("few-shot masks dropped", len(dropped_present) == 0,
          f"still present: {dropped_present}")

    # ---- re-indexing alignment, proved against the ORIGINAL file ----
    check("orig_index map present and 1-to-1",
          hasattr(new, "orig_index")
          and new.orig_index.numel() == n
          and int(new.orig_index.unique().numel()) == n)
    oi = new.orig_index
    check("x rows match original rows for every kept node",
          torch.equal(new.x, orig.x[oi]))
    check("y values match original for every kept node",
          torch.equal(new.y, orig.y[oi]))
    texts_ok = all(new.raw_texts[i] == orig.raw_texts[int(oi[i])]
                   for i in range(n))
    check("raw_texts match original for every kept node", texts_ok)

    # independent structural proof: map new edges back to original ids and
    # confirm they are genuine original edges, and that none were lost.
    N_old = int(orig.y.numel())
    orig_pairs = set((orig.edge_index[0] * N_old + orig.edge_index[1]).tolist())
    mapped = (oi[ei[0]] * N_old + oi[ei[1]]).tolist()
    check("every retained edge is a real edge of the original graph",
          all(p in orig_pairs for p in mapped),
          f"{len(mapped)} edges checked")

    keep_mask = torch.zeros(N_old, dtype=torch.bool)
    keep_mask[oi] = True
    expected = int((keep_mask[orig.edge_index[0]]
                    & keep_mask[orig.edge_index[1]]).sum())
    check("no induced edge was lost (edge count == induced count)",
          ei.size(1) == expected, f"{ei.size(1)} == {expected}")

    print(f"\n  summary: N={n} (class0={n0}, class1={n1}), E={ei.size(1)}, "
          f"train/val/test = {int(tm.sum())}/{int(vm.sum())}/{int(sm.sum())}")


# =====================================================================
# 2. sampler checks
# =====================================================================
def validate_sampler(data, sampler_dir, topn, delta, lm_unit, n_check):
    section("2. SAMPLER  (sampler.py output)")

    N = int(data.y.numel())
    edge_pairs = set((data.edge_index[0] * N + data.edge_index[1]).tolist())
    splits = {"train": data.train_mask, "val": data.val_mask,
              "test": data.test_mask}

    rng = random.Random(0)
    for name, mask in splits.items():
        path = os.path.join(sampler_dir, f"{name}_sampler.pt")
        batches = safe_load(path)
        expected_targets = set(mask.nonzero(as_tuple=False).view(-1).tolist())

        print(f"\n  -- {name} ({len(batches)} subgraphs from {path})")

        # --- target-node coverage (uses every batch) ------------------
        centrals = [int(b.central) for b in batches]
        check(f"[{name}] target nodes == nodes in {name}_mask",
              set(centrals) == expected_targets
              and len(centrals) == len(expected_targets),
              f"{len(centrals)} targets vs {len(expected_targets)} masked")

        # --- structural checks on every batch -------------------------
        bad_fields = bad_central = bad_size = bad_local = bad_texts = 0
        for b in batches:
            if not all(hasattr(b, f) for f in
                       ("subset", "central", "edge_index", "raw_texts")):
                bad_fields += 1
                continue
            if int(b.central) not in set(b.subset.tolist()):
                bad_central += 1
            if b.subset.numel() > topn + 1:
                bad_size += 1
            if b.edge_index.numel() > 0 and int(b.edge_index.max()) >= b.subset.numel():
                bad_local += 1
            if len(b.raw_texts) != b.subset.numel():
                bad_texts += 1
        check(f"[{name}] every batch has subset/central/edge_index/raw_texts",
              bad_fields == 0, f"{bad_fields} bad")
        check(f"[{name}] central is in subset", bad_central == 0,
              f"{bad_central} bad")
        check(f"[{name}] len(subset) <= topn+1 = {topn + 1}", bad_size == 0,
              f"{bad_size} bad")
        check(f"[{name}] edge_index is LOCAL (max < len(subset))",
              bad_local == 0, f"{bad_local} bad")
        check(f"[{name}] raw_texts aligned with subset", bad_texts == 0,
              f"{bad_texts} bad")

        # --- deeper checks on a sample --------------------------------
        k = min(max(n_check, 200), len(batches))
        sample = rng.sample(range(len(batches)), k)

        bad_edges = bad_sim = bad_text_content = bad_first = 0
        n_edges_checked = 0
        min_sim = 1.0
        for i in sample:
            b = batches[i]
            sub = b.subset
            if int(b.central) != int(sub[0]):
                bad_first += 1
            # induced-subgraph correctness: local -> global must be real edges
            if b.edge_index.numel() > 0:
                g_src = sub[b.edge_index[0]]
                g_dst = sub[b.edge_index[1]]
                keys = (g_src * N + g_dst).tolist()
                n_edges_checked += len(keys)
                if any(kk not in edge_pairs for kk in keys):
                    bad_edges += 1
            # Eq. 4 threshold
            if sub.numel() > 1:
                v = int(b.central)
                sims = lm_unit[sub[1:]] @ lm_unit[v]
                min_sim = min(min_sim, float(sims.min()))
                if float(sims.min()) < delta - 1e-6:
                    bad_sim += 1
            if [data.raw_texts[int(j)] for j in sub.tolist()] != list(b.raw_texts):
                bad_text_content += 1

        check(f"[{name}] central node is subset[0]", bad_first == 0,
              f"{bad_first} bad")
        check(f"[{name}] every batch edge is a real graph edge "
              f"(induced-subgraph correctness)",
              bad_edges == 0,
              f"{k} batches / {n_edges_checked} edges checked, {bad_edges} bad")
        check(f"[{name}] every selected neighbour has sim >= delta={delta}",
              bad_sim == 0,
              f"{k} batches checked, min sim seen = {min_sim:.4f}")
        check(f"[{name}] raw_texts content matches data.raw_texts[subset]",
              bad_text_content == 0, f"{bad_text_content} bad")

        sizes = torch.tensor([b.subset.numel() for b in batches],
                             dtype=torch.float)
        iso = int((sizes == 1).sum())
        print(f"     |subset|: mean={sizes.mean():.2f} max={int(sizes.max())} "
              f"isolated(no neighbour)={iso} ({iso / len(batches):.1%})")


# =====================================================================
# 3. Figure 3(a): average subgraph edge homophily (Eq. 5)
# =====================================================================
STRATEGY_LABELS = {
    "ns": ("NS ", "no sampling: the full k-hop ego-graph"),
    "rs": ("RS ", "random N neighbours from the k-hop set"),
    "fs": ("FS ", "top-N by cosine on shallow features data.x"),
    "ss_star": ("SS*", "top-N by cosine on LM text embeddings, no threshold"),
    "ss": ("SS ", f"top-N by LM cosine WITH threshold (paper's method)"),
}


def subgraph_homophily(sampler, y, targets, strategy, drop_self_loops=True):
    """Mean over subgraphs of  (# edges with y(u)==y(v)) / |E|   (Eq. 5).

    Also returns a secondary, more targeted statistic: the fraction of the
    *selected* neighbours that share the central node's label.  Eq. 5 counts
    every edge of the induced subgraph, including neighbour-neighbour edges
    that the selection rule does not control; the central-agreement number
    isolates what the sampler actually decides.
    """
    vals = []
    cvals = []
    n_empty = 0
    tot_edges = 0
    tot_nodes = 0
    it = (tqdm(targets, desc=STRATEGY_LABELS[strategy][0], unit="node",
               leave=False) if tqdm is not None else targets)
    for v in it:
        b = sampler.build(v, strategy=strategy)
        ei = b.edge_index
        tot_nodes += b.subset.numel()
        if b.subset.numel() > 1:
            cvals.append(float((y[b.subset[1:]] == y[b.subset[0]]).float().mean()))
        if drop_self_loops and ei.numel() > 0:
            ei = ei[:, ei[0] != ei[1]]
        if ei.size(1) == 0:
            n_empty += 1
            continue
        ys = y[b.subset]
        same = (ys[ei[0]] == ys[ei[1]]).float()
        vals.append(float(same.mean()))
        tot_edges += ei.size(1)
    mean = sum(vals) / len(vals) if vals else float("nan")
    return {
        "homophily": mean,
        "central_agree": (sum(cvals) / len(cvals)) if cvals else float("nan"),
        "n_scored": len(vals),
        "n_empty": n_empty,
        "avg_nodes": tot_nodes / max(len(targets), 1),
        "avg_edges": tot_edges / max(len(vals), 1),
    }


def figure3a_study(data, lm_emb, k, topn, delta, seed, device, n_nodes,
                   drop_self_loops=True):
    section("3. PAPER REPRODUCTION - Figure 3(a): avg subgraph edge homophily "
            "(Eq. 5)")

    N = int(data.y.numel())
    y = data.y.to(device)
    g = torch.Generator().manual_seed(seed)
    all_nodes = torch.randperm(N, generator=g)[:min(n_nodes, N)].tolist()

    print(f"  sampled {len(all_nodes)} target nodes (seed={seed}), k={k}, "
          f"top-N={topn}, delta={delta}")
    print(f"  self-loops {'EXCLUDED from' if drop_self_loops else 'INCLUDED in'}"
          f" the homophily denominator")

    results = {}
    for strat in ("ns", "rs", "fs", "ss_star", "ss"):
        sampler = EgoSampler(data.edge_index, N, lm_emb=lm_emb,
                             shallow_x=data.x if strat == "fs" else None,
                             k=k, topn=topn, delta=delta, seed=seed,
                             device=device)
        results[strat] = subgraph_homophily(sampler, y, all_nodes, strat,
                                            drop_self_loops=drop_self_loops)
        del sampler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # chance level implied by the class distribution alone
    p = torch.bincount(data.y, minlength=2).float() / N
    chance = float((p * p).sum())

    print()
    print(f"  {'strategy':<9} {'homo(Eq.5)':>11} {'centr.agree':>12} "
          f"{'scored':>7} {'no-edge':>8} {'avg|V|':>8} {'avg|E|':>9}   description")
    print("  " + "-" * 112)
    for strat in ("ns", "rs", "fs", "ss_star", "ss"):
        r = results[strat]
        label, desc = STRATEGY_LABELS[strat]
        print(f"  {label:<9} {r['homophily']:>11.4f} {r['central_agree']:>12.4f} "
              f"{r['n_scored']:>7} {r['n_empty']:>8} {r['avg_nodes']:>8.2f} "
              f"{r['avg_edges']:>9.2f}   {desc}")
    print(f"  {'chance':<9} {chance:>11.4f} {chance:>12.4f} "
          f"{'-':>7} {'-':>8} {'-':>8} {'-':>9}   "
          f"label-agreement of two random nodes (sum p_c^2)")

    order = ["ns", "rs", "fs", "ss_star", "ss"]
    vals = [results[s]["homophily"] for s in order]
    reproduced = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print()
    print("  paper's Figure 3(a) ordering: NS < RS < FS < SS* < SS")
    print("  observed ordering (low -> high): "
          + " < ".join(STRATEGY_LABELS[s][0].strip()
                       for s in sorted(order, key=lambda s: results[s]["homophily"])))
    print(f"  ORDERING REPRODUCED: {'YES' if reproduced else 'NO'}")
    if not reproduced:
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            rel = "<" if vals[i] < vals[i + 1] else (">" if vals[i] > vals[i+1] else "==")
            flag = "ok" if rel == "<" else "VIOLATED"
            print(f"    expected {STRATEGY_LABELS[a][0].strip()} < "
                  f"{STRATEGY_LABELS[b][0].strip()}: got "
                  f"{vals[i]:.4f} {rel} {vals[i+1]:.4f}  [{flag}]")
    cvals = [results[s]["central_agree"] for s in order]
    c_repro = all(cvals[i] < cvals[i + 1] for i in range(len(cvals) - 1))
    print()
    print("  same ordering test on the central-agreement statistic: "
          + " < ".join(STRATEGY_LABELS[s][0].strip()
                       for s in sorted(order, key=lambda s: results[s]["central_agree"]))
          + f"   -> {'YES' if c_repro else 'NO'}")

    NOTES.append(("figure3a_ordering_reproduced", reproduced))
    return results, reproduced


def delta_sweep(data, lm_emb, k, topn, seed, device, n_nodes, deltas,
                drop_self_loops=True):
    """delta=0 makes SS almost identical to SS*; show what the threshold does."""
    section("3b. Effect of the Eq. 4 threshold delta on SS")
    N = int(data.y.numel())
    y = data.y.to(device)
    g = torch.Generator().manual_seed(seed)
    targets = torch.randperm(N, generator=g)[:min(n_nodes, N)].tolist()

    print(f"  {'delta':>7} {'homo(Eq.5)':>11} {'centr.agree':>12} "
          f"{'avg|V|':>8} {'no-edge':>8}")
    print("  " + "-" * 52)
    for d in deltas:
        s = EgoSampler(data.edge_index, N, lm_emb=lm_emb, k=k, topn=topn,
                       delta=d, seed=seed, device=device)
        r = subgraph_homophily(s, y, targets, "ss",
                               drop_self_loops=drop_self_loops)
        print(f"  {d:>7.2f} {r['homophily']:>11.4f} {r['central_agree']:>12.4f} "
              f"{r['avg_nodes']:>8.2f} {r['n_empty']:>8}")
        del s
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="FLAG data validation / acceptance tests")
    p.add_argument("--orig", type=str, default="Instagram/instagram.pt")
    p.add_argument("--data", type=str, default="Instagram/instagram_1to10.pt")
    p.add_argument("--sampler-dir", type=str, default="Instagram/ss_d0_n10_k2")
    p.add_argument("--emb-cache", type=str, default="Instagram/lm_embeddings.pt")
    p.add_argument("--ratio", type=float, default=10.0)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-check", type=int, default=500,
                   help="batches per split for the expensive per-edge checks")
    p.add_argument("--homo-nodes", type=int, default=2000,
                   help="target nodes used for the Figure 3(a) study")
    p.add_argument("--homo-self-loops", action="store_true",
                   help="count self-loops in the homophily denominator")
    p.add_argument("--delta-sweep", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5",
                   help="comma-separated deltas for the SS threshold sweep "
                        "(empty string disables)")
    args = p.parse_args()

    device = resolve_device(args.device)
    print("=" * 72)
    print("FLAG data validation")
    print("=" * 72)
    print(f"original : {args.orig}")
    print(f"processed: {args.data}")
    print(f"sampler  : {args.sampler_dir}")
    print(f"device   : {device}")

    orig = safe_load(args.orig)
    new = safe_load(args.data)

    validate_preprocessing(orig, new, args.ratio)
    del orig

    lm_emb = load_or_build_embeddings(new.raw_texts, args.emb_cache, device,
                                      verbose=False)
    lm_unit = F.normalize(lm_emb, p=2, dim=1)

    validate_sampler(new, args.sampler_dir, args.topn, args.delta,
                     lm_unit, args.n_check)

    results, reproduced = figure3a_study(
        new, lm_emb, args.k, args.topn, args.delta, args.seed, device,
        args.homo_nodes, drop_self_loops=not args.homo_self_loops)

    if args.delta_sweep:
        deltas = [float(d) for d in args.delta_sweep.split(",")]
        delta_sweep(new, lm_emb, args.k, args.topn, args.seed, device,
                    args.homo_nodes, deltas,
                    drop_self_loops=not args.homo_self_loops)

    section("RESULT")
    n_pass = sum(1 for _, ok in PASSED if ok)
    print(f"  assertions: {n_pass}/{len(PASSED)} passed")
    print(f"  Figure 3(a) ordering reproduced: {'YES' if reproduced else 'NO'}")
    print()
    print("  ALL ASSERTIONS PASSED" if n_pass == len(PASSED)
          else "  SOME ASSERTIONS FAILED")


if __name__ == "__main__":
    main()
