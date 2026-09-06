"""FLAG preprocessing (paper section 4.1.1).

Downsamples the minority (fraud / commercial / popular) class so that the
minority:majority ratio becomes about 1:`--ratio` (default 1:10), builds the
induced subgraph over the surviving nodes, and regenerates seeded *stratified*
train/val/test masks (the masks shipped with the source file refer to the old
node indexing and are meaningless after node removal).

Usage
-----
python preprocess.py --data Instagram/instagram.pt --out Instagram/instagram_1to10.pt
"""

import argparse
import os

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


# The minority / positive class is label 1 in both FLAG datasets
# ('Commercial Users' for Instagram, 'Popular Users' for Reddit).
MINORITY_LABEL = 1

SHOT_MASK_PREFIXES = ("one_shot", "three_shot", "five_shot")


def safe_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path)


def stratified_split(y, train_ratio, val_ratio, generator):
    """Seeded stratified split; returns three bool masks over ``len(y)`` nodes.

    Every class keeps (approximately) the same train/val/test proportions, so
    the global ~1:10 imbalance is preserved inside each split.
    """
    num_nodes = y.numel()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for cls in torch.unique(y).tolist():
        cls_idx = (y == cls).nonzero(as_tuple=False).view(-1)
        perm = torch.randperm(cls_idx.numel(), generator=generator)
        cls_idx = cls_idx[perm]

        n_cls = cls_idx.numel()
        n_train = int(round(train_ratio * n_cls))
        n_val = int(round(val_ratio * n_cls))
        # guard against pathological tiny classes
        n_train = min(n_train, n_cls)
        n_val = min(n_val, n_cls - n_train)

        train_mask[cls_idx[:n_train]] = True
        val_mask[cls_idx[n_train:n_train + n_val]] = True
        test_mask[cls_idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def describe_split(name, mask, y):
    idx = mask.nonzero(as_tuple=False).view(-1)
    n0 = int((y[idx] == 0).sum())
    n1 = int((y[idx] == 1).sum())
    ratio = (n0 / n1) if n1 else float("inf")
    print(f"  {name:<5} n={idx.numel():>6}  class0={n0:>6}  class1={n1:>6}  "
          f"maj:min = {ratio:.2f}:1")


def main():
    parser = argparse.ArgumentParser(description="FLAG 1:10 downsampling preprocessor")
    parser.add_argument("--data", type=str, default="Instagram/instagram.pt")
    parser.add_argument("--out", type=str, default="Instagram/instagram_1to10.pt")
    parser.add_argument("--ratio", type=float, default=10.0,
                        help="majority:minority ratio to target (default 10 -> 1:10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1.0")

    data = safe_load(args.data)

    old_n = int(data.y.numel())
    old_e = int(data.edge_index.size(1))
    y_old = data.y

    maj_idx = (y_old != MINORITY_LABEL).nonzero(as_tuple=False).view(-1)
    min_idx = (y_old == MINORITY_LABEL).nonzero(as_tuple=False).view(-1)

    print("=" * 72)
    print("FLAG preprocessing - minority downsampling (paper sec. 4.1.1)")
    print("=" * 72)
    print(f"input   : {args.data}")
    print(f"BEFORE  : nodes={old_n}  edges={old_e}")
    print(f"          class0(majority)={maj_idx.numel()}  "
          f"class1(minority)={min_idx.numel()}  "
          f"maj:min = {maj_idx.numel() / max(min_idx.numel(), 1):.2f}:1")
    if hasattr(data, "label_name"):
        print(f"          label_name={list(data.label_name)}")

    # ---- select the nodes to keep -------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    n_keep_min = int(round(maj_idx.numel() / args.ratio))
    n_keep_min = min(n_keep_min, min_idx.numel())
    perm = torch.randperm(min_idx.numel(), generator=g)
    kept_min = min_idx[perm[:n_keep_min]]

    subset_idx = torch.cat([maj_idx, kept_min]).sort().values  # ascending, stable
    new_n = int(subset_idx.numel())

    # ---- induced subgraph ---------------------------------------------
    new_edge_index, _ = subgraph(subset_idx, data.edge_index,
                                 relabel_nodes=True, num_nodes=old_n)

    # ---- re-index node-level tensors / lists --------------------------
    new_x = data.x[subset_idx].clone()
    new_y = data.y[subset_idx].clone()
    new_texts = [data.raw_texts[int(i)] for i in subset_idx.tolist()]

    # ---- regenerate stratified masks ----------------------------------
    g_split = torch.Generator().manual_seed(args.seed)
    train_mask, val_mask, test_mask = stratified_split(
        new_y, args.train_ratio, args.val_ratio, g_split)

    out = Data(
        x=new_x,
        y=new_y,
        edge_index=new_edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    out.raw_texts = new_texts
    out.label_name = list(data.label_name) if hasattr(data, "label_name") else None
    # index map back into the source file, so alignment can be audited later
    out.orig_index = subset_idx.clone()
    out.num_nodes = new_n

    dropped = [k for k in data.keys() if k.startswith(SHOT_MASK_PREFIXES)]

    n0 = int((new_y == 0).sum())
    n1 = int((new_y == 1).sum())
    print(f"AFTER   : nodes={new_n}  edges={int(new_edge_index.size(1))}")
    print(f"          class0(majority)={n0}  class1(minority)={n1}  "
          f"maj:min = {n0 / max(n1, 1):.3f}:1  (requested {args.ratio}:1)")
    print(f"          removed {old_n - new_n} nodes and "
          f"{old_e - int(new_edge_index.size(1))} edges")
    print("splits  : (seeded stratified, "
          f"{args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1 - args.train_ratio - args.val_ratio:.0%})")
    describe_split("train", train_mask, new_y)
    describe_split("val", val_mask, new_y)
    describe_split("test", test_mask, new_y)

    if dropped:
        print("NOTE    : dropped few-shot masks " + ", ".join(sorted(dropped)))
        print("          They index the ORIGINAL node ordering and cannot be")
        print("          meaningfully recomputed after node removal, so they are")
        print("          intentionally not carried over.")
    print(f"NOTE    : original train/val/test masks were discarded and "
          f"regenerated (they were invalid after re-indexing).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"saved   : {args.out}")
    print(out)


if __name__ == "__main__":
    main()
