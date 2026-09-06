"""Regression tests for the FLAG bug fixes (plain asserts, no pytest).

Run with:  ./.venv/Scripts/python.exe test_fixes.py
"""
import torch

from utils import causal_loss, non_causal_loss, orthogonal_loss, ECELoss
from models import GCN, GAT, GraphSAGE
from caregnn import CAREGNN
from dga import DGA
from pmp import LASAGE_S

PASS = 0


def ok(msg):
    global PASS
    PASS += 1
    print("  PASS  " + msg)


def section(name):
    print("\n=== " + name + " ===")


# --------------------------------------------------------------------------
# FIX 1 -- orthogonal_loss (paper Eq. 9)
# --------------------------------------------------------------------------
section("FIX 1: orthogonal_loss")

a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
perp = orthogonal_loss(a, b).item()
assert abs(perp) < 1e-6, f"perpendicular should be ~0, got {perp}"
ok(f"perpendicular 1-D vectors -> {perp:.3e} (~0)")

par = orthogonal_loss(a, a).item()
anti = orthogonal_loss(a, -a).item()
assert par > 0.9, f"parallel should be large, got {par}"
assert anti > 0.9, f"anti-parallel should be large, got {anti}"
assert abs(par - anti) < 1e-6, f"parallel {par} != anti-parallel {anti}"
# The key regression: the old buggy version REWARDED anti-parallel (returned -1).
assert not (anti < par), "anti-parallel must NOT be cheaper than parallel"
ok(f"parallel {par:.6f} ~= anti-parallel {anti:.6f}, both >> perpendicular")

assert perp < par and perp < anti, "orthogonal must be the minimum"
ok("orthogonal case is the strict minimum")

# batched (N, C)
A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
B = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
bperp = orthogonal_loss(A, B)
assert bperp.dim() == 0, "batched result must be a scalar"
assert abs(bperp.item()) < 1e-6, f"batched perpendicular ~0, got {bperp.item()}"
ok(f"batched (3,2) perpendicular -> {bperp.item():.3e}")

bpar = orthogonal_loss(A, A).item()
banti = orthogonal_loss(A, -A).item()
assert abs(bpar - banti) < 1e-6 and bpar > 0.9
ok(f"batched parallel {bpar:.6f} ~= anti-parallel {banti:.6f}")

# a genuinely random batch: value must live in [0, 1]
R1, R2 = torch.randn(16, 8), torch.randn(16, 8)
rv = orthogonal_loss(R1, R2).item()
assert 0.0 <= rv <= 1.0, f"squared cosine must be in [0,1], got {rv}"
ok(f"random batched input in [0,1] -> {rv:.6f}")

# gradient actually points toward orthogonality
v = torch.tensor([1.0, 1.0], requires_grad=True)
w = torch.tensor([1.0, 1.0])
orthogonal_loss(v, w).backward()
assert v.grad is not None and torch.any(v.grad != 0), "loss must be differentiable"
ok("differentiable w.r.t. the causal embedding")


# --------------------------------------------------------------------------
# FIX 2 -- non_causal_loss (paper Eq. 8, KL(pred || uniform))
# --------------------------------------------------------------------------
section("FIX 2: non_causal_loss")

uni = torch.tensor([0.0, 0.0])
u = non_causal_loss(uni).item()
assert abs(u) < 1e-6, f"uniform prediction should give ~0, got {u}"
ok(f"uniform 1-D prediction -> {u:.3e} (~0)")

vals = []
for gap in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    v = non_causal_loss(torch.tensor([gap, 0.0])).item()
    vals.append(v)
    assert v >= -1e-7, f"KL must be non-negative, got {v} at gap={gap}"
for lo, hi in zip(vals, vals[1:]):
    assert hi > lo, f"KL must strictly increase with skew: {vals}"
ok("strictly increases with skew: " + ", ".join(f"{v:.4f}" for v in vals))
ok("non-negative for every skew level")

# max KL for 2 classes is log(2)
assert vals[-1] < torch.log(torch.tensor(2.0)).item() + 1e-6
ok(f"bounded by log(2)={torch.log(torch.tensor(2.0)).item():.4f}; extreme -> {vals[-1]:.4f}")

# batched
bu = non_causal_loss(torch.zeros(5, 2)).item()
assert abs(bu) < 1e-6, f"batched uniform -> {bu}"
ok(f"batched (5,2) uniform -> {bu:.3e}")

bskew = non_causal_loss(torch.tensor([[6.0, 0.0]] * 5)).item()
single = non_causal_loss(torch.tensor([6.0, 0.0])).item()
assert abs(bskew - single) < 1e-6, f"batch mean {bskew} != single {single}"
assert bskew > bu, "skewed batch must cost more than uniform batch"
ok(f"batched skewed {bskew:.4f} == single-sample {single:.4f} (mean over batch)")

# 3-class, batched, with num_classes passed through
b3 = non_causal_loss(torch.zeros(4, 3), num_classes=3).item()
assert abs(b3) < 1e-6, f"3-class uniform -> {b3}"
ok(f"num_classes=3 uniform (4,3) -> {b3:.3e}")

g = torch.tensor([3.0, 0.0], requires_grad=True)
non_causal_loss(g).backward()
assert g.grad is not None and torch.any(g.grad != 0)
ok("differentiable w.r.t. the non-causal logits")


# --------------------------------------------------------------------------
# causal_loss -- unchanged behaviour
# --------------------------------------------------------------------------
section("causal_loss (unchanged)")

logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
labels = torch.tensor([0, 1])
good = causal_loss(logits, labels).item()
bad = causal_loss(logits, torch.tensor([1, 0])).item()
assert good >= 0 and good < 1e-3, f"confident+correct should be ~0, got {good}"
assert bad > good, "confident+wrong must cost more than confident+correct"
ok(f"correct {good:.3e} < wrong {bad:.3f}; still cross-entropy")

ce = torch.nn.functional.cross_entropy(logits, labels).item()
assert abs(ce - good) < 1e-9
ok("matches F.cross_entropy exactly")


# --------------------------------------------------------------------------
# FIX 3 / FIX 4 -- skip connection (paper Eq. 6) on every backbone
# --------------------------------------------------------------------------
section("FIX 3/4: skip connection Z = GNN(X,A) + Linear(X)")

N, IN, OUT = 8, 16, 2
EDGES = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 0, 3],
     [1, 2, 3, 4, 5, 6, 7, 0, 4, 6]],
    dtype=torch.long,
)


def build(name, hidden):
    torch.manual_seed(0)
    if name == "GCN":
        return GCN(IN, hidden, OUT)
    if name == "GAT":
        return GAT(IN, hidden, OUT)
    if name == "GraphSAGE":
        return GraphSAGE(IN, hidden, OUT)
    if name == "CAREGNN":
        return CAREGNN(IN, hidden, OUT)
    if name == "DGA":
        return DGA(IN, hidden, OUT)
    if name == "LASAGE_S":
        return LASAGE_S(IN, hidden, OUT)
    raise ValueError(name)


MODELS = ["GCN", "GAT", "GraphSAGE", "CAREGNN", "DGA", "LASAGE_S"]

for hidden in (32, 64):
    print(f"\n-- hidden={hidden} --")
    for name in MODELS:
        torch.manual_seed(0)
        x = torch.randn(N, IN)
        model = build(name, hidden)
        model.eval()  # deterministic: no dropout noise

        with torch.no_grad():
            out = model(x, EDGES)

        assert isinstance(out, tuple) and len(out) == 2, \
            f"{name}: expected a 2-tuple, got {type(out)}"
        x32, logits = out
        assert logits.shape == (N, OUT), \
            f"{name}: logits shape {tuple(logits.shape)} != {(N, OUT)}"
        assert x32.shape[0] == N, f"{name}: x32 batch dim {tuple(x32.shape)}"
        assert torch.isfinite(logits).all(), f"{name}: non-finite logits"

        # PROOF the skip connection is live: perturb ONLY linear1 and require
        # the logits to move.  With the old code (initial_x discarded) this
        # assertion fails.
        before = logits.clone()
        model.linear1.weight.data += 1.0
        with torch.no_grad():
            _, after = model(x, EDGES)
        delta = (after - before).abs().max().item()
        assert delta > 1e-6, \
            f"{name}: linear1 perturbation did not change logits -> skip connection missing"

        # and the change is exactly the linear1 delta, i.e. a pure additive skip
        expected = (torch.ones(IN) @ x.T).unsqueeze(1).expand(N, OUT)
        assert torch.allclose(after - before, expected, atol=1e-4), \
            f"{name}: skip connection is not a clean additive Linear(X) term"

        print(f"  PASS  {name:<10} hidden={hidden:<3} x32={tuple(x32.shape)} "
              f"logits={tuple(logits.shape)} d(linear1)={delta:.4f}")
        PASS += 1

# FIX 4 explicitly: hidden != 32 must not raise UnboundLocalError
for name in ("DGA", "LASAGE_S"):
    for hidden in (7, 64, 128):
        m = build(name, hidden)
        m.eval()
        with torch.no_grad():
            h, lg = m(torch.randn(N, IN), EDGES)
        assert h.shape == (N, hidden), f"{name}: x32 should be the hidden repr, got {tuple(h.shape)}"
        assert lg.shape == (N, OUT)
    ok(f"{name}: no UnboundLocalError for hidden in (7, 64, 128); x32 tracks hidden dim")


# --------------------------------------------------------------------------
# FIX 6 -- ECELoss
# --------------------------------------------------------------------------
section("FIX 6: ECELoss")

torch.manual_seed(0)
conf_right = torch.Tensor([[8.0, 0.0]] * 50)
lab_right = torch.LongTensor([0] * 50)
e_good = ECELoss(conf_right, lab_right)
assert isinstance(e_good, float), f"must return a plain float, got {type(e_good)}"
assert not isinstance(e_good, torch.Tensor)
ok(f"returns a plain Python float ({e_good!r})")

assert 0.0 <= e_good <= 1.0, f"out of range: {e_good}"
assert e_good < 0.01, f"confident+correct should be well calibrated, got {e_good}"
ok(f"confident and correct -> small ECE {e_good:.6f}")

e_bad = ECELoss(conf_right, torch.LongTensor([1] * 50))
assert 0.0 <= e_bad <= 1.0, f"out of range: {e_bad}"
assert e_bad > 0.9, f"confident and wrong should give large ECE, got {e_bad}"
ok(f"confident and wrong -> large ECE {e_bad:.6f}")
assert e_bad > e_good
ok("miscalibrated > calibrated")

# 50% confident, 50% accurate -> near-perfect calibration
half = torch.Tensor([[0.4055, 0.0]] * 100)  # softmax ~ [0.6, 0.4]
lab_half = torch.LongTensor([0] * 60 + [1] * 40)
e_half = ECELoss(half, lab_half)
assert 0.0 <= e_half <= 1.0 and e_half < 0.02, f"expected small, got {e_half}"
ok(f"60% confident / 60% accurate -> ECE {e_half:.6f} (~0)")

# random logits stay in range, and n_bins is honoured
rnd = torch.randn(200, 4)
rl = torch.randint(0, 4, (200,))
for nb in (1, 5, 15, 50):
    v = ECELoss(rnd, rl, n_bins=nb)
    assert isinstance(v, float) and 0.0 <= v <= 1.0, f"n_bins={nb} -> {v}"
ok("random 4-class input in [0,1] for n_bins in (1, 5, 15, 50)")

import statistics
assert isinstance(statistics.mean([ECELoss(rnd, rl), ECELoss(conf_right, lab_right)]), float)
ok("result is consumable by statistics.mean (as test.py does)")


print(f"\n{'='*60}\nALL {PASS} CHECKS PASSED\n{'='*60}")
