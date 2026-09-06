# FLAG — reproduction of "Fraud Detection with LLM-enhanced Graph Neural Network"

A reproduction of **FLAG** (Yang et al., KDD '25). The paper detects fraud on text-rich
graphs by combining three ideas:

1. **Semantic similarity neighbour sampling** (§3.1, Eq. 3–4) — build each node's ego-subgraph
   from the *k*-hop neighbours whose text is most semantically similar to it, which both shrinks
   the LLM input and filters out camouflaged neighbours.
2. **LLM-based node enhancement** (§3.2) — an LLM distils each node's raw text into
   *discriminative text*, encoded by a frozen LM and fed to a **skip-GNN**,
   `Z = GNN(X, A) + Linear(X)` (Eq. 6).
3. **Fine-tuning** (§3.3, Eq. 7–10) — an auxiliary *residual text* task with three losses
   (BCE, KL-to-uniform, orthogonality), optimised by alternating LLM and GNN updates.

> **This is a work in progress.** The data pipeline is built and validated; the training and
> evaluation scripts are **not yet rewired** to it. Read [Current status](#current-status)
> before running anything.

---

## Current status

| Stage | Script | State |
|---|---|---|
| Minority downsampling to 1:10 (§4.1.1) | `preprocess.py` | ✅ works, validated |
| Semantic similarity sampler (§3.1, Eq. 3–4) | `sampler.py` | ✅ works, validated |
| Pipeline validation + Figure 3(a) check | `validate_data.py` | ✅ works, 44/44 assertions |
| Loss functions (Eq. 8, 9) and skip-GNN (Eq. 6) | `utils.py`, `models.py`, `caregnn.py`, `dga.py`, `pmp.py` | ✅ fixed, 38/38 tests |
| LLM discriminative-text generation | `chat.py`, `chat1.py` | ❌ **broken** — see [Known issues](#known-issues) |
| Legacy embedding/split script | `encode.py` | ⛔ **deprecated** — superseded, do not run |
| GNN training / evaluation | `test.py`, `test_dual.py` | ❌ **broken** — not yet rewired |
| LLM fine-tuning (FLAG\*) | `train.py`, `train1.py` | ❌ **broken** — see [FLAG\*](#a-note-on-flag-the-fine-tuning-stage) |

**You can currently run steps 1–3 end to end.** Steps 4–6 are the remaining work.

---

## Environment

- **Python 3.11.x** (recommended)
- **OS**: Windows 10/11 or Linux
- **GPU**: NVIDIA CUDA card. The data pipeline runs fine on CPU; the LLM stage wants ≥16 GB VRAM.

### 1. Create a virtual environment

```powershell
cd "D:\Manohar\Uni\Major Project\Codes\FLAG"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install dependencies

GPU (CUDA 12.4):

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install torch-geometric
python -m pip install -r requirements.txt
```

CPU only — swap the first two URLs to the `cpu` variants:

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip install torch-geometric
python -m pip install -r requirements.txt
```

`torch-scatter` must match your exact torch build; it is only needed by `dga.py`.

### 3. Verify

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_geometric, torch_scatter; print('pyg ok')"
```

### Windows encoding note

Dataset text contains emoji and non-cp1252 characters. Any script that prints node text will
raise `UnicodeEncodeError` on Windows unless you set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

```bash
export PYTHONIOENCODING=utf-8   # bash
```

---

## Datasets

Two graphs from GLBench, placed at the project root. **Only these two files are inputs** —
everything else is generated.

```text
FLAG/
  Instagram/instagram.pt      # 11,339 nodes · 155,349 edges · 7224 normal / 4115 commercial
  Reddit/reddit.pt            # 33,434 nodes · 302,876 edges · 16717 normal / 16717 popular
```

Each is a PyG `Data` object with `x` (4096-d), `y`, `edge_index`, `raw_texts` (list of str),
train/val/test masks, and `label_name`. **Label 1 is the minority class** in both
(`Popular Users` / `Commercial Users`), per §4.1.1.

`Instagram/` and `Reddit/` are in `.gitignore` — datasets and generated `.pt` files are never
committed.

---

## The pipeline, A–Z

Run these in order. Instagram is the default everywhere, so bare invocations target Instagram.
Instagram is the faster dataset — validate there first, then repeat for Reddit.

### Step 1 — Downsample to a 1:10 class ratio (§4.1.1)

The raw graphs are near-balanced, but FLAG is an *imbalanced* fraud-detection benchmark. This
keeps every majority node and randomly retains `n_majority / 10` minority nodes, rebuilds the
induced subgraph with relabelled edges, and regenerates stratified splits.

```bash
python preprocess.py --data Instagram/instagram.pt --out Instagram/instagram_1to10.pt
python preprocess.py --data Reddit/reddit.pt       --out Reddit/reddit_1to10.pt
```

| Option | Default | Meaning |
|---|---|---|
| `--ratio` | `10.0` | majority : minority target ratio |
| `--seed` | `0` | seeds both the minority draw and the split |
| `--train-ratio` / `--val-ratio` | `0.1` / `0.1` | remainder becomes test |

Produces `Instagram/instagram_1to10.pt` (7,946 nodes, ratio 1:10.01) and
`Reddit/reddit_1to10.pt` (18,389 nodes, ratio 1:9.998). Takes seconds.

> The original masks are **discarded and regenerated** — they index the pre-downsampling node
> ordering and are invalid afterwards. The `one_shot_*` / `three_shot_*` / `five_shot_*` masks are
> dropped for the same reason. An `orig_index` field maps each new node back to its original id.

### Step 2 — Semantic similarity neighbour sampling (§3.1, Eq. 3–4)

For every node: take its *k*-hop neighbourhood, rank candidates by cosine similarity between
frozen-LM text embeddings (Eq. 3), keep those above `--delta` and take the top-N (Eq. 4), then
emit the induced subgraph.

```bash
python sampler.py --data Instagram/instagram_1to10.pt \
                  --out-dir Instagram/ss_d0_n10_k2 \
                  --emb-cache Instagram/lm_embeddings.pt

python sampler.py --data Reddit/reddit_1to10.pt \
                  --out-dir Reddit/ss_d0_n10_k2 \
                  --emb-cache Reddit/lm_embeddings.pt
```

| Option | Default | Paper |
|---|---|---|
| `--k` | `2` | 2-hop subgraphs, §4.1.2 |
| `--topn` | `10` | top-10 neighbours, §4.1.2 |
| `--delta` | `0.0` | similarity threshold δ, §4.5 |
| `--strategy` | `ss` | `ss` / `ss_star` / `fs` / `rs` / `ns` for the Figure 3(a) comparison |
| `--emb-cache` | — | LM embeddings are computed once and reused |

Writes `train_sampler.pt`, `val_sampler.pt`, `test_sampler.pt` — each a Python list of `Data`
objects, **one per target node**. Roughly 30 s for Instagram, 45 s for Reddit including LM encoding.

#### Batch data contract

Downstream code depends on this exactly:

| Field | Meaning |
|---|---|
| `subset` | **Global** node ids, with the target node **first** |
| `central` | Global id of the target node; always equals `subset[0]` |
| `edge_index` | **Local** indices `0 .. len(subset)-1`, to be used with `data.x[batch.subset]` |
| `raw_texts` | Node texts, aligned with `subset` |

So `data.x[batch.subset]` gives the local feature rows, `subset == batch.central` locates the
target's row, and `data.y[batch.central]` is its label. Nodes with no *k*-hop neighbour yield a
single-node subgraph with an empty `(2, 0)` `edge_index`.

### Step 3 — Validate (and reproduce Figure 3a)

```bash
python validate_data.py --orig Instagram/instagram.pt \
                        --data Instagram/instagram_1to10.pt \
                        --sampler-dir Instagram/ss_d0_n10_k2 \
                        --emb-cache Instagram/lm_embeddings.pt
```

44 assertions covering the class ratio, edge relabelling, mask disjointness, `x`/`y`/`raw_texts`
alignment against the original file, and — for the sampler — that `central ∈ subset`,
`edge_index` is local, similarities respect δ, and **every batch edge is a real edge of the source
graph**. It then computes average subgraph edge homophily (Eq. 5) across all five sampling
strategies to test the paper's Figure 3(a) ordering. Add `--delta-sweep "0.0,0.1,0.2,0.3,0.4,0.5"`
to see how δ trades homophily against connectivity. Takes a few minutes.

### Step 4 — Generate discriminative text ❌ *not working*

`chat.py` (Reddit) and `chat1.py` (Instagram) are meant to run the LLM over each subgraph and
produce discriminative text. Both are currently broken — see [Known issues](#known-issues).

### Step 5 — Encode and train ❌ *not working*

`test.py` / `test_dual.py` train and evaluate the GNN backbones. Not yet rewired to the new
sampler output.

### Step 6 — Fine-tune the LLM ❌ *not working*

`train.py` / `train1.py`. See [FLAG\*](#a-note-on-flag-the-fine-tuning-stage).

---

## Running the tests

```bash
python test_fixes.py      # 38 checks: losses (Eq. 8, 9), skip connections (Eq. 6), ECELoss
python validate_data.py   # 44 checks: preprocessing + sampler contract + Figure 3(a)
```

`test_fixes.py` uses synthetic tensors only and needs no dataset. It includes a regression test
that mutating `linear1`'s weights must change the output logits — which fails if anyone drops the
Eq. 6 skip connection again.

---

## Known issues

Ordered by what blocks progress first.

### `chat1.py` — SyntaxError, will not import

```python
def generate_summary(batch, llm, tokenizer, max_retries=1, dataset_type):   # line 63
```

A non-default parameter cannot follow a default one. Move `dataset_type` before `max_retries`.

### `chat.py` — missing argument in the test loop

Line 117 calls `generate_summary(batch, llm, tokenizer)` without the now-required
`dataset_type`, so the script crashes with `TypeError` when it reaches the test split. The train
and val loops pass it correctly.

### `chat.py` / `chat1.py` — stale dataset paths

Both still load `Reddit/0_10_0/train_sampler1.pt` / `Instagram/0_10_0/train_sampler.pt`, which no
longer exist. They should read `<Dataset>/ss_d0_n10_k2/*_sampler.pt` and the `*_1to10.pt` graph.
`chat1.py` also hardcodes `gemma-2-9b-it`, a gated model.

### `chat.py` / `chat1.py` — residual text is never generated

Both define `common_prompt` (the §3.3 / Table 2 residual-text prompt) but never call the LLM with
it, so the residual branch the fine-tuning losses need has no data source.

### `test.py` / `test_dual.py` — text embeddings are computed then ignored

`text_embeddings` is selected from `batch.unique_embeddings`, but the forward pass calls
`model(data.x[batch.subset], edge_index)` — the raw 4096-d shallow features. Every configuration
therefore evaluates the *baseline*, so "+FLAG" numbers from this script are meaningless.

### `test.py` — undefined names at import

Uses `get_device()` and `safe_torch_load()` without importing them from `utils`, and loads
non-existent paths (`Reddit/reddit1.pt`, `Reddit/embeddings1.pt`).

### `train.py` / `train1.py` — accumulation and optimiser bugs

The batch counter `i` is shadowed by an inner `for i in range(len(batch.subset))` loop, so the
gradient-accumulation condition `(i + 1) % 10 == 0` no longer tracks batches. `train_gnn` also
calls `optimizer.step()` (the LLM optimiser) where `gnn_optimizer.step()` was meant.

### `encode.py` — deprecated, do not run

Superseded by `preprocess.py` + `sampler.py`. It contains a path bug
(`args.path + '.pt'` yields `"Instagram/.pt"`), strips generated-text prefixes inconsistently
(`[2:]` for train vs `[3:]` for val/test), does no graph sampling whatsoever, and writes batches
with no `edge_index` and no `central`. **Its output files (`instagram_train.pt` and friends) use
pre-downsampling node indices and are invalid.** Delete any that exist.

---

## A note on FLAG\* (the fine-tuning stage)

FLAG's fine-tuning (§3.3) backpropagates into the LLM through *generated text*, which is discrete
and therefore **not differentiable**. The paper does not explain how gradients reach the LLM.

In `train.py` the chain is `generate()` → decode → `str` → `SentenceTransformer.encode()` (NumPy)
→ GNN → loss. Gradients reach only the GNN; the LoRA parameters get `grad=None`, so
`optimizer.step()` silently updates nothing. **FLAG\* can never differ from FLAG as written.**

**FLAG\* is deliberately deferred.** Reproduce `baseline`, `+text` and zero-shot `+FLAG` first —
they need no LLM training and hold most of the paper's claimed gain. By the paper's own Table 4,
of the headline +3.14% F1, zero-shot FLAG contributes +1.58% over `+text` while FLAG\* adds only
+0.84% (and is *worse* than FLAG in 3 of 28 cells).

If revisited, the practical approach is **rejection sampling + SFT**: generate *k* candidate texts,
score each with the frozen skip-GNN using Eq. 7–9, and supervise-fine-tune on the winners.
REINFORCE is the theoretically correct alternative. Gumbel-softmax is not viable here — the LLM
and the frozen LM use different tokenizers, so soft tokens cannot pass between them.

---

## Reproduction findings so far

Measured on the pipeline above; see the paper's Figure 3(a) and §4.5.

**Figure 3(a), Reddit** (the paper's dataset for that figure) — subgraph edge homophily, Eq. 5:

| Strategy | Homophily | Paper's claim | Reproduced |
|---|---|---|---|
| NS — no sampling | 0.6581 | worst | ✅ |
| RS — random | 0.7257 | NS < RS | ✅ |
| FS — shallow features | 0.7402 | RS < FS | ✅ |
| SS\* — text, no threshold | 0.7377 | FS < SS\* | ❌ by 0.0025 |
| **SS — text + threshold** | **0.7411** | best overall | ✅ |

Four of five relations hold, including the central claim that SS wins. The one miss is
noise-level.

**The δ=0 threshold is a no-op on Instagram.** Only 0.18% of Instagram edges have a negative
cosine similarity, versus 11.52% on Reddit — so `sim ≥ 0` filters essentially nothing there and
SS and SS\* become the same method. The cause is text length: Instagram bios average 110
characters against Reddit's 811. The paper's claim that thresholding helps cannot be tested on
Instagram at the published δ=0.

**Higher δ raises homophily but destroys the graph** (Reddit): homophily climbs from 0.7411 at
δ=0 to 0.8828 at δ=0.4, but average subgraph size collapses from 6.82 nodes to 1.76 and 85% of
target nodes are left with *no neighbours at all*. This is an independent structural confirmation
of the paper's δ=0 choice in §4.5, which the authors justified with downstream AUC.

**The downsampled Reddit graph is genuinely heterophilous** — every strategy scores below the
0.8347 chance floor — which is the neighbourhood-camouflage phenomenon FLAG targets. Instagram
scores *above* chance, so it is not camouflaged in the same sense. The paper does not discuss this
difference between the two benchmarks.

---

## Deviations from the paper

| Paper | This repo | Why |
|---|---|---|
| Gemma-2-9b-it as LLM | `microsoft/Phi-3.5-mini-instruct` | Gemma is gated; no access |
| Sentence-BERT as frozen LM | `all-MiniLM-L6-v2` (384-d) | A Sentence-BERT model; the repo's existing choice |
| Hidden size 64 (§4.1.2) | 32 in several scripts | Not yet aligned |
| 25 runs (5 seeds × 5 inits) | 5 runs | Not yet aligned |
| Eq. 9 on raw representations | Squared **normalised** dot product | Raw dot products are trivially minimised by shrinking ‖Z‖ → 0; normalising keeps the minimum at true orthogonality |
| Split ratios unstated | 10% / 10% / 80%, stratified | Matches the source GLBench files' own proportions |

---

## Repository map

| File | Role |
|---|---|
| `preprocess.py` | §4.1.1 — 1:10 downsampling, induced subgraph, stratified splits |
| `sampler.py` | §3.1 Eq. 3–4 — semantic similarity neighbour sampling |
| `validate_data.py` | Pipeline assertions + Figure 3(a) homophily comparison |
| `test_fixes.py` | Unit tests for the loss and skip-GNN fixes |
| `utils.py` | Losses (Eq. 7–9), `ECELoss`, `FocalLoss`, device/loading helpers |
| `models.py` | GCN, GAT, GraphSAGE, `DualGNN` (§3.4 attention fusion), `CaGCN` |
| `geniepath.py`, `bwgnn.py`, `caregnn.py`, `dga.py`, `pmp.py` | Baseline GNN backbones (§4.1.2) |
| `chat.py`, `chat1.py` | LLM discriminative-text generation (Reddit / Instagram) |
| `test.py`, `test_dual.py` | GNN training + evaluation (single / dual-branch) |
| `train.py`, `train1.py` | LLM fine-tuning loop (§3.3) |
| `encode.py` | **Deprecated** — superseded by `preprocess.py` + `sampler.py` |

---

## Running on Vast.ai

An RTX 3090/4090 (24 GB) is a practical starting point for `Phi-3.5-mini-instruct`.

```bash
git clone <your-repository-url> FLAG
cd FLAG
chmod +x setup_vast.sh
./setup_vast.sh
source .venv/bin/activate
```

Datasets are gitignored, so upload `Instagram/instagram.pt` and `Reddit/reddit.pt` separately,
then run Steps 1–3 on the instance — they are cheap and avoid transferring the generated files.

Download results before destroying the instance:

```bash
tar -czf flag-results.tar.gz Reddit/ss_d0_n10_k2/*.pt Instagram/ss_d0_n10_k2/*.pt
```

Stop or destroy the instance afterwards to avoid further charges.

---

## Reference

Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang, Chuan Shi.
*FLAG: Fraud Detection with LLM-enhanced Graph Neural Network.* KDD '25.
<https://doi.org/10.1145/3711896.3737220>
