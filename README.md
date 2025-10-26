**PuzzlePilot** - AI Animal Puzzle Solver

# Ensemble + Hungarian + Edge-tie-breaking
(using a per‑tile classifier, a pairwise relative model, a transformer/permutation model,
Hungarian assignment for initialization, and a pairwise/edge‑based local refinement).
--------------

**Contents**
- Full pipeline (step‑by‑step)
- Training details for each model
  - Classifier (per‑tile)
  - Pairwise (relative)
  - Transformer / permutation (global)
- Calibration & log‑space fusion (with pseudo‑code)
- Hungarian initialization
- Pairwise and edge compatibility computations
- Local search / swap refinement (pseudo‑code + efficient delta computation)
- Tie‑breaking strategies using edge scores (two options)
- Worked numeric toy example
- Project file layout (suggested)
- Usage examples (train / validate / solve)
- Hyperparameters, evaluation metrics & diagnostics
- Implementation tips, speed & memory considerations
- Reproducibility & checkpoints
- References

---------------

**Overview:**
The solver trains three model types independently:
1. Classifier — per‑tile absolute position predictor (outputs 9 logits per tile).
2. Pairwise model — predicts relative relationships/adjacency between two tiles.
3. Transformer / permutation model — uses global context across all 9 tiles to predict per‑tile position logits.

**At inference**:
- Calibrate and fuse per‑tile logits from classifier and transformer into a 9×9 combined score matrix.
- Run Hungarian (linear assignment) on the negative combined scores to get an initial permutation.
- Compute pairwise compatibility (from the pairwise model) and edge matching scores (border correlations).
- Run a local swap refinement (or simulated annealing) using fused per‑tile scores + pairwise + edge scores to improve the permutation while keeping the permutation constraint.
- Optionally ensemble over model weightings or multiple runs.

---------------

**Full pipeline**
1. Train the three models independently (see “Training details”).
2. Precompute / store:
   - Classifier per‑tile logits L_class (9 tiles × 9 pos)
   - Transformer per‑tile logits L_trans (9 × 9)
   - Pairwise probabilities P_pair(i, j, dir) (tile pairs × directions)
   - Edge similarity scores Edge(i, j, dir) (computed from image borders)
3. Calibration & fusion:
   - Convert logits to log‑probabilities, temperature‑scale if needed.
   - Weighted fusion in log space: combined_logp = w_c * logp_class + w_t * logp_trans
4. Hungarian initialization:
   - cost = -combined_logp
   - assignment = linear_sum_assignment(cost)
5. Compute pairwise & edge totals for assignment.
6. Local refinement:
   - Evaluate swaps or block moves that change local adjacency; accept if they improve GlobalScore.
   - Optionally use simulated annealing to escape local maxima.
7. Tie‑breaking / secondary Hungarian:
   - Option A: approximate pairwise/edge contribution to per‑tile costs and re-run Hungarian.
   - Option B: two‑stage Hungarian that augments cost with edge_contrib when confidence is low.
8. Optionally repeat with different ensemble weights or average multiple permutations and select best.

--------------------

1) Train each model

Classifier (per‑tile)
- Input: single tile image (resized consistently, e.g., 96×96 or 128×128).
- Output: 9 logits (one for each absolute grid position). Softmax yields per‑tile probabilities.
- Loss: CrossEntropy between true position (0..8) and prediction.
- Backbone: ResNet / EfficientNet / small ViT.
- Augmentation: preserve orientation (no rotations unless labels adjusted). Use color jitter, brightness, slight translation, cutout.
- Validation: per‑tile accuracy AND puzzle accuracy (assemble validation puzzles with Hungarian).
- Save: model + validation performance & per‑tile logits on a held‑out set.

Pairwise model (relative relationships)
- Input: pair of tiles (tile_i, tile_j). Siamese shared backbone recommended.
- Output: multi‑class relative relation, e.g. classes: {top-left, top, top-right, left, right, bottom-left, bottom, bottom-right, none} OR {up, down, left, right, none} for adjacency.
- Loss: CrossEntropy on true relation.
- Optionally pretrain backbone from classifier weights (fine‑tune).
- Store pairwise output as P_pair[i, j, dir] where dir can be one of 4 or 8 relative directions.

Transformer / permutation model (global)
- Input: all 9 tiles as a sequence. Use per‑tile encoder → transformer encoder.
- Output: per‑tile logits across 9 positions (9×9) OR autoregressive permutation (less common for 9 tiles).
- Loss: cross‑entropy per tile. You can train with a permutation‑aware loss (Sinkhorn / Relaxed Hungarian) to directly optimize permutation consistency.
- Benefit: learns global context (object parts spanning tiles).

Validation strategy for all models
- Use held‑out puzzles.
- For puzzle accuracy (PRA): convert per‑tile logits from classifier/transformer to costs and run Hungarian to form a puzzle. PRA is fraction of puzzles solved exactly or fraction of tiles correct.
- For pairwise: evaluate accuracy on pair relations and adjacency ROC/AUC.

---

2) Calibration & combining per‑tile logits

Purpose: different models have different confidence calibration — fuse in log‑space.

Pseudo‑code (calibration + fusion):
```python
# L_class, L_trans: shape (N_tiles=9, N_pos=9) raw logits
def logprob_from_logits(logits, temperature=1.0):
    L = logits / temperature
    logp = L - logsumexp(L, axis=1, keepdims=True)
    return logp

logp_class = logprob_from_logits(L_class, T_class)
logp_trans  = logprob_from_logits(L_trans,  T_trans)

# weighted fusion in log-space
combined_logp = w_c * logp_class + w_t * logp_trans
# If you have more per-tile models, include them similarly.
# Convert to cost for Hungarian
cost_matrix = -combined_logp  # Hungarian minimizes cost
```

Temperature tuning and weights
- Tune (T_class, T_trans) and (w_c, w_t) on validation set to maximize PRA.
- Use grid search or coordinate descent.

Notes about pairwise integration
- Pairwise model outputs are not naturally per‑tile, so they are integrated during refinement or approximated into per‑tile scores (see Tie‑breaking Option A).

---

3) Use Hungarian to get an initial permutation

Python example using SciPy:
```python
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# row_ind = [0..8], col_ind is assigned positions. Build assignment A where A[row] = col.
assignment = {tile: pos for tile,pos in zip(row_ind, col_ind)}
```

This enforces a valid permutation (one tile per position and vice versa).

---

4) Compute pairwise and edge compatibility matrices

Pairwise compatibility
- For every ordered pair (i, j) and direction d, use pairwise model probabilities:
  P_pair[i, j, d] = probability tile j is in direction d relative to tile i.
- For any assignment A (tile→pos), we compute:
  pair_total_score(A) = sum over all adjacent grid neighbor pairs (p,q) of P_pair[tile_at_p, tile_at_q, dir_of_q_from_p]
- Efficient precompute: keep P_pair in memory as a [9, 9, D] tensor.

Edge matching matrix
- For every ordered pair (i, j) and direction d, compute Edge[i, j, d] = similarity between tile i's border (towards d) and tile j's matching border.
- Methods:
  - Normalized cross correlation (NCC) on 1D border pixel vectors.
  - Sum of squared differences (SSD), inverted and normalized to [0,1].
  - Small CNN trained to score border compatibility.
- MPI-friendly: computing borders for all pairs is cheap (9×9×D).

Edge_total_score(A) = sum over adjacent positions (p,q) of Edge[tile_at_p, tile_at_q, dir_of_q_from_p].

Normalization:
- Normalize P_pair and Edge scores to comparable ranges before combining (e.g., scale to mean 0 and std 1 on validation, or to [0,1]).

---

5) Local search / swap refinement (core idea)

Global objective to maximize:
GlobalScore(A) = alpha * sum_per_tile_combined_logp(A)
               + beta  * pair_total_score(A)
               + gamma * edge_total_score(A)

Where:
- sum_per_tile_combined_logp(A) = sum_t combined_logp[t, A[t]]
- alpha, beta, gamma tuned on validation.

We seek to maximize GlobalScore while preserving permutation constraints.

Efficient swap evaluation
- When swapping tiles i and j at positions p and q, only local adjacency terms around p and q change.
- Precompute neighbors for each position (up,down,left,right).
- Compute delta in O(1) by summing differences for:
  - per_tile terms: combined_logp[i,p] + combined_logp[j,q] before vs combined_logp[i,q] + combined_logp[j,p] after
  - adjacent pairs that involve p or q: recompute their pairwise + edge contributions.

Pseudo‑code (greedy hill climbing):
```python
def local_refinement(initial_assignment, combined_logp, P_pair, Edge, alpha, beta, gamma, max_iters=200):
    A = initial_assignment.copy()
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        tile pairs or random subset
        for i in range(9):
            for j in range(i+1, 9):
                p, q = A[i], A[j]  # positions of tiles i and j
                delta = delta_score_swap(i, j, p, q, A, combined_logp, P_pair, Edge, alpha, beta, gamma)
                if delta > 0:
                    A[i], A[j] = q, p
                    improved = True
to avoid order bias
    return A
```

Simulated annealing variant:
- Accept swap with probability 1 if delta > 0 else exp(delta / T).
- Decrease T across iterations.

Termination:
- Stop after a full pass with no improvements, or after max_iters swaps.

Important implementation note:
- Compute delta_score_swap by enumerating positions affected: p and q plus their neighbors (up to 10 adjacency pairs), not the entire board.

---

6) Tie‑breaking with edge / pairwise scores (detailed)

Option A — Soft approximate pairwise into per‑tile costs (cheap)
- Compute approx_pair_contrib[i,p] ≈ sum_j max_q P_pair[i,j,dir_between(p,q)] * p_prior_j(q)
- p_prior_j(q) is marginal probability tile j is at position q (from classifier/transformer softmax).
- This yields a 9×9 matrix approx_pair_contrib which can be added to combined_logp before Hungarian.

Pseudo‑code:
```python
# p_prior: shape (9_tiles, 9_pos) (softmax of per-tile logits)
approx_pair_contrib = np.zeros((9,9))
for i in range(9):
    for p in range(9):
        s = 0.0
        for j in range(9):
            if i == j: continue
            # sum over neighbor positions q for p; or max over q
            for q in neighbor_positions_of(p):
                dir_pq = dir_between(p, q)
                # either take expectation or max:
                s += P_pair[i,j,dir_pq] * p_prior[j,q]
        approx_pair_contrib[i,p] = s
# Add to fused log-probabilities (after scaling)
combined_logp_augmented = combined_logp + lambda_pair * approx_pair_contrib
```

Option B — Two-stage Hungarian with edge tie-breaker
- Run Hungarian on combined_logp → A0.
- Compute a confidence metric (e.g., sum of assigned probabilities, or difference between best and second-best cost).
- If confidence low:
  - Compute edge_contrib[i,p] = sum_j (Edge[i, j, dir] * p_prior_j(q))
  - new_combined = combined_logp + lambda_edge * edge_contrib
  - rerun Hungarian.

Tune lambda_pair / lambda_edge on validation.

---------------

Hungarian minimizes cost → assignment:
- tile0 → pos0 (0.32)
- tile1 → pos1 (0.42)
- tile2 → pos2 (0.14)

Suppose Hungarian misassigns tile1 and tile2 in a different run; local search using pairwise & edge scores can swap them if that increases global score.

---

8) **Modular code structure**
- README.md (this file)
- requirements.txt
- data_loader.py — PyTorch Dataset & DataLoader helpers (tile splitting, labels)
- models.py — TileClassifier, PairwiseModel, TransformerSolver
- train_classifier.py
- train_pairwise.py
- train_transformer.py
- utils.py — image utilities, border extraction, edge metrics, neighbor functions, evaluation
- solver.py — fusion, Hungarian init, pairwise & edge scoring, local refinement
- inference.py — loads models & runs full pipeline on a scrambled image
- experiments/ — hyperparameter configs & checkpoints
- notebooks/ — visualization and debugging notebooks
- scripts/ — helper scripts (compute_all_edges.py, compute_pair_probs.py)

Example requirements:
```text
torch>=1.10
torchvision
numpy
scipy
tqdm
Pillow
opencv-python
scikit-image
matplotlib
einops
transformers  # optional if using transformer encoder helpers
```

---

9) **Usage examples**

Train classifier (example):
```bash
python train_classifier.py --data /path/to/dataset --epochs 40 --batch-size 128 --backbone resnet18 --out models/classifier.pth
```

Train pairwise:
```bash
python train_pairwise.py --data /path/to/dataset --epochs 30 --batch-size 64 --out models/pairwise.pth
```

Train transformer:
```bash
python train_transformer.py --data /path/to/dataset --epochs 50 --batch-size 32 --out models/transformer.pth
```

Build inference cache (compute logits / pairwise / edge matrices for validation set):
```bash
python compute_caches.py --data /val/set --models models/*.pth --out cache/
```

Solve a puzzle (single image):
```bash
python inference.py --image scrambled.png --models models/*.pth --weights "0.6,0.4" --output solved.png
```

Evaluate on validation set (compute PRA and per‑tile acc):
```bash
python evaluate.py --cache cache/ --weights "0.6,0.4" --refine --alpha 1.0 --beta 0.8 --gamma 1.2
```

---

10) **Hyperparameters, evaluation & diagnostics**

Important hyperparameters
- Model training: learning rate, weight decay, batch size, augmentations, backbone.
- Calibration: T_class, T_trans (temperatures).
- Fusion: w_c, w_t (weights).
- Refinement: alpha, beta, gamma; swap strategy and max_iters.
- Edge/pairwise tie parameters: lambda_pair, lambda_edge.

**Evaluation metrics**
- Tile accuracy: fraction of tiles placed in correct positions across puzzles.
- Puzzle Reconstruction Accuracy (PRA): fraction of puzzles completely solved.
- Pairwise accuracy / adjacency AUC for pairwise model.
- Edge matching ROC if trained as classifier.

**Diagnostics**
- Visualize per‑tile marginal distributions (softmax across positions).
- Confusion matrix of classifier across positions.
- Example failure cases: symmetric textures — inspect edge scores and pairwise probabilities.
- Track global objective during refinement — plot GlobalScore per swap iteration.

------------------

11) **Implementation tips, speed & memory considerations**

- Precompute and cache model outputs (per‑tile logits, pairwise probs, edge scores) for validation/inference to avoid recomputation.
- Hungarian is cheap: O(n^3) but for n=9 it's negligible.
- Local search: examine only swaps (or include 2×2 block rotations) — keep neighborhood small for speed.
- Vectorize edge / pairwise computations to use batched numpy/torch ops.
- Use half precision (float16) for inference if memory is constrained.
---

12) **Reproducibility & checkpoints**

- Save random seeds for torch/numpy/python random.
- Save model checkpoints, optimizer states, and validation caches (logits/pairwise/edges).
- Record the exact hyperparameters & git commit in experiment metadata.

---

**References & further reading**
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem.
- Sinkhorn / Differentiable permutation relaxations for permutation supervision.
- Papers on jigsaw puzzle solving for images using deep learning (pairwise and global strategies).
