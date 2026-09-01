# core — base JEPA: representation, planning, uncertainty

Two-room toy environment (64×64 grayscale, point agent, a doorway between
rooms). Small enough to get exact ground truth for every diagnostic.

## Files

| File | Purpose |
|---|---|
| `env.py` | Two-room environment |
| `models.py` | `Encoder` (CNN, frame→128-d latent), `Predictor` (residual MLP dynamics) |
| `losses.py` | VICReg (invariance + variance + covariance) |
| `collect_data.py` | Random-policy transition collection |
| `train.py` | Trains encoder + predictor |
| `eval_probe.py` | Linear probe (position) + multi-step rollout error |
| `train_distance.py` | Learned temporal-distance metric (replaces naive L2 for planning) |
| `plan_greedy.py`, `plan_cem.py`, `plan_distance.py` | Three planner variants, in order of what actually worked |
| `train_ensemble.py`, `eval_uncertainty.py` | Predictor ensemble + calibration diagnostics |
| `train_decoder.py`, `visualize_dreams.py` | Diagnostic-only decoder (frozen encoder) for visualizing imagined rollouts |

## Run order

```bash
python collect_data.py --transitions 50000
python train.py --epochs 40
python eval_probe.py                         # R² = 0.999, smooth rollout curve

python plan_greedy.py                        # works in-room, fails through doorway
python train_distance.py                     # fixes the L2 trap (corr 0.67 → 0.86)
python plan_distance.py                      # now walks through the doorway

python train_ensemble.py --k 5
python eval_uncertainty.py                   # disagreement/error corr ≈0.98

python train_decoder.py
python visualize_dreams.py                   # imagined vs. real rollout, side by side
```

## Key finding: L2 in latent space is not a valid planning metric

Correlation between raw Euclidean latent distance and true reachability:
**0.67**. The naive planner found "cheap" local minima where standing still
scored better than moving toward the goal — because two states on opposite
sides of a wall can be latently close despite being many steps apart. A
separately trained temporal-distance network (self-supervised: distance =
number of steps between states in the same trajectory) raises this to
**0.86** and lets the planner route through the doorway.

This same bug reappeared, independently, in `drone_quantum/` navigation —
see that folder's README for the honest account of repeating the mistake in
a new codebase.
