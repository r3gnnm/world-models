# When Does Hierarchical Abstraction in World Models Actually Learn Anything? A Controlled Diagnostic Study

**Draft — workshop paper (target: 4–8 pages)**

---

## Abstract

Hierarchical predictive architectures such as H-JEPA propose learning
abstractions at multiple temporal and spatial scales, but the proposal is
architectural rather than algorithmic: it specifies neither a training
objective for the abstract level nor a way to verify that a trained
abstraction encodes anything a random projection of the same shape would not.
We introduce a diagnostic protocol built around a *random-abstractor
control* — comparing a trained top level against an identically-parameterized
but untrained one — and apply it across a 2×2 design crossing observability
(full vs. egocentric) with abstractor type (instantaneous vs. recurrent) in
controlled gridworld environments with exact ground truth at both scales.
Three of four conditions yield abstractions that are indistinguishable from
random projections, for two distinct reasons: under full observability the
abstraction has nothing to add, since the slow variable is already linearly
decodable from the fast level; under partial observability an instantaneous
abstractor lacks any mechanism to integrate the history required to recover
it. Only the combination of partial observability and a recurrent abstractor
produces a measurably learned abstraction (0.638 vs. 0.400 top-level probe
accuracy, a 24-point gap). We further report a negative result: attempts to
establish a quantitative relationship between the density of persistent
visual landmarks and abstraction quality in an open-world setting did not
survive multi-seed replication, despite a promising single-run trend. Our
results give an empirical account of *why* architectures such as DreamerV3's
RSSM combine a recurrent deterministic state with an instantaneous latent,
and provide a reusable diagnostic that we argue should accompany any claim
that a hierarchical model has learned a meaningful abstraction.

---

## 1. Introduction

Predicting in representation space rather than pixel space is the core
commitment of the JEPA family of world models: rather than reconstructing
future observations, the model learns to predict future *embeddings*,
avoiding spending capacity on unpredictable detail. LeCun's 2022 position
paper extends this to H-JEPA — a hierarchical variant predicting at multiple
temporal and spatial scales, with a slow abstract level providing context to
a fast local one.

H-JEPA is presented as a normative blueprint. It does not specify a concrete
training loss for the abstract level, a probabilistic semantics, or a
training algorithm — and, critically for empirical work, it does not specify
how one would *check* that a trained hierarchical model has learned a
meaningful abstraction rather than a decorative one. This gap matters
because hierarchical models are known to be prone to a specific failure:
the lower level solves the task alone, gradients through the upper level
become uninformative, and aggregate metrics still look healthy because the
lower level carries them.

**Contributions.**

1. A diagnostic protocol for hierarchical world models centered on a
   *random-abstractor control*, plus two supporting diagnostics
   (level-use ablation and equal-capacity differential probing).
2. A 2×2 empirical study crossing observability with abstractor type,
   showing that three of four conditions produce abstractions statistically
   indistinguishable from random projections — for two structurally
   different reasons.
3. An honest negative result on landmark density in open-world settings,
   including the methodological failure mode (a convincing trend on single
   runs that did not survive seed replication) that produced it.

---

## 2. Method

### 2.1 Environments

All environments render 64×64 grayscale observations and expose ground truth
at two scales, enabling exact probing of both levels.

**Building (3×3 rooms).** A point agent moves in a grid of nine rooms
connected by doorways. The *fast* variable is position within the current
room; the *slow* variable is room identity, which changes rarely (empirically
~2 transitions per 75 steps under a random inertial policy). Two observation
modes: `full` (top-down view of the whole map) and `ego` (a window cropped
around the agent, so rooms are mutually indistinguishable from a single
frame and room identity is only recoverable by integrating history).

**Open world.** A larger (128×128) continuous space scattered with circular
obstacles and no room structure; the "slow variable" is an artificial 4×4
zone grid imposed only for diagnostic probing, not present in the
environment's dynamics. Three variants: obstacles fixed across the dataset
(landmark shortcuts possible), reshuffled every episode (pure dead
reckoning), and reshuffled with `n` persistent landmarks retained.

### 2.2 Architecture

The fast level is a convolutional encoder `z1 = Enc(o)` (128-d) with a
residual dynamics predictor conditioned on both action and abstract context:
`ẑ1_{t+1} = z1_t + f(z1_t, a_t, z2_t)`. The abstract level is either

- **instantaneous**: `z2_t = MLP(z1_t)` (32-d), or
- **recurrent**: `z2_t = GRU(z1_t, z2_{t-1})` (32-d),

with a jumpy level-2 predictor over k-step intervals taking an aggregated
action summary as input. Both levels are trained with VICReg
(invariance + variance + covariance) in latent space; no pixel
reconstruction is used anywhere in training.

### 2.3 Diagnostics

**Random-abstractor control (primary).** We instantiate a second abstractor
with identical architecture and dimensionality but untrained (random
initialization), encode the same data through it, and fit the same probe.
If trained ≈ random, training moved the abstraction nowhere a random
projection of the fast level could not already reach. This control is the
core of our protocol: without it, a high top-level probe score establishes
only that the slow variable is *recoverable*, not that it was *learned*.

**Level-use ablation.** During training we measure the increase in level-1
prediction error when `z2` is replaced by another batch element's — a
hierarchical analogue of an action-conditioning ablation. Near-zero
indicates the lower level ignores the abstract context.

**Equal-capacity differential probing.** Because `z2 = g(z1)` for a
nonlinear `g`, a *linear* probe on `z2` is effectively a *nonlinear* probe on
`z1`, and comparing the two directly is confounded. We therefore additionally
probe both levels with matched-capacity nonlinear probes.

---

## 3. Experiments

### 3.1 The 2×2 design

Top-level probe accuracy on the slow variable (room identity, 9 classes),
trained abstractor vs. random control:

| Observability | Abstractor | Trained | Random | Gap |
|---|---|---|---|---|
| Full | Instantaneous | 0.997 | 0.995 | +0.2 pp |
| Full | Recurrent | 0.989 | 0.978 | +1.1 pp |
| Egocentric | Instantaneous | 0.340 | 0.358 | −1.8 pp |
| Egocentric | Recurrent | **0.638** | **0.400** | **+23.8 pp** |

Only one cell shows a learned abstraction. The three failures have two
distinct causes:

**Nothing to add (full observability).** Room identity is already almost
perfectly linearly decodable from `z1` (probe accuracy up to 1.000 in the
recurrent-abstractor run), so any projection — trained or random — preserves
it. The near-perfect scores in the top two rows measure *recoverability*,
not learning. This is the failure mode the random control was designed to
catch, and without it these rows would read as strong successes.

**No mechanism (egocentric + instantaneous).** With an egocentric crop,
room identity is not recoverable from any single frame; it requires
integrating a history of door transitions. An instantaneous
`z2 = MLP(z1)` is a deterministic function of the current frame alone and
therefore cannot encode more than `z1` does. The trained abstractor scores
*marginally below* random here, consistent with there being no learnable
signal to acquire.

**Both conditions met (egocentric + recurrent).** Making the abstractor
recurrent supplies the missing mechanism, and the 23.8-point gap over random
indicates a genuinely learned abstraction. The level-use ablation moves in
the same direction (0.0088 → 0.0706, a ~5× increase over the instantaneous
case), indicating the lower level begins to rely on the abstract context
once that context carries information the frame does not.

### 3.2 Memory ablation with an observability control

A separate experiment isolates memory in the *dynamics* predictor (rather
than the abstractor), training an identical pipeline with a GRU-carrying
versus memoryless predictor:

| Environment | Probe R² (no memory) | Probe R² (with memory) |
|---|---|---|
| Egocentric | 0.821 | **0.898** |
| Full observability (control) | 0.997 | 0.997 |

Memory helps precisely where information is missing and nowhere else,
ruling out increased capacity as the explanation.

### 3.3 Negative result: landmark density in open worlds

We tested whether the *quantity* of persistent visual anchors quantitatively
predicts abstraction quality, using the open-world environment with `n`
persistent landmarks among per-episode-reshuffled obstacles. Single runs at
n = 0, 3, 6 produced an apparently clean monotonic trend (0.6 → 5.5 → 8.5 pp
gap). Replicating with three seeds per point falsified it:

| n landmarks | Gap, mean (pp) | Std (pp) |
|---|---|---|
| 0 | 5.77 | 2.45 |
| 3 | 11.97 | 4.49 |
| 6 | 4.03 | 2.10 |
| 10 | 4.68 | 2.13 |
| 15 | 5.41 | 5.35 |

No monotonic relationship survives; between-point differences are comparable
to within-point standard deviation. We report this because the single-run
version was convincing enough that we had begun writing it up as a positive
finding.

What *does* hold qualitatively across every variant tested: pure dead
reckoning (zero landmarks, obstacles reshuffled every episode) never
produces a large gap, while the building environment's doorways — frequent,
reliable, visually distinct anchors encountered on every room transition —
produce the largest gap observed (23.8 pp). We therefore conjecture, without
claiming to have established, that anchor *reliability and frequency of
encounter* matter more than raw count.

---

## 4. Discussion

Our results suggest hierarchical abstraction becomes measurably real only
when two conditions hold simultaneously: **a mechanism capable of
integrating history**, and **information genuinely unavailable to the lower
level without it**. Neither alone suffices, and each failure mode is
invisible to standard metrics — the full-observability cells post the
highest absolute probe scores in the entire study while learning nothing.

This offers an empirical account of an architectural choice usually stated
without justification: RSSM-style world models (e.g. DreamerV3) pair a
recurrent deterministic state with an instantaneous stochastic latent. Our
2×2 indicates that on fully-observable benchmarks the recurrent component is
approximately inert, and its contribution appears only under partial
observability — which is consistent with, and may partly explain, why
hierarchical or recurrent components sometimes appear to contribute little
on toy fully-observable tasks.

**Limitations.** All environments are small synthetic gridworlds with
hand-specified slow variables; nothing here has touched real sensor data or
learned its own abstraction hierarchy rather than being probed against a
predefined one. The open-world "slow variable" is an artificial zone grid
imposed for probing, not an emergent structure. The landmark-density
question remains open rather than answered negatively — our null result
bounds the effect size detectable at our sample size, it does not
demonstrate absence.

---

## 5. Conclusion

We provide a diagnostic protocol for verifying that hierarchical world-model
abstractions are learned rather than decorative, and apply it to show that
three of four natural design conditions produce abstractions
indistinguishable from random projections. We argue the random-abstractor
control is cheap enough — one extra untrained module and one extra probe —
that it should be standard in any work claiming a hierarchical model has
learned a meaningful abstraction.

Code, environments, and all diagnostics: [repository link]

---

## Appendix A — reproduction

```bash
# 2x2 design
python train_hier.py --env full --episodes 200 --ep-len 48 --epochs 40 --k 8
python train_hier.py --env ego  --episodes 200 --ep-len 48 --epochs 40 --k 8

# memory ablation with observability control
python compare_memory.py --env egocentric --episodes 150 --ep-len 32 --epochs 20
python compare_memory.py --env base       --episodes 150 --ep-len 32 --epochs 20

# landmark density sweep (multi-seed)
python run_landmark_sweep.py --landmarks 0 3 6 10 15 --seeds 0 1 2 \
    --episodes 200 --ep-len 48 --epochs 40 --k 8
```
