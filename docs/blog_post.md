# Building a JEPA World Model From Scratch: What Broke, and Why That Was the Point

*A solo project on a single GPU, following the JEPA line of work associated with Yann LeCun — from a toy two-room environment to a controlled experiment on when hierarchy actually helps.*

---

## Why I started this

I wanted to understand World Models properly — not by reading about them, but by building one and breaking it in every way I could think of. I picked the JEPA (Joint-Embedding Predictive Architecture) approach: instead of predicting future *pixels*, you predict future *representations*. No decoder in the training loop, no wasted capacity on texture and noise.

The environment is deliberately tiny: a point agent in a 64×64 grayscale world, walls with a doorway between two rooms. Small enough to get exact ground truth for every diagnostic. That choice turned out to matter more than I expected.

## The core model

- **Encoder**: a small CNN, frame → 128-d latent.
- **Predictor**: `z_{t+1} = z_t + f(z_t, a_t)` — residual, conditioned on a 2D action.
- **Loss**: VICReg in latent space — invariance (prediction accuracy) + variance (anti-collapse, forces per-dimension std ≥ 1) + covariance (decorrelation).

First validation looked great:

- **Linear probe R² = 0.999** on agent position — the latent linearly encodes exactly where the agent is.
- **Smooth rollout error growth** over 30 imagined steps, no blow-up.

Textbook JEPA behavior. Then I tried to actually *use* the model — and that's where it got interesting.

## Failure #1: the planner wouldn't move

I built a CEM planner: encode a goal image, roll candidate action sequences through the predictor, pick the ones that minimize latent distance to the goal, execute the first action, replan (MPC).

The agent just sat there.

![The naive L2 planner gets stuck near the start](greedy_trajectory.png)

Diagnosis, not guessing: I directly compared the cost of "drive toward the goal" against "stand still."

```
всё вправо   -> итоговый cost 2.2145
стоять       -> итоговый cost 1.2336
```

Standing still was *cheaper* than moving toward the goal. Not a bug — a property of the latent geometry. I measured correlation between L2 latent distance and true pixel-space distance across a grid of positions: **0.67**. States on the other side of a wall can be latently "close" even though they're many steps away — L2 distance doesn't respect the manifold's curvature, and the planner exploits that.

## Fix: learn the distance instead of assuming it

I trained a separate `d(z, z')` network with a self-supervised signal that costs nothing extra: pairs of states *k* steps apart in the same episode get target distance *k*; pairs from different episodes get a hinge loss (not a hard target — many "negative" pairs are spatially close purely by chance, and forcing them to a fixed large distance actively destabilized training the first time I tried it).

Correlation with true distance jumped to **0.86**. And the planner, using this metric as cost, did what L2 never could:

![Agent successfully navigating through the doorway using the learned distance metric](distance_trajectory.png)

The curved path *away* from the straight line to the goal, through the gap, is the whole point — it shows the model learned topology, not just "minimize pixel distance."

## Stress-testing what's actually learned

Once planning worked, I tried to break the representation on purpose, with a control for each claim:

**More complex topology** (three rooms instead of two): no degradation. Probe R² unchanged, rollout only slightly worse. The model generalized the *principle*, not a memorized map.

**An independent distractor** moving with no relation to the agent's actions: JEPA is supposed to learn to ignore unpredictable elements rather than waste capacity modeling them. Partially true — position probe stayed at R²=0.99, but the normalized action-gap (a measure of how much the predictor relies on the action) dropped threefold. The model didn't fully ignore the noise; it partially absorbed it.

**Partial observability** (egocentric crop instead of top-down view) broke the memoryless predictor outright — global position couldn't be recovered from a single frame. I ran the direct control: same code, same data, only difference is a GRU hidden state carried across the episode.

| | probe R² from single frame | probe R² with memory |
|---|---|---|
| Egocentric env | 0.821 | **0.898** |
| Full-observability env (control) | 0.997 | 0.997 |

Memory only helps exactly where it should. On the fully-observable control, the same GRU adds nothing — ruling out "it's just a bigger network" as the explanation.

## Hierarchy: where it got genuinely interesting

LeCun's 2022 autonomous-intelligence manifesto sketches H-JEPA — prediction at multiple temporal/spatial scales — as an architectural *aspiration*, without a concrete training recipe. I wanted to actually test it, with a control that most descriptions of hierarchical models skip: **compare a trained top-level abstraction against a random, untrained one with identical architecture.**

Built a 3×3 grid-of-rooms environment — an explicit two-scale structure (fast: position inside a room, slow: which room) with exact ground truth for both.

![The building environment: 3x3 rooms with doors](building_preview.png)

First version of the abstractor was instantaneous — `z2 = MLP(z1)`, no memory. On the fully-observable environment:

```
z2 (trained) room accuracy:  0.997
z2 (random)  room accuracy:  0.995
```

The trained abstraction added essentially nothing. Everything it "learned" was already linearly available in the frame — the collapse wasn't a training bug, it was structurally inevitable: there was nothing left for the top level to add.

So I switched to an egocentric view, where room identity genuinely isn't recoverable from a single frame — and initially still saw nothing (0.048 vs 0.060 trained vs random), because the abstractor was still instantaneous and had no mechanism to accumulate the history needed to infer which room you're in.

![The egocentric building view — rooms look identical from a single frame](ego_building_preview.png)

Made the abstractor recurrent (a GRU accumulating over time, not a per-frame projection) — and got the full 2×2 result:

| | Full observability | Egocentric (partial) |
|---|---|---|
| Instantaneous abstractor | trained ≈ random (0.997 vs 0.995) | trained ≈ random (0.048 vs 0.060) |
| Recurrent abstractor | trained ≈ random (0.989 vs 0.978) | **trained ≫ random (0.638 vs 0.400)** |

Only one cell shows a real, learned abstraction. The other three are decorative for two *different* reasons: full observability means there's nothing to add; instantaneous computation means there's no mechanism to add it even when something's missing.

**The takeaway**: hierarchy helps if and only if two conditions hold simultaneously — a mechanism to accumulate history, and information that's genuinely unavailable below without it. Neither condition alone is sufficient. This is essentially *why* DreamerV3's RSSM is built the way it is (deterministic recurrent state + instantaneous latent) — except I arrived at it experimentally rather than by reading the architecture off a diagram.

## What I'd tell someone starting this

Run the control, not just the metric. Every "surprising" result in this project turned out to have a boring explanation once I ran a control experiment — a random baseline of the same architecture, a frame-shuffled ablation, a fully-observable version of the same test. Without those, I would have reported at least two false positives (the L2 planner "sort of working," the instantaneous abstractor "learning" a room representation).

A negative result with a diagnosis is worth more than a positive result you can't explain. The most useful finding in this whole project — the 2×2 hierarchy matrix — is two negative results and one positive one, and the negatives are what make the positive one credible.

## What's next

Normalizing the action-gap metric properly for recurrent models (the naive ablation currently underestimates it, since the hidden state already "knows" the true action history). A flat baseline with matched parameter count for the hierarchy experiment, to isolate the effect of hierarchy from raw capacity. And eventually, moving the same methodology onto real spatiotemporal data — satellite time series are a natural next testbed, since they come with genuine multi-scale structure (parcel → region → season → year) and ground truth at every level.

Code for the full project — environments, training, planners, all the diagnostics above — is available on request.

---

*If you're working in this space or thinking about it, I'd be glad to compare notes.*
