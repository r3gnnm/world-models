# drone_quantum — frontier selection and closed-loop exploration

Applied track: can a world model's uncertainty estimate drive autonomous
exploration decisions, and does formulating waypoint selection as a QUBO
(quantum-annealing-ready) do anything a greedy selector can't?

**Honest framing up front:** this is engineering integration of already-
validated pieces (`core/`'s encoder+predictor+ensemble, this folder's QUBO
solver), not a controlled scientific experiment. The QUBO-vs-greedy result
below *is* rigorously validated (10 seeds); the closed-loop demo is a
qualitative integration test, not a benchmark.

## Files

| File | Purpose |
|---|---|
| `collect_openworld.py` | Transition collection on the open-world env (same `.npz` format as `core/train.py`, fully compatible) |
| `frontier_qubo.py` | QUBO formulation: pick M of K candidate points, maximizing ensemble-predicted uncertainty while penalizing redundant nearby pairs. Solved via classical simulated annealing (`dwave-neal`/`dimod`) — architecturally a one-line swap to real quantum annealing hardware |
| `frontier_multiseed.py` | Multi-seed validation of the QUBO result (learned the hard way in `hierarchy/` that single-run comparisons mislead) |
| `closed_loop_exploration.py` | Full loop: sample candidates → ensemble uncertainty → QUBO select → chained latent-space navigation → repeat |

## Setup

```bash
pip install dwave-neal dimod
python collect_openworld.py --transitions 30000
python train.py --data data/openworld.npz --epochs 30 --out checkpoints/openworld_jepa.pt
python train_ensemble.py --data data/openworld.npz --ckpt checkpoints/openworld_jepa.pt --k 5 --epochs 20 --out checkpoints/openworld_ensemble.pt
python train_distance.py --data data/openworld.npz --ckpt checkpoints/openworld_jepa.pt --steps 5000 --out checkpoints/openworld_distance.pt
```

*(`train.py`, `train_ensemble.py`, `train_distance.py`, `models.py`, `losses.py`
are duplicated here from `core/` so this folder runs standalone — see the
top-level README for why.)*

## Result: QUBO avoids redundant candidates at negligible cost

```bash
python frontier_multiseed.py --seeds 0 1 2 3 4 5 6 7 8 9
```

Over 10 independent seeds: QUBO selects **0 redundant pairs** (candidates
within the redundancy radius of each other) in every single run, vs. a mean
of **3.1** for greedy top-M selection — while sacrificing only **2.6%** of
total predicted information value (u_qubo/u_greedy = 0.974 ± 0.012). Unlike
the hierarchy landmark sweep, this effect is fully robust across seeds — it
comes from solving a fixed optimization problem, not from training a neural
network, so there's far less run-to-run noise to control for.

## Closed-loop exploration demo

```bash
python closed_loop_exploration.py --rounds 5 --n-candidates 20 --waypoints-per-round 3
```

Repeats: sample candidates within realistic travel range of the drone's
current position → score uncertainty via the ensemble → QUBO-select a
diverse subset → navigate to each via chained short-horizon MPC (long hops
broken into ≤25-unit legs — necessary because a single short-horizon CEM
step can't route around dense obstacle clusters from far away) → repeat with
the newly visited area excluded from future candidates.

Current success rate: **67% of waypoints reached** (10/15 in the reference
run), with successful legs completing in 6–14 steps. Remaining failures have
a known, unfixed cause: within-round nearest-neighbor ordering can leave the
last waypoint of a batch farther from the *current* position than the
`max_travel_dist` cap intended (the cap is enforced from the position at
round start, not between waypoints within the round).

### A bug worth documenting: the same L2-latent trap, twice

The first version of `closed_loop_exploration.py`'s navigator used raw
Euclidean distance in latent space as the CEM cost — exactly the mistake
identified and fixed in `core/` (see that README). It was rewritten from
scratch here and reintroduced the same flaw, which produced the same
symptom (the agent getting stuck near the start, unable to route through
obstacle clusters). Fixed by reusing `core/train_distance.py`'s trained
metric instead of raw MSE. Kept in this README as a reminder: a lesson
learned in one module doesn't automatically transfer to a new one — reuse
the component, don't just remember the principle.

## Quantum computing: what's real here and what isn't

- The QUBO formulation is genuine and the classical solver (`dwave-neal`)
  uses the same `dimod.BinaryQuadraticModel` interface as real D-Wave
  hardware — swapping `SimulatedAnnealingSampler()` for `DWaveSampler()` is
  the only change needed to run on real annealing hardware.
- No experiments in this repository have run on real quantum hardware.
- See `../docs/quantum_explainer.md` for the broader, non-technical case for
  (and against) quantum computing in this project — written for a
  non-specialist stakeholder audience.

## Update: energy/risk-aware QUBO + route compactness fix

Extended `build_qubo` with three additional terms beyond the original
information-value objective: **energy** (distance from drone to candidate),
**risk** (mean ensemble disagreement *along the path*, not just at the
destination), and a **route-compactness penalty** (total pairwise distance
among selected candidates) — the last one doubles as a fix for a routing bug
where a batch of waypoints could each be individually reachable from the
round's start position, yet the last one in the visiting order ends up
unreachable from wherever the drone is by the time its turn comes.

Validated over 10 seeds: routes exceeding a realistic travel budget dropped
from **9/10 (greedy) to 1/10 (QUBO)**, at the cost of average selected value
dropping to 0.730±0.038 of greedy's (up from the simpler redundancy-only
version's 0.974 — a more complete, more expensive, more realistic
trade-off).

Closed-loop exploration success rate improved correspondingly as the
structural failure mode was removed: **67% → 73% → 80%** waypoint success
across three iterations (route-compactness fix, then a modest per-leg step
budget increase from 25 to 40 — remaining failures were mild
under-budget cases, not structural ones). Diminishing returns from here;
this iteration of the 2D testbed is considered complete.

## Update: recurrent encoder for velocity recovery (three-way control)

The 2.5D quadrotor dynamics predictor exhibits systematic amplitude damping,
traced to two candidate causes and resolved: **more training epochs** (not
loss-weight rebalancing, which made rollouts *less* stable at small scale)
closed most of the gap for directly-observable state variables. Same-
trajectory action ablation gap grew monotonically with training length
(40→80→150 epochs: 0.007→0.067→0.18 gap), confirming the fix.

Velocity variables (vx, vz, omega) plateaued regardless of training length —
consistent with the hierarchy/ finding that instantaneous frames cannot
encode information requiring temporal integration. A recurrent GRU layer
was added on top of the frozen encoder, validated with a **three-way
control** distinguishing recurrence from mere dimensionality:

| Variable | z1 only | +trained GRU | +random GRU | +random linear projection |
|---|---|---|---|---|
| x, z, theta | high | unchanged | unchanged | unchanged (exactly 0.000 gain) |
| vx | 0.164 | **0.680** | 0.378 | 0.164 (exactly 0.000 gain) |
| vz | 0.034 | **0.270** | 0.133 | 0.034 (exactly 0.000 gain) |
| omega | 0.157 | **0.564** | 0.461 | 0.157 (exactly 0.000 gain) |

The random linear projection control gives *exactly zero* gain everywhere —
expected, since a linear map of z1 adds no information a linear probe on z1
couldn't already extract. The random (untrained) GRU gives a substantial,
non-trivial gain on velocity variables specifically — a genuine reservoir-
computing effect from recurrence alone, not a dimensionality artifact.
Training the GRU roughly doubles this further. Position/angle variables are
untouched by any of the three interventions, exactly as predicted.

This is an independent replication, on a physically distinct underactuated
dynamical system, of the central finding from `hierarchy/`: memory helps
precisely where information is structurally absent from a single
observation, and is inert everywhere else.

## Final result: observation design determines whether latent planning works

Goal-image planning for an underactuated quadrotor requires the latent to
carry three things simultaneously: **position** (where to go — enters the
planner's cost), **attitude** (how to tilt — horizontal motion exists only
via tilt), and **velocity** (when to brake — the system has inertia). We
measured which observation designs supply which, and how each deficit
propagates to control. Baseline: CEM on *true physics* reaches the goal
10/10, isolating planner correctness from model quality.

| Configuration | probe x | probe z | probe θ | probe vx | planning | mean min dist |
|---|---|---|---|---|---|---|
| Egocentric crop | 0.53 | 0.69 | 0.92 | 0.16 | 0/10 | 22.4 |
| Global view, per-episode maps | −0.94 | −0.56 | −0.34 | −0.06 | 0/10 | 31.8 |
| Global view, fixed map | 0.99 | 0.99 | 0.34 | 0.06 | 1/10 | 32.9 |
| Dual-channel (global + ego) | 0.98 | 0.99 | 0.94 | 0.28 | 2/10 | 20.6 |
| Dual-channel + recurrent context | 0.99 | 0.99 | 0.95 | **0.77** | **5/10** | **15.0** |
| *CEM on true physics (baseline)* | — | — | — | — | *10/10* | — |

Each row closes one measured information deficit, and planning improves
monotonically. Three findings are worth extracting:

**Shortcut learning under global view (row 2).** With per-episode obstacle
maps, prediction loss *decreased* (17.7 vs 25.8) while probe R² went
negative and action-gap sat at zero. Measured cause: between-episode
variance (map identity) is 5.1× the within-episode variance (drone motion),
and the map is static within an episode — so the cheapest way to minimise
prediction error is to encode the background and ignore the agent. Fixing
the map across episodes drops that ratio to 1.2× and the failure disappears.

**Observation modes are complementary (rows 1, 3, 4).** Egocentric crops
give attitude (0.92) but lose position; global view gives position (0.99)
but loses attitude (0.34) since the drone is a small mark in a downsampled
frame. Neither alone suffices; supplying both as two channels recovers both.

**Velocity requires recurrence, not capacity (row 5).** With a three-way
control — trained GRU vs untrained GRU vs a random *instantaneous* linear
projection of matched dimensionality — the linear projection yields
*exactly zero* gain on every variable, while recurrence lifts vx 0.275 →
0.770, vz 0.082 → 0.553, omega 0.264 → 0.652, leaving position and attitude
untouched. This is the same principle established in `hierarchy/`,
replicated on a physically distinct underactuated system: memory helps
precisely where information is structurally absent from a single frame.

**Remaining gap (5/10 vs 10/10).** All three information deficits are now
closed at the representation level, yet a gap persists — attributable to
15-step rollout accuracy of the learned dynamics rather than to missing
information. Not pursued further.

**Engineering implication.** A purely visual world model must recover
position, attitude *and* velocity from pixels before latent goal-image
planning becomes viable; observation design, not model capacity, was the
binding constraint at every step here. In a real system these quantities
come from VIO/SLAM + IMU, which is why a hybrid architecture — classical
state estimation underneath, world model on top for prediction, uncertainty
and exploration — is the practical choice. This project's uncertainty
ensemble and QUBO frontier selection carry over to that architecture
unchanged, since neither depends on how position is obtained.
