# hierarchy — when does hierarchical abstraction actually help?

The main result of this project. LeCun's 2022 manifesto proposes H-JEPA —
multi-scale predictive abstraction — as an architectural blueprint, with no
concrete training recipe and, critically, no way to verify that a trained
"slow" abstraction encodes anything real rather than being decorative. This
folder builds that verification.

## Environments

| File | What it tests |
|---|---|
| `env_building.py` | 3×3 grid of rooms; `BuildingEnv` (top-down, full obs) and `EgocentricBuildingEnv` (agent always centered, partial obs) |
| `env_openworld.py` | Open 128×128 world, no rooms. Three variants: `OpenWorldEnv` (static obstacles — landmark shortcuts possible), `OpenWorldDynamicEnv` (obstacles reshuffled every episode — pure dead reckoning), `OpenWorldLandmarkEnv` (dynamic obstacles + `n_landmarks` persistent anchors, parametrized) |
| `env_variants.py` | Stress-test variants for the core JEPA (three rooms, moving distractor, egocentric fog — see `stress_test.py`) |

## Models

| File | Purpose |
|---|---|
| `models_hier.py` | `Abstractor` (recurrent GRU, z1→z2), `Level1Predictor`/`Level2Predictor`, `FlatPredictor` (equal-parameter control) |
| `models_recurrent.py` | `RecurrentPredictor` (GRU) vs `MLPPredictorSeq` — same interface, isolates the effect of memory |

## The critical diagnostic: random-abstractor control

`train_hier.py` doesn't just report the trained top level's probe accuracy —
it compares against an **identically-shaped but untrained** abstractor. If
trained ≈ random, the abstraction learned nothing; the training loss went
somewhere, but not into anything a linear probe can find that a random
projection couldn't already find. This single control overturns most of the
"it's working" appearances in this codebase's history — see results below.

```bash
python train_hier.py --env full --episodes 200 --epochs 40   # decorative
python train_hier.py --env ego  --episodes 200 --epochs 40   # decorative (instantaneous abstractor)
```

At this point `Abstractor` was still an instantaneous `z2 = MLP(z1)` — no
memory. Switching it to a recurrent GRU (`z2_t = GRU(z1_t, z2_{t-1})`) is
what makes the difference below.

## Results

| Environment | Abstractor | Trained top-level acc | Random-control acc | Gap |
|---|---|---|---|---|
| Full obs | recurrent | 0.989 | 0.978 | ~1pp — decorative, nothing to add |
| Partial obs (ego), instantaneous | — | 0.340 | 0.358 | ~0pp (trained slightly *below* random) — no mechanism to integrate history |
| Partial obs (ego), recurrent | recurrent | **0.638** | **0.400** | **24pp — real** |

Full run:
```bash
python train_hier.py --env full --episodes 200 --epochs 40   # recurrent, full obs
python train_hier.py --env ego  --episodes 200 --epochs 40   # recurrent, partial obs
```

## Open-world extension: does this survive outside a closed room grid?

Landmark density sweep (`run_landmark_sweep.py`), averaged over 3 seeds per
point — **read this section before quoting a single-run number**:

```bash
python run_landmark_sweep.py --landmarks 0 3 6 10 15 --seeds 0 1 2 \
    --episodes 200 --ep-len 48 --epochs 40 --k 8
```

**Methodological note, kept here deliberately:** a first pass with single
runs at n=0, 3, 6 landmarks looked like a clean monotonic trend (0.6pp →
5.5pp → 8.5pp). Averaging 3 seeds per point falsified that: means came back
noisy (5.8, 12.0, 4.0, 4.7, 5.4 pp for n=0,3,6,10,15) with std comparable to
the differences between points. **No significant landmark-density law was
established.** What *did* hold up, consistently, across every variant
tested: pure dead reckoning (`OpenWorldDynamicEnv`, zero landmarks) never
produces a real abstraction gap, while `EgocentricBuildingEnv`'s door — a
frequent, reliable anchor — produces a large one. The takeaway is qualitative
(rare-but-present anchors > none) not quantitative (a landmark-count law).

## Stress tests (core JEPA, not hierarchy)

`stress_test.py` — does the base representation survive harder environments?

```bash
python stress_test.py --transitions 20000 --epochs 25
```

| Variant | Result |
|---|---|
| Three rooms (harder topology) | No degradation — R²=0.998, principle generalizes |
| Moving distractor | Position probe holds (R²=0.99) but action-gap drops 3× — partial, not full, noise rejection |
