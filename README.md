# Wordle as Geometry

**Can Wordle competence be stored as fixed geometry instead of per-move search?**

3b1b's classic solver does expensive combinatorial search (entropy maximization)
on every move. This project asks whether that competence can instead be
**amortized once into the geometry of an embedding space**, so that at play-time
the agent only ever:

1. reads the **nearest answer-word** to its current position vector, and
2. **moves** that vector by a **frozen, build-time lookup** keyed by the feedback
   it just saw.

No per-move search. No consistency filtering. No learned play-time parameters.
If the geometry carries the competence, this near-trivial controller should win.

See [`docs/superpowers/specs/2026-05-30-wordle-geometry-design.md`](docs/superpowers/specs/2026-05-30-wordle-geometry-design.md)
for the full design and the anti-circularity "honesty contract".

## The pipeline

```
P[i,j]   pattern matrix (2315 x 2315), 3b1b-style ternary feedback   src/patterns.py
   |
S[a,b]   Relation-B similarity = mean_g 1[P[g,a]==P[g,b]]            src/similarity.py
   |
E        geometry: word -> R^d   (random | semantic-GloVe | MDS | contrastive)  src/geometries.py
   |
T[g,f]   FROZEN (guess,pattern) -> consistency-cell centroid        src/controller.py
   |
play     p<-centroid; repeat: guess=nearest(p); p<-(1-eta)p+eta*T[g,f]; stop on green
```

The **honesty contract**: the play-time navigator (`WordleNavigator`) sees only
`(position, guess_index, pattern)`. It never reads `P` or the candidate set at
play-time. Enforced by a test that the controller module imports neither
`src.patterns` nor `src.similarity`.

## Geometry sources (the independent variable)

| source | what it encodes | role |
|---|---|---|
| `random-geom` | nothing (random vectors) | **load-bearing control** — floor for "any geometry + this rule" |
| `semantic-glove` | linguistic meaning (GloVe 6B) | control — tests the "meaningful embedding" intuition |
| `mds` | classical MDS of `1 - S` | baseline — does the feedback structure embed? |
| `contrastive` | metric-learned `S` (SupCon/triplet) | main method (M2) |

Same frozen controller throughout, so **competence vs. geometry-source** is a clean
comparison. The thesis predicts win-rate rises random → semantic → MDS → contrastive.

## Reproduce

```bash
micromamba run -p ./.venv python -m scripts.build_substrate   # P and S
micromamba run -p ./.venv python -m scripts.run_eval          # decisive figure + table
micromamba run -p ./.venv python -m pytest tests/ -q          # all tests
```

## Result (held-out eval, same frozen controller)

| geometry | win@6 | mean guesses |
|---|---|---|
| random-guess / freq-greedy | 0.002 | ~7.0 |
| random-geometry | 0.31 | 6.16 |
| semantic-GloVe | 0.17 | 6.46 |
| MDS (feedback) | 0.66 | 5.34 |
| contrastive-SupCon | **0.75** | **4.90** |
| entropy-solver (ceiling) | 0.997 | 3.58 |

**Competence scales with geometry quality.** Feedback-learned geometry beats
random by 2.4x under an identical controller; semantic meaning is *worse* than
random (it's anti-correlated with letter structure); SupCon (local
neighborhood shaping) beats closed-form MDS. The per-turn distance to the secret
converges only for the feedback geometries — direct evidence that "moving
downhill" is real. See [`FINDINGS.md`](FINDINGS.md).

## Status

- M0 substrate (Wordle rule, pattern matrix, similarity): **done**.
- M1 MDS + random/semantic controls + frozen controller + eval: **done**.
- M2 contrastive geometry (reconstruction + SupCon, CLXAI losses): **done**.
- M3 analysis (convergence trace, cell convexity, entropy-solver ceiling): **done**.
- M4 (future): differentiable play-and-move training; 13k-guess list;
  information-gain-aware readout.

```bash
micromamba run -p ./.venv python -m scripts.run_analysis   # M3 figures + entropy ceiling
```
