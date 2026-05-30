# Findings: Wordle as Geometry

Experiment date: 2026-05-30. Answers-only scope (2,315 official Wordle answers),
tune/eval split (1,157 / 1,158), same frozen `(guess, pattern)` cell-centroid
controller across all geometries. All numbers on the **held-out eval split**.

## Headline result

A **frozen embedding + a frozen `(guess, pattern)` → cell-centroid lookup**, with
no per-move search and no consistency filtering at play-time, plays Wordle far
above chance. Competence scales cleanly with how well the geometry encodes the
game's *feedback* structure.

| Agent | Win@6 | Mean guesses (loss=7) |
|---|---|---|
| random-guess | 0.002 | 6.99 |
| freq-greedy | 0.002 | 7.00 |
| random-geometry | 0.306 | 6.16 |
| semantic-GloVe | 0.171 | 6.46 |
| MDS (closed-form, feedback) | 0.663 | 5.34 |
| contrastive-reconstruction | 0.377 | 6.13 |
| **contrastive-SupCon** | **0.746** | **4.90** |
| *entropy-solver (ceiling)* | *0.997* | *3.58* |

## Four findings

1. **The thesis holds.** Feedback-learned geometry (SupCon 0.75, MDS 0.66) hugely
   beats random geometry (0.31) under an *identical* controller. The geometry,
   not the controller, carries the competence. The bar ("beat random + beat
   semantic") is cleared decisively.

2. **Semantics is worse than random.** GloVe (0.17) underperforms random geometry
   (0.31). Linguistic meaning is *anti-correlated* with Wordle's combinatorial
   structure — synonyms are close in meaning but far in letters, so a semantic
   geometry actively misleads the movement. The original "meaningful embedding"
   intuition is falsified, in an interesting direction.

3. **Local structure is what matters.** SupCon (which explicitly tightens each
   word's top-k feedback-neighbors) beats closed-form MDS, while gradient RBF
   reconstruction (0.38) underperforms MDS (0.66) despite fitting a similar
   global target — it collapses onto the dominant `S=0` mass and under-resolves
   the rare high-similarity neighborhoods the cell centroids depend on. What
   helps is faithfully encoding *local consistency neighborhoods*.

4. **Movement is genuinely "downhill" — mechanistic evidence.** Normalized
   per-turn distance `||p - E[secret]||` (turn 1 = 1.0):
   - MDS: 1.0 → 0.83 → 0.68 → 0.58 → **0.53** (monotone convergence)
   - SupCon: 1.0 → 0.84 → 0.68 → 0.61 → **0.59** (monotone convergence)
   - random: 1.0 → 0.95 → 0.88 → 0.86 → 0.89 → **0.91** (sags, then drifts back)
   - semantic: 1.0 → 0.98 → ... → **0.96** (barely moves)

   The position homes in on the secret only for the feedback geometries; the
   controls wander. This convergence split maps exactly onto the win-rate split.

## Amortization

The entropy solver (expensive per-move search) wins 99.7% in ~3.58 guesses. The
best frozen-geometry agent (SupCon) recovers ~75% of games in ~4.9 guesses with
**zero per-move search** — a frozen vector-add + nearest-neighbor lookup. So a
large fraction of full-search competence *can* be amortized into static geometry,
though a real gap to optimality remains (the natural M4 target: differentiable
play-and-move training, or the 13k-guess list / information-gain readout).

## Honesty contract (why the win is meaningful)

- Geometry frozen at play-time; controller sees only `(position, guess, pattern)`.
- Movement = a frozen `(g, f)` lookup baked once from `(P, E)`; zero learned
  play-time parameters; no `P` read and no candidate filtering during play.
- Enforced by a test asserting `src/controller.py` imports neither
  `src.patterns` nor `src.similarity`.

## M4 attempts (closing the gap to the entropy solver) — two clean negatives

The gap from SupCon's 0.75 to the entropy solver's 0.997 was diagnosed first:
SupCon wins **0.73 @ 6 guesses, 0.87 @ 8, 0.93 @ 10, 0.98 @ 15** — it is *pointed
correctly* but converges too slowly to fit inside 6 guesses. Two cheap,
frozen-contract-respecting levers were tested on the eval split:

1. **Information-aware readout** (`src/m4.py`): bake a per-word probe-quality
   score (entropy of the word's pattern distribution) at build time, then read
   out the highest-scoring word among the k-nearest to the position.
   **Result: monotonically WORSE.** win@6 fell 0.75 → 0.51 (k=3) → 0.27 (k=8) →
   0.19 (k=12). *Why:* the position vector already encodes the agent's belief
   about the answer; the nearest word IS that belief. Substituting an
   informative-but-farther probe discards the belief the geometry built — at
   large k the agent just plays generic high-entropy openers and collapses
   toward the random-geometry floor. **The information lives in the position,
   not in a separate probe score; the readout should stay greedy-nearest.**

2. **eta schedule** (commit harder over turns): no schedule beat flat eta=0.7
   (0.746). Increasing schedules were all slightly worse (0.69–0.72). Flat
   eta=0.7 is already near-optimal for this readout/geometry.

**Conclusion:** the residual gap is not a readout or step-size problem. It is
that a single frozen movement carries limited combinatorial information per turn.
Closing it requires the heavy lever — **differentiable play-and-move training**
that retrains E end-to-end on the win objective so the geometry itself learns to
converge inside 6 turns. Left as future work (M5).

## Limitations / honest caveats

- The cell-silhouette diagnostic was **inconclusive** (all near zero): raw
  243-way Euclidean cluster separation does not predict success; the *local
  convergence behavior* does. Reported as a negative result, not overclaimed.
- A debugging episode mattered: the first controller used a pattern-only arrow
  table (averaged over all guesses) and won 0% — fixed by keying the frozen
  table on `(guess, pattern)`. See git history.
- Answers-only, greedy nearest-answer readout. The 13k-guess list and an
  information-gain-aware readout are untested and would likely raise the ceiling.
