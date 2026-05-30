# Wordle as Geometry: Can a Frozen Embedding + Trivial Movement Play Wordle?

**Date:** 2026-05-30
**Author:** Ivan Gentile (with Claude Code)
**Status:** Design approved, implementation in progress

---

## 1. Thesis

Wordle competence is usually expressed as *per-move combinatorial search* (e.g. 3b1b's
entropy solver recomputes information gain every turn). This experiment asks whether that
competence can instead be **amortized once into the geometry of an embedding space**, such
that at play-time the agent does something trivially cheap:

> **A single frozen embedding `E: word → R^d` plus a fixed 243-entry "pattern → direction"
> table, driving a memoryless position vector by simple vector addition, with
> nearest-neighbor readout and NO per-move search and NO consistency filtering, can play
> Wordle far better than chance — and the quality of play scales with how well `E` encodes
> the game's feedback structure.**

The payoff is the **geometry itself**: what does a space look like in which "reasoning about
letters" becomes "moving downhill"? This is the representation-quality question from the
author's CLXAI work (feature attribution for contrastive learning) transplanted to a
combinatorial game.

### Why this is not circular

The danger is that a "success" is secretly the entropy solver in disguise. Three structural
rules prevent this (the **honesty contract**, §3):

1. **Geometry frozen at play-time** — no re-fitting, no per-game learning.
2. **Controller sees only `(p, last guess g, feedback f)`** — never the surviving-candidate
   set, never a per-turn consistency recomputation, never `P` at play-time.
3. **Δ-rule is a frozen 243-entry lookup table** with zero learned play-time parameters.

If the geometry carries the competence, this near-trivial controller suffices. If *no* fixed
geometry + trivial controller can do it, that is itself a sharp, publishable statement about
the limits of embedding combinatorial structure as a fixed metric.

---

## 2. Scope (first experiment)

- **Word list:** answers-only, the 2,309 official Wordle answers. Agent guesses only real
  answers (one list, cleanest geometry). The ~13k allowed-guess list is a later ablation.
- **Feedback:** real Wordle rules, including the duplicate-letter subtlety (ternary patterns
  0=gray, 1=yellow, 2=green; 3^5 = 243 patterns).
- **Success bar (headline):** the feedback-learned geometry + trivial controller must
  (a) crush a random baseline and (b) clearly beat a semantic-embedding control. Full
  guess-count distribution always reported. Optimality (matching the entropy solver) is a
  later, optional rung.

---

## 3. Architecture & the honesty contract

```
BUILD TIME (expensive, done once)
  Pattern matrix P [2309 x 2309], ternary           (3b1b generate_pattern_matrix, vectorized)
        |
        v
  Relation-B similarity S[a,b] = mean_g 1[P[g,a] == P[g,b]]
        |
        v
  Geometry E: word -> R^d   (FROZEN)   built 3 ways:  MDS | contrastive | semantic(control) | random(control)
        |
        v
  243-entry direction table  delta_f   (FROZEN)       built from P and E only

PLAY TIME (trivial, per game)  -- imports NO consistency function
  p <- centroid(E)                                    (no candidate set in memory)
  repeat up to 6x:
     g = argmin_{w in answers} || E[w] - p ||          (nearest answer-word)
     f = wordle_feedback(g, secret)                    (0/1/2 per letter)
     p <- (1-eta_t) p + eta_t (E[g] + delta_f)         (FIXED rule, no search, no P lookup)
     stop if f == all-green
```

### The honesty contract (spine of the experiment)

1. Geometry frozen at play-time.
2. Controller sees only `(p, g, f)`. Never the surviving-candidate set; never recomputes
   consistency; never reads `P` during play.
3. Δ-rule = frozen 243-arrow table. Zero learned play-time params. Logged and asserted in code
   (the play module must not import the consistency/pattern code path).
4. Nearest-neighbor readout only. No information-gain tie-break (that is an explicit later
   ablation).

**Honest subtlety, explicitly allowed:** the agent holds the static `word → vector` table (it
must, to emit a word). That is a fixed dictionary, like knowing the alphabet — inert. The cheat
would be *filtering* that table by consistency each turn. We never do; only `p` moves.

---

## 4. The geometry E

### 4a. Pattern matrix P
`P[i,j]` = ternary feedback when guessing `i` against secret `j`. Correct Wordle duplicate
handling. Vectorized; computed once; cached to `data/pattern_matrix.npy`.

### 4b. Relation-B similarity (load-bearing)
```
S[a,b] = (1/|G|) * sum_{g in G} 1[ P[g,a] == P[g,b] ]
```
Fraction of probe words `g` that give identical feedback for secrets `a` and `b`.
`S ~ 1` => nearly indistinguishable (GRAZE/GRAVE); `S ~ 0` => almost always separated.
`G` = the 2309 answers. This is the "which beliefs is the answer consistent with" geometry.

### 4c. Four geometries from the same S (geometry-source is the independent variable)
| Geometry | Construction | Role |
|---|---|---|
| **random** | each word -> random R^d vector (ignores P) | **load-bearing control**: floor for "any geometry + this rule" |
| **semantic** | GloVe/word2vec restricted to 2309 answers (ignores P) | control: tests the linguistic "meaningful embedding" intuition |
| **MDS** | classical MDS / eigendecomp of D = 1 - S | baseline: does the structure embed at all? closed-form, interpretable |
| **contrastive** | positives = high-S pairs, negatives = low-S pairs; SupCon/triplet (CLXAI losses) | **main method**: shapes consistency cells into convex blobs |

Same `d`, same Δ-rule, same eval harness for all four.

### 4d. The strict Δ-rule (frozen 243-arrow table)
Play-time: `p <- (1-eta_t) p + eta_t (E[g] + delta_f)`. Table built once at build time:
- **guess-relative (main):** `delta_f` such that "from a guess at `E[g]`, feedback `f`
  implies the secret lies near `E[g] + delta_f`", estimated as
  `delta_f = mean_{(g,a): P[g,a]=f} (E[a] - E[g])`.
- **context-free (sanity rung):** one global arrow per pattern, `delta_f = mean (E[a] - E[g])`
  applied as `p <- p + eta_t delta_f`.

`eta_t`: fixed scalar schedule (constant or mildly increasing), swept on the TUNE split, then
frozen. Zero learned play-time parameters. The 243 arrows are derived from `E`, so they inherit
whatever competence `E` has — the point of the experiment.

---

## 5. Evaluation

### 5a. Threat model -> control
| Attack | Control that defuses it |
|---|---|
| "small list / easy game" | **random-guess** and **first-letter-frequency greedy** agents (floor) |
| "any embedding + this rule wins; geometry does nothing" | **random-geometry** agent (THE key control). Gap vs. feedback-geometry IS the result. |
| "semantics would have worked" | **semantic-geometry** agent (GloVe). Its place on the spectrum is a finding either way. |
| "Δ-rule is the entropy solver in disguise" | strict contract; assert zero play-time params and no `P`/consistency import in the play module |
| "tuned eta on the test set" | **tune/eval split** of the 2309; all hyperparams frozen on tune, reported on held-out eval |

### 5b. Upper reference
Constrained **entropy solver** (3b1b method, answers-only, greedy one-step) — the
"expensive search every turn" agent. Not a required bar; the gap quantifies how much
competence the geometry amortized.

### 5c. Metrics (full distribution, never just a mean)
1. **Win rate** within 6 guesses, plus an **uncapped** variant (does movement even point the
   right way given unlimited turns?).
2. **Guess-count distribution** (histogram 1..6+, mean, median).
3. **Convergence geometry:** per-turn `||p - E[secret]||` trace, averaged. Monotone decrease =
   "moving downhill" is real. The mechanistic evidence.
4. **Cell convexity diagnostic:** silhouette of pattern-induced clusters in `E` (learned vs.
   random). Ties *why* to *what*.

### 5d. Decisive figure
Win-rate and mean-guesses **as a function of geometry-source**
(random -> semantic -> MDS -> contrastive), same Δ-rule throughout, entropy-solver ceiling as a
dashed line. Bars rising left-to-right + random-geometry pinned low ⟹ thesis holds:
*competence scales with how well the geometry encodes feedback structure.*

### 5e. Reproducibility / scale (Pandora)
2309^2 pattern matrix and `d in {32..128}` embedding fit trivially in RAM. Contrastive training
is seconds-to-minutes on one L40S (CPU fine for smoke). MDS needs no GPU. Seeds fixed and
logged. Smoke test runs in the login shell (<1 min, per machine etiquette); longer runs go
through `srun --gres=gpu:1`.

---

## 6. Module boundaries (each unit: one purpose, testable in isolation)

| Module | Purpose | Depends on |
|---|---|---|
| `src/wordle.py` | feedback rules (ternary pattern, duplicate handling), word-list loading | numpy |
| `src/patterns.py` | vectorized pattern matrix P; cache to disk | wordle, numpy |
| `src/similarity.py` | Relation-B similarity S from P | patterns, numpy |
| `src/geometries.py` | build E: random / semantic / mds / contrastive (returns word->vec table) | similarity, torch, CLXAI-style losses |
| `src/controller.py` | frozen 243-arrow table builder + strict play-time Δ-rule. **No P/consistency at play-time.** | numpy only (table is pre-baked) |
| `src/agents.py` | play loop; baseline agents (random, freq-greedy); entropy solver reference | wordle, controller |
| `src/evaluate.py` | run agents over eval split, collect metrics + traces | agents |
| `scripts/build_all.py` | build P, S, all geometries, tables; cache everything | all |
| `scripts/run_eval.py` | produce the decisive figure + tables | evaluate, matplotlib |
| `tests/` | smoke tests for every module; known-pattern fixtures (CRANE/CRATE etc.) | pytest |

**Critical isolation invariant:** `src/controller.py` (play-time) must not import
`src/patterns.py` or `src/similarity.py`. Enforced by a test that inspects imports.

---

## 7. Milestones

1. **M0 — substrate:** `wordle.py` + `patterns.py` + `similarity.py`, tests green, P cached.
2. **M1 — de-risk (MDS + controls):** MDS geometry, random + semantic geometries, strict
   controller, baselines, eval harness. Produce first version of the decisive figure.
3. **M2 — main method:** contrastive geometry (CLXAI losses). Add to the figure.
4. **M3 — analysis:** convergence-geometry plot, cell-convexity diagnostic, entropy-solver
   ceiling, write up findings.

Each milestone is independently runnable and produces a result (or a clean negative one).
