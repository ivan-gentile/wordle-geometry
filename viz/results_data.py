"""Single source of truth for the numbers the Manim scenes visualize.

All figures are from the held-out eval split (1,158 secrets) with the same frozen
(guess, pattern) cell-centroid controller across geometries. See ../FINDINGS.md.
"""

# (label, win_rate, mean_guesses_all, color_hint)
GEOMETRY_RESULTS = [
    ("Random guessing", 0.002, 6.99, "#bbbbbb"),
    ("Letter-frequency", 0.002, 7.00, "#bbbbbb"),
    ("Random geometry", 0.306, 6.16, "#d9a679"),
    ("Semantic (GloVe)", 0.171, 6.46, "#7fa8d9"),
    ("MDS (feedback)", 0.663, 5.34, "#4caf50"),
    ("Contrastive (SupCon)", 0.746, 4.90, "#1b5e20"),
    ("Entropy solver (ceiling)", 0.997, 3.58, "#333333"),
]

# normalized per-turn distance to the secret embedding (turn 1 = 1.0)
CONVERGENCE = {
    "MDS (feedback)":        [1.00, 0.83, 0.68, 0.58, 0.53, 0.54],
    "Contrastive (SupCon)":  [1.00, 0.84, 0.68, 0.61, 0.59, 0.59],
    "Random geometry":       [1.00, 0.95, 0.88, 0.86, 0.89, 0.91],
    "Semantic (GloVe)":      [1.00, 0.98, 0.94, 0.93, 0.95, 0.96],
}

# most-confusable answer pairs by feedback similarity S (sanity check from M0)
CONFUSABLE_PAIRS = [
    ("boozy", "booby", 0.97),
    ("gauze", "gauge", 0.95),
    ("adobe", "abode", 0.95),
    ("jaunt", "vaunt", 0.92),
]

# SupCon "pointed right but slow": win rate vs guess budget
WIN_VS_BUDGET = {6: 0.73, 8: 0.87, 10: 0.93, 15: 0.98}

TITLE = "Wordle as Geometry"
SUBTITLE = "Can a frozen embedding + a trivial movement rule play Wordle?"
