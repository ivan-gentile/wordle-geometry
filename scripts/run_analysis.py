"""M3 analysis: convergence trace, cell-convexity diagnostic, entropy ceiling.

Produces:
  data/../convergence_figure.png  -- per-turn ||p - E[secret]|| for each geometry
  data/../convexity_figure.png    -- silhouette of consistency cells per geometry
  console: entropy-solver upper reference on the eval split

    micromamba run -p ./.venv python -m scripts.run_analysis
"""
import os

import numpy as np

from src.wordle import load_answers
from src.patterns import get_pattern_matrix
from src.geometries import random_geometry, mds_geometry, semantic_geometry
from src.controller import build_cell_centroid_table, WordleNavigator
from src.agents import play_geometry_game
from src.analysis import mean_convergence_curve, cell_convexity_score
from src.entropy_solver import entropy_solver_game
from src.evaluate import split_secrets

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 0
MAXG = 6


def load_glove_matrix(words, d):
    z = np.load(os.path.join(DATA, "glove_vocab_100d.npz"), allow_pickle=True)
    M = z["vectors"]
    vectors = {w: M[i] for i, w in enumerate(words) if not np.isnan(M[i]).any()}
    return semantic_geometry(words, vectors, d=d, seed=SEED)


def main():
    words = load_answers()
    P = get_pattern_matrix(words, cache_path=os.path.join(DATA, "pattern_matrix.npy"))
    S = np.load(os.path.join(DATA, "similarity_S.npy")).astype(np.float64)
    _, evl = split_secrets(words, frac_tune=0.5, seed=SEED)

    # geometries at their best eval dims (from run_eval), with best eta
    geoms = {
        "random-geom": (random_geometry(len(words), d=128, seed=SEED), 1.0, "#d9a679"),
        "semantic-glove": (load_glove_matrix(words, 128), 1.0, "#7fa8d9"),
        "mds": (mds_geometry(S, d=64, seed=SEED), 0.85, "#4caf50"),
    }
    for kind, d in [("recon", 64), ("supcon", 32)]:
        path = os.path.join(DATA, f"geom_{kind}_d{d}.npy")
        if os.path.exists(path):
            eta = 1.0 if kind == "recon" else 0.7
            col = "#2e7d32" if kind == "recon" else "#1b5e20"
            geoms[f"contrastive-{kind}"] = (np.load(path), eta, col)

    # ---- convergence curves + convexity ----
    curves, convex = {}, {}
    sample = evl[:300]  # 300 held-out secrets is enough for a stable mean curve
    for name, (E, eta, _col) in geoms.items():
        nav = WordleNavigator(E, cell_table=build_cell_centroid_table(P, E))
        traces = [play_geometry_game(nav, words, s, MAXG, eta, mode="cell").distance_trace
                  for s in sample]
        # normalize each trace by its turn-1 distance so geometries are comparable
        norm = []
        for tr in traces:
            if tr and tr[0] > 0:
                norm.append([x / tr[0] for x in tr])
        curves[name] = mean_convergence_curve(norm, max_len=MAXG)
        # convexity over a set of probe guesses
        probes = [words.index(w) for w in ["slate", "crane", "raise", "adieu"] if w in words]
        convex[name] = float(np.nanmean([cell_convexity_score(P, E, pr) for pr in probes]))
        print(f"{name:22s} convergence(norm)={[round(x,2) for x in curves[name]]}  "
              f"cell-silhouette={convex[name]:.3f}")

    # ---- entropy-solver ceiling on the eval split ----
    es = [entropy_solver_game(P, words, s, MAXG) for s in evl]
    es_win = np.mean([r.won for r in es])
    es_mg = np.mean([r.n_guesses if r.won else MAXG + 1 for r in es])
    print(f"\nENTROPY-SOLVER ceiling (eval): win@6={es_win:.3f}  mean_all={es_mg:.2f}")

    _figures(curves, convex, geoms)


def _figures(curves, convex, geoms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # convergence
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (_E, _eta, col) in geoms.items():
        y = curves[name]
        ax.plot(range(1, len(y) + 1), y, marker="o", label=name, color=col)
    ax.set_xlabel("guess number")
    ax.set_ylabel("normalized ||p - E[secret]||  (turn 1 = 1.0)")
    ax.set_title("Does the position move downhill toward the secret?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(DATA, "..", "convergence_figure.png")
    fig.savefig(out1, dpi=130); print(f"figure -> {os.path.abspath(out1)}")

    # convexity vs (recall win rates are in the decisive figure)
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(convex.keys())
    ax.bar(names, [convex[n] for n in names],
           color=[geoms[n][2] for n in names])
    ax.set_ylabel("mean consistency-cell silhouette")
    ax.set_title("Cell convexity by geometry (higher = cleaner cells)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out2 = os.path.join(DATA, "..", "convexity_figure.png")
    fig.savefig(out2, dpi=130); print(f"figure -> {os.path.abspath(out2)}")


if __name__ == "__main__":
    main()
