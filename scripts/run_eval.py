"""Run the decisive comparison: geometry-source vs. Wordle-play quality.

Builds random / semantic(GloVe) / MDS geometries, the frozen (guess,pattern)
cell-centroid controller for each, sweeps (eta, d) on the TUNE split, and reports
the headline metrics on the disjoint EVAL split. Baselines: random-guess and
freq-greedy. Emits a results table and the decisive bar figure.

    micromamba run -p ./.venv python -m scripts.run_eval
"""
import os
import time

import numpy as np

from src.wordle import load_answers
from src.patterns import get_pattern_matrix
from src.geometries import random_geometry, mds_geometry, semantic_geometry
from src.contrastive import train_reconstruction, train_supcon
from src.controller import build_cell_centroid_table, WordleNavigator
from src.evaluate import (
    split_secrets, eval_geometry_agent, eval_random_agent, eval_freq_greedy, Metrics,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_GUESSES = 6
SEED = 0


def build_nav(E):
    return WordleNavigator(E, cell_table=build_cell_centroid_table(P_GLOBAL, E))


def load_glove_matrix(words, d):
    z = np.load(os.path.join(DATA, "glove_vocab_100d.npz"), allow_pickle=True)
    M = z["vectors"]  # (N, 100), nan for missing
    vectors = {w: M[i] for i, w in enumerate(words) if not np.isnan(M[i]).any()}
    return semantic_geometry(words, vectors, d=d, seed=SEED)


def cached_train(kind, S, d):
    """Train (or load cached) a contrastive geometry of the given kind/dim."""
    path = os.path.join(DATA, f"geom_{kind}_d{d}.npy")
    if os.path.exists(path):
        return np.load(path)
    if kind == "recon":
        E = train_reconstruction(S, d=d, epochs=800, seed=SEED)
    elif kind == "supcon":
        E = train_supcon(S, d=d, epochs=800, seed=SEED, k_pos=8)
    else:
        raise ValueError(kind)
    np.save(path, E)
    return E


def sweep_eta(nav, words, secrets, etas, mode="cell"):
    best = None
    for eta in etas:
        m, _ = eval_geometry_agent(nav, words, secrets, eta=eta, mode=mode,
                                   max_guesses=MAX_GUESSES, name=f"eta={eta}")
        # rank by win rate, then by mean guesses among wins
        key = (m.win_rate, -(m.mean_guesses_won if m.win_rate > 0 else 99))
        if best is None or key > best[0]:
            best = (key, eta, m)
    return best[1], best[2]


def fmt(m: Metrics) -> str:
    return (f"{m.name:24s}  win@6={m.win_rate:6.3f}  "
            f"mean_guesses(won)={m.mean_guesses_won:5.2f}  "
            f"mean_all={m.mean_guesses_all:5.2f}  n={m.n}")


def main():
    global P_GLOBAL
    words = load_answers()
    P_GLOBAL = get_pattern_matrix(words, cache_path=os.path.join(DATA, "pattern_matrix.npy"))
    S = np.load(os.path.join(DATA, "similarity_S.npy")).astype(np.float64)
    tune, evl = split_secrets(words, frac_tune=0.5, seed=SEED)
    print(f"vocab={len(words)}  tune={len(tune)}  eval={len(evl)}")

    etas = [0.4, 0.55, 0.7, 0.85, 1.0]
    dims = [32, 64, 128]

    results = {}

    # --- baselines (no tuning) ---
    results["random-guess"] = eval_random_agent(words, evl, MAX_GUESSES, seed=SEED)
    results["freq-greedy"] = eval_freq_greedy(words, evl, MAX_GUESSES)
    print(fmt(results["random-guess"]))
    print(fmt(results["freq-greedy"]))

    # --- geometry agents: tune (eta, d) on tune split, report on eval split ---
    for gname, builder in [
        ("random-geom", lambda d: random_geometry(len(words), d=d, seed=SEED)),
        ("semantic-glove", lambda d: load_glove_matrix(words, d)),
        ("mds", lambda d: mds_geometry(S, d=d, seed=SEED)),
        ("contrastive-recon", lambda d: cached_train("recon", S, d)),
        ("contrastive-supcon", lambda d: cached_train("supcon", S, d)),
    ]:
        t0 = time.time()
        best_overall = None
        for d in dims:
            E = builder(d)
            nav = build_nav(E)
            eta, m_tune = sweep_eta(nav, words, tune, etas)
            key = (m_tune.win_rate, -(m_tune.mean_guesses_won if m_tune.win_rate > 0 else 99))
            if best_overall is None or key > best_overall[0]:
                best_overall = (key, d, eta, E)
        _, d_best, eta_best, E_best = best_overall
        nav = build_nav(E_best)
        m_eval, res = eval_geometry_agent(nav, words, evl, eta=eta_best, mode="cell",
                                          max_guesses=MAX_GUESSES, name=gname)
        results[gname] = m_eval
        print(f"{fmt(m_eval)}   [best d={d_best} eta={eta_best}]  ({time.time()-t0:.1f}s)")
        # stash traces for MDS for the convergence plot later
        if gname == "mds":
            np.save(os.path.join(DATA, "mds_eval_traces.npy"),
                    np.array([r.distance_trace for r in res], dtype=object), allow_pickle=True)

    _make_figure(results)


def _make_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["random-guess", "freq-greedy", "random-geom", "semantic-glove", "mds",
             "contrastive-recon", "contrastive-supcon"]
    labels = [o for o in order if o in results]
    wins = [results[o].win_rate for o in labels]
    means = [results[o].mean_guesses_all for o in labels]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    color_map = {
        "random-guess": "#bbb", "freq-greedy": "#bbb", "random-geom": "#d9a679",
        "semantic-glove": "#7fa8d9", "mds": "#4caf50",
        "contrastive-recon": "#2e7d32", "contrastive-supcon": "#1b5e20",
    }
    colors = [color_map.get(o, "#888") for o in labels]
    ax[0].bar(labels, wins, color=colors[:len(labels)])
    ax[0].set_title("Win rate within 6 guesses (held-out eval)")
    ax[0].set_ylabel("win rate"); ax[0].set_ylim(0, 1)
    for i, v in enumerate(wins):
        ax[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].bar(labels, means, color=colors[:len(labels)])
    ax[1].set_title("Mean guesses (loss = 7); lower is better")
    ax[1].set_ylabel("mean guesses")
    for i, v in enumerate(means):
        ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Wordle competence scales with geometry quality (same frozen controller)")
    fig.tight_layout()
    out = os.path.join(DATA, "..", "decisive_figure.png")
    fig.savefig(out, dpi=130)
    print(f"figure -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
