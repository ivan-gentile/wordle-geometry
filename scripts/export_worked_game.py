"""Export a real solver trajectory (default: YACHT) + 2D PCA projection to viz/worked_game.json.

Used to drive the hero animation. Everything is real output of the frozen SupCon agent.

    micromamba run -p ./.venv python -m scripts.export_worked_game [SECRET]
"""
import json
import os
import sys

import numpy as np

from src.wordle import load_answers, feedback, pattern_to_int, int_to_pattern, EXACT
from src.patterns import get_pattern_matrix
from src.controller import build_cell_centroid_table, WordleNavigator

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "viz", "worked_game.json")
ALL_GREEN = pattern_to_int((EXACT,) * 5)


def main(secret: str = "yacht", eta: float = 0.7, maxg: int = 6) -> None:
    words = load_answers()
    P = get_pattern_matrix(words, cache_path=os.path.join(DATA, "pattern_matrix.npy"))
    E = np.load(os.path.join(DATA, "geom_supcon_d32.npy"))
    nav = WordleNavigator(E, cell_table=build_cell_centroid_table(P, E))
    idx = {w: i for i, w in enumerate(words)}
    si = idx[secret]

    p = nav.initial_position()
    positions = [p.copy()]
    steps = []
    tried = set()
    for _ in range(maxg):
        gi = nav.nearest_word(p, exclude=tried)
        tried.add(gi)
        pat = pattern_to_int(feedback(words[gi], secret))
        steps.append(dict(gi=gi, word=words[gi],
                          codes=[int(c) for c in int_to_pattern(pat)],
                          dist=float(np.linalg.norm(p - E[si]))))
        if pat == ALL_GREEN:
            break
        p = nav.move_to_cell(p, gi, pat, eta)
        positions.append(p.copy())
    positions = np.array(positions)

    guessed = [s["gi"] for s in steps]
    d2 = np.sum((E - E[si]) ** 2, axis=1)
    neigh = [int(n) for n in np.argsort(d2)[:40]]
    ctx = sorted(set(guessed + [si] + neigh))
    X = np.vstack([E[ctx], positions])
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:2].T

    def proj(v):
        return [float(x) for x in ((v - mu) @ W)]

    data = dict(
        secret=secret.upper(),
        guesses=[dict(word=s["word"].upper(), codes=s["codes"], dist=round(s["dist"], 3)) for s in steps],
        secret_xy=proj(E[si]),
        guess_xy=[proj(E[s["gi"]]) for s in steps],
        belief_xy=[proj(p) for p in positions],
        context=[dict(word=words[i].upper(), xy=proj(E[i]),
                      is_secret=bool(i == si), is_guess=bool(i in guessed)) for i in ctx],
    )
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", OUT, "->", " -> ".join(g["word"] for g in data["guesses"]))


if __name__ == "__main__":
    main(sys.argv[1].lower() if len(sys.argv) > 1 else "yacht")
