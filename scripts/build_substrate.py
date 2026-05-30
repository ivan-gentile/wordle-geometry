"""Build and cache the substrate: pattern matrix P and Relation-B similarity S.

Run once; downstream geometry builders load the cached arrays.

    micromamba run -p ./.venv python -m scripts.build_substrate
"""
import os
import time

import numpy as np

from src.wordle import load_answers
from src.patterns import get_pattern_matrix
from src.similarity import relation_b_similarity

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
P_PATH = os.path.join(DATA, "pattern_matrix.npy")
S_PATH = os.path.join(DATA, "similarity_S.npy")


def main() -> None:
    words = load_answers()
    print(f"loaded {len(words)} answers")

    t0 = time.time()
    P = get_pattern_matrix(words, cache_path=P_PATH)
    print(f"pattern matrix {P.shape} ready in {time.time() - t0:.1f}s -> {P_PATH}")

    t0 = time.time()
    S = relation_b_similarity(P).astype(np.float32)
    np.save(S_PATH, S)
    print(f"similarity {S.shape} ready in {time.time() - t0:.1f}s -> {S_PATH}")
    off = S[~np.eye(len(words), dtype=bool)]
    print(f"off-diagonal similarity: mean={off.mean():.4f} max={off.max():.4f}")


if __name__ == "__main__":
    main()
