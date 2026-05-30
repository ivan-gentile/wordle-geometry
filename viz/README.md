# viz — Manim animations + public page for the Wordle-Geometry results

Public-facing visualization built with [Manim CE](https://www.manim.community/) (via
[HarleyCoops/Math-To-Manim](https://github.com/HarleyCoops/Math-To-Manim)). We bypass
Math-To-Manim's LLM generation pipeline and hand-author scenes directly for *our* results,
then render with Manim CE — no paid API needed.

## What's here

- `index.html` — self-contained, offline public page explaining the results to a general
  audience. Embeds the four videos from `site_media/`.
- `site_media/*.mp4` — the four rendered scenes (720p):
  - `idea.mp4` — **The Idea**: words as points, a "belief" dot, the guess→feedback→nudge loop.
  - `bars.mp4` — **The Result**: Win@6 by geometry (incl. semantic < random surprise).
  - `converge.mp4` — **Moving Downhill**: per-turn distance-to-answer; feedback geometries converge.
  - `amortize.mp4` — **Search vs. Memory**: per-move search vs. a frozen lookup.
- `scene_*.py` — the Manim source for each scene. `results_data.py` is the single source of
  truth for every number shown (from `../FINDINGS.md`).

## View

Open `index.html` in any browser (works offline). Or render from source:

```bash
# one-time env (no sudo): prebuilt cairo/pango binaries, then pure-python manim
micromamba create -y -p ./.manim-env python=3.11 ffmpeg cairo pango pkg-config -c conda-forge
micromamba install -y -p ./.manim-env -c conda-forge manimpango pycairo
micromamba run -p ./.manim-env pip install --no-build-isolation "manim==0.20.1"

# render a scene
micromamba run -p ./.manim-env manim -qm scene_results.py ResultsBars
```
