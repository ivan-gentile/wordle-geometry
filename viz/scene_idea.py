"""SCENE 1 - "The Idea": Wordle skill stored as geometry.

A small cloud of labeled 5-letter word-dots, a glowing belief dot at the centroid,
one guess -> colored feedback tiles -> a fixed nudge arrow that moves the belief,
nearest word updates. Loop caption + tagline. No per-move search.
"""
from manim import *

# Wordle feedback colors (gray / yellow / green)
WORDLE_GRAY = "#787c7e"
WORDLE_YELLOW = "#c9b458"
WORDLE_GREEN = "#6aaa64"
BELIEF_COLOR = "#ffd54f"
NEAREST_COLOR = "#6aaa64"


class TheIdea(Scene):
    def construct(self):
        self.camera.background_color = "#11141a"

        # ---- Title ----
        title = Text("The Idea", weight=BOLD, color=WHITE).scale(0.95).to_edge(UP, buff=0.35)
        self.play(Write(title))

        # ---- Word cloud: ~6 real 5-letter words as Text dots scattered in 2D ----
        words = ["CRANE", "SLATE", "TRACE", "GRACE", "PLATE", "BLADE"]
        positions = [
            np.array([-4.2, 1.4, 0.0]),
            np.array([-2.3, -1.6, 0.0]),
            np.array([-0.2, 1.9, 0.0]),
            np.array([2.1, 0.4, 0.0]),
            np.array([3.9, -1.4, 0.0]),
            np.array([0.6, -1.9, 0.0]),
        ]

        word_dots = {}
        word_labels = {}
        word_group = VGroup()
        for w, p in zip(words, positions):
            dot = Dot(point=p, radius=0.10, color="#9aa4b2")
            label = Text(w, color="#c8d0dc").scale(0.42).next_to(dot, UP, buff=0.12)
            word_dots[w] = dot
            word_labels[w] = label
            word_group.add(dot, label)

        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in word_group], lag_ratio=0.12),
                  run_time=2.0)

        # ---- Belief dot at the centroid, glowing ----
        centroid = np.mean(positions, axis=0)
        belief = Dot(point=centroid, radius=0.16, color=BELIEF_COLOR)
        glow = Circle(radius=0.34, color=BELIEF_COLOR, fill_opacity=0.25, stroke_width=0)
        glow.move_to(centroid)
        belief_label = Text("belief", color=BELIEF_COLOR).scale(0.36).next_to(belief, DOWN, buff=0.12)

        self.play(GrowFromCenter(glow), GrowFromCenter(belief), FadeIn(belief_label))
        self.play(glow.animate.scale(1.25).set_opacity(0.12), rate_func=there_and_back, run_time=0.8)

        def nearest_word(point):
            best, bd = None, 1e9
            for w, p in zip(words, positions):
                d = np.linalg.norm(p - point)
                if d < bd:
                    bd, best = d, w
            return best

        # ---- First guess: highlight nearest word ----
        g1 = nearest_word(centroid)
        ring1 = Circle(radius=0.30, color=NEAREST_COLOR, stroke_width=4).move_to(word_dots[g1])
        self.play(Create(ring1),
                  word_labels[g1].animate.set_color(NEAREST_COLOR).scale(1.1),
                  word_dots[g1].animate.set_color(NEAREST_COLOR))

        # ---- Feedback row: 5 colored Wordle tiles ----
        # pattern for the guessed word (gray / yellow / green)
        pattern = [WORDLE_GREEN, WORDLE_GRAY, WORDLE_YELLOW, WORDLE_GRAY, WORDLE_GREEN]
        tiles = VGroup()
        for i, (ch, col) in enumerate(zip(g1, pattern)):
            sq = Square(side_length=0.5, fill_color=col, fill_opacity=1.0, stroke_color="#222629", stroke_width=2)
            letter = Text(ch, color=WHITE, weight=BOLD).scale(0.34).move_to(sq)
            tiles.add(VGroup(sq, letter))
        tiles.arrange(RIGHT, buff=0.08).to_edge(DOWN, buff=1.15)
        fb_caption = Text("read feedback", color="#c8d0dc").scale(0.32).next_to(tiles, UP, buff=0.12)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in tiles], lag_ratio=0.18), run_time=1.4)
        self.play(FadeIn(fb_caption))
        self.wait(0.3)

        # ---- Fixed pre-computed nudge arrow toward a new belief location ----
        new_belief_pt = centroid + np.array([1.7, -1.1, 0.0])  # fixed nudge from table T[g,f]
        nudge = Arrow(start=centroid, end=new_belief_pt, color=BELIEF_COLOR,
                      buff=0.0, stroke_width=6, max_tip_length_to_length_ratio=0.18)
        nudge_label = Text("fixed nudge", color=BELIEF_COLOR).scale(0.30)
        nudge_label.next_to(nudge.get_center(), UP, buff=0.18)
        self.play(GrowArrow(nudge), FadeIn(nudge_label))

        # move belief + glow along the arrow
        self.play(
            belief.animate.move_to(new_belief_pt),
            glow.animate.move_to(new_belief_pt).set_opacity(0.25),
            belief_label.animate.next_to(new_belief_pt, DOWN, buff=0.12),
            run_time=1.2,
        )
        self.play(FadeOut(nudge), FadeOut(nudge_label))

        # ---- Nearest word updates ----
        g2 = nearest_word(new_belief_pt)
        if g2 == g1:
            # ensure a visible change for the demo
            order = sorted(words, key=lambda w: np.linalg.norm(positions[words.index(w)] - new_belief_pt))
            g2 = order[1] if order[0] == g1 else order[0]
        # un-highlight old
        self.play(
            FadeOut(ring1),
            word_labels[g1].animate.set_color("#c8d0dc").scale(1 / 1.1),
            word_dots[g1].animate.set_color("#9aa4b2"),
        )
        ring2 = Circle(radius=0.30, color=NEAREST_COLOR, stroke_width=4).move_to(word_dots[g2])
        self.play(
            Create(ring2),
            word_labels[g2].animate.set_color(NEAREST_COLOR).scale(1.1),
            word_dots[g2].animate.set_color(NEAREST_COLOR),
        )
        self.wait(0.3)

        # ---- Loop caption ----
        self.play(FadeOut(tiles), FadeOut(fb_caption))
        loop = Text("guess nearest word  ->  read feedback  ->  nudge the belief  ->  repeat",
                    color="#e8edf3").scale(0.40).to_edge(DOWN, buff=0.9)
        no_search = Text("no search  -  just a fixed, pre-computed nudge",
                         color="#9aa4b2").scale(0.34).next_to(loop, DOWN, buff=0.18)
        self.play(Write(loop))
        self.play(FadeIn(no_search))
        self.wait(0.6)

        # ---- Tagline ----
        self.play(
            FadeOut(word_group), FadeOut(belief), FadeOut(glow), FadeOut(belief_label),
            FadeOut(ring2), FadeOut(loop), FadeOut(no_search), FadeOut(title),
        )
        tagline = Text("Skill stored as geometry.", weight=BOLD, color=BELIEF_COLOR).scale(1.0)
        self.play(Write(tagline))
        self.wait(1)
