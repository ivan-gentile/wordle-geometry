"""SCENE 4 - Search vs. Memory.

Contrasts the entropy solver (expensive per-move search, animated as repeated
branching fan-outs; 99.7% win @ 3.58 guesses) with the geometry agent (one frozen
lookup table baked ONCE, then a trivial vector add at play time; 75% win @ 4.9).
All numbers from results_data.py.
"""
from manim import *


class Amortization(Scene):
    def _fan(self, origin, color, n=5, length=1.1, spread=0.9, depth=2):
        """Build a small branching fan-out rooted at `origin`."""
        group = VGroup()
        tips = [(origin, PI / 2)]
        for d in range(depth):
            new_tips = []
            branch = max(2, n - d)
            for (pt, ang) in tips:
                for k in range(branch):
                    if branch == 1:
                        da = 0
                    else:
                        da = (k / (branch - 1) - 0.5) * spread
                    a = ang + da
                    end = pt + length * np.array([np.cos(a), np.sin(a), 0])
                    line = Line(pt, end, color=color, stroke_width=2.2)
                    line.set_opacity(0.85 - 0.18 * d)
                    group.add(line)
                    new_tips.append((end, a))
            tips = new_tips
            length *= 0.62
        for (pt, _ang) in tips:
            group.add(Dot(pt, radius=0.03, color=color))
        return group

    def construct(self):
        # ---- Header ----
        header = Text("Search vs. Memory", color=WHITE, weight=BOLD).scale(0.95)
        header.to_edge(UP, buff=0.35)
        self.play(Write(header))

        # ---- Divider ----
        divider = DashedLine(
            header.get_bottom() + DOWN * 0.25 + LEFT * 0,
            np.array([0, -3.4, 0]),
            color=GREY_B,
            stroke_width=1.5,
        ).set_opacity(0.5)
        # vertical divider down the middle
        divider = DashedLine(
            np.array([0, header.get_bottom()[1] - 0.15, 0]),
            np.array([0, -3.4, 0]),
            color=GREY_B,
            stroke_width=1.5,
        ).set_opacity(0.5)

        # ---- Left column: entropy solver (search every move) ----
        left_title = Text("Entropy solver", color="#9fd0ff", weight=BOLD).scale(0.6)
        left_sub = Text("search every move", color="#9fd0ff").scale(0.42)
        left_title.move_to(np.array([-3.55, 1.95, 0]))
        left_sub.next_to(left_title, DOWN, buff=0.12)

        # ---- Right column: geometry agent (compute once, then add) ----
        right_title = Text("Geometry agent", color="#7ee08a", weight=BOLD).scale(0.6)
        right_sub = Text("compute once, then add", color="#7ee08a").scale(0.42)
        right_title.move_to(np.array([3.55, 1.95, 0]))
        right_sub.next_to(right_title, DOWN, buff=0.12)

        self.play(
            Create(divider),
            FadeIn(left_title, shift=DOWN * 0.2),
            FadeIn(left_sub, shift=DOWN * 0.2),
            FadeIn(right_title, shift=DOWN * 0.2),
            FadeIn(right_sub, shift=DOWN * 0.2),
        )

        # ----- LEFT: repeated fan-out, one per turn -----
        turn_label = Text("turn 1", color="#9fd0ff").scale(0.4)
        turn_label.move_to(np.array([-3.55, 0.9, 0]))
        self.play(FadeIn(turn_label))

        fan_origin = np.array([-3.55, -1.25, 0])
        for t in range(3):
            fan = self._fan(fan_origin, "#9fd0ff", n=5, length=0.95, depth=2)
            new_label = Text(f"turn {t + 1}", color="#9fd0ff").scale(0.4)
            new_label.move_to(turn_label.get_center())
            self.play(
                Transform(turn_label, new_label),
                Create(fan, run_time=0.55),
            )
            self.play(FadeOut(fan, run_time=0.32))

        cost_left = Text("expensive each turn", color="#ff8a80").scale(0.42)
        cost_left.move_to(np.array([-3.55, -1.55, 0]))
        self.play(FadeIn(cost_left, scale=1.1))

        # ----- RIGHT: fan-out ONCE -> frozen table -> trivial vector add -----
        bake_label = Text("bake once", color="#7ee08a").scale(0.4)
        bake_label.move_to(np.array([3.55, 0.9, 0]))
        big_fan = self._fan(np.array([3.55, -0.55, 0]), "#7ee08a", n=5, length=0.85, depth=2)
        self.play(FadeIn(bake_label), Create(big_fan, run_time=0.7))

        # collapse the fan into a frozen lookup-table icon
        table = VGroup()
        rows, cols = 4, 3
        cell = 0.3
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell, stroke_width=1.4, color="#7ee08a")
                sq.set_fill("#1f3d24", opacity=0.9)
                sq.move_to(
                    np.array([3.55 + (c - 1) * cell, -0.45 + (1.5 - r) * cell, 0])
                )
                table.add(sq)
        table_label = Text("frozen table T[g, f]", color="#7ee08a").scale(0.4)
        table_label.next_to(table, DOWN, buff=0.18)
        self.play(
            ReplacementTransform(big_fan, table),
            FadeOut(bake_label),
            run_time=0.8,
        )
        self.play(FadeIn(table_label))

        # play time: a single trivial vector add, with belief dot + arrow
        play_label = Text("at play time, just add an arrow:", color="#7ee08a").scale(0.36)
        play_label.move_to(np.array([3.55, -1.9, 0]))
        add_eq = MathTex(r"p \;\leftarrow\; p + T[g,f]", color="#7ee08a").scale(0.55)
        add_eq.move_to(np.array([3.0, -2.35, 0]))
        belief = Dot(np.array([4.55, -2.35, 0]), radius=0.06, color=YELLOW)
        arrow = Arrow(
            np.array([4.55, -2.35, 0]),
            np.array([5.65, -2.35, 0]),
            color="#7ee08a",
            buff=0,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.22,
        )
        self.play(FadeIn(play_label), Write(add_eq))
        self.play(Create(belief), GrowArrow(arrow), run_time=0.55)

        # ---- Score badges (symmetric, above the caption strip) ----
        left_score = VGroup(
            Text("99.7% win", color="#9fd0ff", weight=BOLD).scale(0.55),
            Text("3.58 guesses", color="#9fd0ff").scale(0.42),
        ).arrange(DOWN, buff=0.1)
        left_score.move_to(np.array([-3.55, -0.55, 0]))

        right_score = VGroup(
            Text("75% win", color="#7ee08a", weight=BOLD).scale(0.55),
            Text("4.9 guesses", color="#7ee08a").scale(0.42),
        ).arrange(DOWN, buff=0.1)
        right_score.move_to(np.array([3.55, 0.85, 0]))

        self.play(FadeIn(left_score, scale=1.1), FadeIn(right_score, scale=1.1))

        # ---- Takeaway caption ----
        caption = Text(
            "A frozen geometry recovers ~3/4 of full-search skill\n"
            "with almost no thinking at play time.",
            color=WHITE,
            line_spacing=0.8,
        ).scale(0.46)
        caption.to_edge(DOWN, buff=0.18)
        box = SurroundingRectangle(caption, color="#7ee08a", buff=0.18, corner_radius=0.1)
        box.set_fill("#0e1b12", opacity=0.85)
        self.play(FadeIn(box), Write(caption))

        self.wait(1)
