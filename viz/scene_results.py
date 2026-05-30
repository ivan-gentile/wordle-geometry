"""SCENE 2 - "The Result": animated Win@6 bar chart by geometry."""
from manim import *


class ResultsBars(Scene):
    def construct(self):
        # ---- data (bars to grow, left to right) ----
        bars_data = [
            ("Random\nguessing", 0.002, "#bbbbbb"),
            ("Random\ngeometry", 0.306, "#d9a679"),
            ("Semantic\n(GloVe)", 0.171, "#7fa8d9"),
            ("MDS\n(feedback)", 0.663, "#4caf50"),
            ("Contrastive\n(SupCon)", 0.746, "#1b5e20"),
        ]
        ceiling = 0.997  # entropy-solver reference line

        # ---- header ----
        title = Text("The Result: Win@6 by geometry", color=GREEN).scale(0.7)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ---- axes ----
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 25],
            x_length=9.5,
            y_length=4.6,
            axis_config={"include_tip": False, "color": GREY_B},
            x_axis_config={"include_ticks": False},
            y_axis_config={"include_numbers": False},
        )
        axes.shift(DOWN * 0.55)

        # y-axis label
        y_label = Text("win rate", color=GREY_B).scale(0.4)
        y_label.rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.15)

        # y-axis ticks/labels 0-100%
        y_ticks = VGroup()
        for pct in [0, 25, 50, 75, 100]:
            p = axes.c2p(0, pct)
            lab = Text(f"{pct}%", color=GREY_B).scale(0.35)
            lab.next_to(p, LEFT, buff=0.2)
            tick = Line(p + LEFT * 0.08, p + RIGHT * 0.08, color=GREY_B, stroke_width=2)
            y_ticks.add(lab, tick)

        self.play(Create(axes.x_axis), Create(axes.y_axis), FadeIn(y_label), FadeIn(y_ticks))

        # ---- helper geometry for bars ----
        n = len(bars_data)
        slot = 5.0 / n           # width of each x slot in data units
        bar_w_data = slot * 0.55  # bar width in data units

        baseline_y = axes.c2p(0, 0)[1]

        bars = VGroup()
        value_labels = VGroup()
        cat_labels = VGroup()

        for i, (name, val, color) in enumerate(bars_data):
            cx = (i + 0.5) * slot                      # center x in data units
            top = axes.c2p(cx, val * 100)
            bot = axes.c2p(cx, 0)
            left_x = axes.c2p(cx - bar_w_data / 2, 0)[0]
            right_x = axes.c2p(cx + bar_w_data / 2, 0)[0]
            width = right_x - left_x
            height = max(top[1] - bot[1], 0.001)

            bar = Rectangle(
                width=width,
                height=height,
                fill_color=color,
                fill_opacity=1.0,
                stroke_color=color,
                stroke_width=1,
            )
            # anchor bottom edge at baseline so growth reads as rising
            bar.move_to([bot[0], baseline_y + height / 2, 0])
            bars.add(bar)

            vlab = Text(f"{val * 100:.1f}%", color=WHITE).scale(0.4)
            vlab.next_to(bar, UP, buff=0.12)
            value_labels.add(vlab)

            clab = Text(name, color=GREY_A, line_spacing=0.7).scale(0.34)
            clab.next_to(bot, DOWN, buff=0.18)
            cat_labels.add(clab)

        # ---- dashed ceiling reference line ----
        ceil_y = axes.c2p(0, ceiling * 100)[1]
        x_start = axes.c2p(0, ceiling * 100)[0]
        x_end = axes.c2p(5, ceiling * 100)[0]
        ceil_line = DashedLine(
            [x_start, ceil_y, 0], [x_end, ceil_y, 0],
            color="#dddddd", stroke_width=2.5, dash_length=0.12,
        )
        ceil_label = Text("Entropy-solver ceiling 99.7%", color="#dddddd").scale(0.36)
        ceil_label.next_to(ceil_line, UP, buff=0.08).align_to(ceil_line, RIGHT)

        self.play(Create(ceil_line), FadeIn(ceil_label), run_time=0.9)

        # ---- grow bars left to right ----
        for i in range(n):
            self.play(
                GrowFromEdge(bars[i], DOWN),
                FadeIn(value_labels[i], shift=UP * 0.2),
                FadeIn(cat_labels[i]),
                run_time=0.6,
            )

        # ---- callout 1: feedback beats random by ~2.4x, same trivial rule ----
        callout1 = VGroup(
            Text("MDS feedback 66% vs Random 31%", color="#4caf50").scale(0.42),
            Text("~2.4x better with the SAME trivial rule", color=GREY_A).scale(0.38),
        ).arrange(DOWN, buff=0.12)
        callout1.to_corner(UR, buff=0.35).shift(DOWN * 0.6)
        self.play(FadeIn(callout1, shift=LEFT * 0.3))

        # highlight feedback bars
        self.play(
            Indicate(bars[3], color=YELLOW, scale_factor=1.06),
            Indicate(bars[4], color=YELLOW, scale_factor=1.06),
            run_time=1.0,
        )
        self.wait(0.4)
        self.play(FadeOut(callout1))

        # ---- callout 2: the surprise, Semantic < Random ----
        callout2 = VGroup(
            Text("Surprise: Semantic 17% < Random 31%", color="#7fa8d9").scale(0.42),
            Text('"meaning is the wrong kind of', color=GREY_A).scale(0.36),
            Text('similarity for Wordle"', color=GREY_A).scale(0.36),
        ).arrange(DOWN, buff=0.1)
        callout2.to_corner(UR, buff=0.35).shift(DOWN * 0.6)
        self.play(FadeIn(callout2, shift=LEFT * 0.3))

        self.play(
            Indicate(bars[2], color="#7fa8d9", scale_factor=1.08),
            Indicate(bars[1], color="#d9a679", scale_factor=1.08),
            run_time=1.0,
        )
        self.wait(0.6)

        self.wait(1)
