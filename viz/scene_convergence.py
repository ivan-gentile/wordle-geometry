"""SCENE 3 - Moving Downhill.

Animates normalized distance-to-secret per turn for four geometries.
Feedback geometries (MDS, SupCon) dive toward ~0.55; controls wander.
Numbers from CONVERGENCE in results_data.py.
"""
from manim import *

CONVERGENCE = {
    "MDS (feedback)":        [1.00, 0.83, 0.68, 0.58, 0.53, 0.54],
    "Contrastive (SupCon)":  [1.00, 0.84, 0.68, 0.61, 0.59, 0.59],
    "Random geometry":       [1.00, 0.95, 0.88, 0.86, 0.89, 0.91],
    "Semantic (GloVe)":      [1.00, 0.98, 0.94, 0.93, 0.95, 0.96],
}

COLORS = {
    "MDS (feedback)":        "#4caf50",
    "Contrastive (SupCon)":  "#1b5e20",
    "Random geometry":       "#d9a679",
    "Semantic (GloVe)":      "#7fa8d9",
}


class Convergence(Scene):
    def construct(self):
        self.camera.background_color = "#0d0d0d"

        title = Text("Moving Downhill", color=GREEN).scale(0.8)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        axes = Axes(
            x_range=[1, 6, 1],
            y_range=[0.4, 1.05, 0.1],
            x_length=8.0,
            y_length=4.2,
            axis_config={"include_tip": False, "color": GREY_B},
            x_axis_config={"numbers_to_include": [1, 2, 3, 4, 5, 6]},
            y_axis_config={"numbers_to_include": [0.5, 0.7, 0.9]},
        )
        axes.shift(DOWN * 0.4 + LEFT * 0.6)

        x_label = Text("guess number", color=GREY_A).scale(0.4)
        x_label.next_to(axes.x_axis, DOWN, buff=0.35)
        y_label = Text("normalized distance to answer", color=GREY_A).scale(0.4)
        y_label.rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.3)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        # Draw order: controls first, then feedback geometries on top.
        draw_order = [
            "Semantic (GloVe)",
            "Random geometry",
            "Contrastive (SupCon)",
            "MDS (feedback)",
        ]

        legend_items = VGroup()
        for name in draw_order:
            ys = CONVERGENCE[name]
            color = COLORS[name]
            points = [axes.c2p(i + 1, y) for i, y in enumerate(ys)]

            dots = VGroup(*[Dot(p, radius=0.05, color=color) for p in points])
            segments = VGroup()
            for a, b in zip(points[:-1], points[1:]):
                segments.add(Line(a, b, color=color, stroke_width=4))

            # Plot point by point: dot, then connecting segment.
            self.play(GrowFromCenter(dots[0]), run_time=0.2)
            for k in range(len(segments)):
                self.play(
                    Create(segments[k]),
                    GrowFromCenter(dots[k + 1]),
                    run_time=0.25,
                )

        # Legend (built after curves so colors are anchored).
        for name in ["MDS (feedback)", "Contrastive (SupCon)",
                     "Random geometry", "Semantic (GloVe)"]:
            swatch = Line(ORIGIN, RIGHT * 0.45, color=COLORS[name], stroke_width=5)
            txt = Text(name, color=GREY_A).scale(0.34)
            txt.next_to(swatch, RIGHT, buff=0.15)
            legend_items.add(VGroup(swatch, txt))
        legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        legend_items.to_corner(UR, buff=0.4).shift(DOWN * 0.6)
        self.play(FadeIn(legend_items, shift=LEFT * 0.2))

        caption = Text(
            "Only the feedback geometries home in - the others wander.",
            color=WHITE,
        ).scale(0.42)
        caption.to_edge(DOWN, buff=0.25)
        self.play(Write(caption))

        self.wait(1)
