"""HERO SCENE - "A real game": the SupCon solver wins YACHT in 3 guesses.

A faithful walkthrough of REAL solver data (viz/worked_game.json):
 - 40 context words scattered at their true 2D-PCA coords (faint gray dots)
 - the secret YACHT as a larger green dot
 - a glowing yellow belief dot that is nudged from b0 -> b1 -> b2 by a frozen rule
 - three Wordle tile rows on the right with the exact codes/letters
 - a shrinking distance-to-answer meter: 7.33 -> 5.32 -> 1.60 -> ~0

Everything is REAL data; one uniform scale+shift is applied to all points so the
relative geometry is preserved.
"""
import json
import numpy as np
from manim import *

# ---- Wordle feedback colors ----
WORDLE_GRAY = "#787c7e"
WORDLE_YELLOW = "#c9b458"
WORDLE_GREEN = "#6aaa64"
TILE_COLORS = {0: WORDLE_GRAY, 1: WORDLE_YELLOW, 2: WORDLE_GREEN}

BELIEF_COLOR = "#ffd54f"
SECRET_GREEN = "#1b5e20"
SECRET_DOT = "#6aaa64"
BEAM_COLOR = "#ffd700"

DATA = json.load(open("/home/ivangentile/wordle-geometry/viz/worked_game.json"))

# ---- Uniform transform: data xy -> screen xy (left region only) ----
_xs = [c["xy"][0] for c in DATA["context"]] + [b[0] for b in DATA["belief_xy"]]
_ys = [c["xy"][1] for c in DATA["context"]] + [b[1] for b in DATA["belief_xy"]]
_SX = (2.6 - (-6.0)) / (max(_xs) - min(_xs))
_SY = (3.4 - (-3.4)) / (max(_ys) - min(_ys))
_SCALE = min(_SX, _SY)
_CX = (min(_xs) + max(_xs)) / 2.0
_CY = (min(_ys) + max(_ys)) / 2.0
_TCX = (-6.0 + 2.6) / 2.0
_TCY = 0.0


def P(x, y):
    return np.array([_TCX + (x - _CX) * _SCALE, _TCY + (y - _CY) * _SCALE, 0.0])


class WorkedGame(Scene):
    def construct(self):
        self.camera.background_color = "#11141a"

        # ============================================================
        # 0. Title + context scatter + secret
        # ============================================================
        title = Text("A real game: the solver guesses YACHT",
                     weight=BOLD, color=WHITE).scale(0.62).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.2)

        # context words as faint gray dots
        context_dots = VGroup()
        secret_dot = None
        secret_word_xy = None
        nearby_labels = VGroup()
        # Label only words sitting in OPEN space (the upper-left is a dense cluster
        # of the answer's rhyme-neighbours -- CATCH/HATCH/MATCH/... -- which we leave
        # as unlabelled dots and annotate once, below, to avoid label collisions).
        label_set = {"PROSE", "SHEAR", "KNOCK", "SMELT", "PATIO", "BLESS"}

        for c in DATA["context"]:
            x, y = c["xy"]
            pt = P(x, y)
            if c["is_secret"]:
                secret_word_xy = (x, y)
                continue  # drawn specially below
            dot = Dot(point=pt, radius=0.045, color="#5b626d", fill_opacity=0.7)
            context_dots.add(dot)
            if c["word"] in label_set:
                lab = Text(c["word"], color="#7c8492").scale(0.26)
                # place label on the side that keeps it inside the frame
                lab.next_to(dot, RIGHT if x < 0 else UP, buff=0.06)
                lab.set_opacity(0.85)
                nearby_labels.add(lab)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in context_dots],
                              lag_ratio=0.02), run_time=1.6)
        self.play(FadeIn(nearby_labels), run_time=0.8)

        # SECRET = YACHT : larger green dot + label
        secret_pt = P(*secret_word_xy)
        secret_dot = Dot(point=secret_pt, radius=0.14, color=SECRET_DOT)
        secret_glow = Circle(radius=0.30, color=SECRET_DOT, fill_opacity=0.22,
                             stroke_width=0).move_to(secret_pt)
        secret_label = Text("YACHT (answer)", color=SECRET_DOT, weight=BOLD).scale(0.34)
        # place to the RIGHT (open space) -- the upper-left is the dense rhyme cluster
        secret_label.next_to(secret_dot, RIGHT, buff=0.14)
        self.play(GrowFromCenter(secret_glow), GrowFromCenter(secret_dot),
                  FadeIn(secret_label))
        self.play(secret_glow.animate.scale(1.25).set_opacity(0.1),
                  rate_func=there_and_back, run_time=0.7)

        # ============================================================
        # Right strip: persistent header for tile rows + meter
        # ============================================================
        strip_x = 4.9  # center of right strip
        rows_title = Text("guesses", color="#c8d0dc").scale(0.32)
        rows_title.move_to(np.array([strip_x, 3.05, 0.0]))
        self.play(FadeIn(rows_title))

        # vertical anchors for the three tile rows
        row_y = {1: 2.35, 2: 1.45, 3: 0.55}

        def make_tile_row(word, codes, y):
            grp = VGroup()
            for ch, code in zip(word, codes):
                sq = Square(side_length=0.40, fill_color=TILE_COLORS[code],
                            fill_opacity=1.0, stroke_color="#222629", stroke_width=2)
                letter = Text(ch, color=WHITE, weight=BOLD).scale(0.30).move_to(sq)
                grp.add(VGroup(sq, letter))
            grp.arrange(RIGHT, buff=0.06)
            grp.move_to(np.array([strip_x + 0.25, y, 0.0]))
            tag = Text(f"#{list(row_y.keys())[list(row_y.values()).index(y)]}",
                       color="#7c8492").scale(0.28)
            tag.next_to(grp, LEFT, buff=0.18)
            return VGroup(tag, grp), grp

        def flip_in_row(row_group, tiles):
            # tag fades, tiles flip in left-to-right
            self.play(FadeIn(row_group[0]), run_time=0.25)
            self.play(LaggedStart(*[GrowFromCenter(t) for t in tiles],
                                  lag_ratio=0.18), run_time=1.1)

        # ============================================================
        # Distance-to-answer meter (right, below the rows)
        # ============================================================
        meter_top = -0.25
        meter_label = Text("distance to answer", color="#c8d0dc").scale(0.30)
        meter_label.move_to(np.array([strip_x, meter_top, 0.0]))

        BAR_X = strip_x - 1.35   # left edge of bar
        BAR_W = 2.7              # full bar width
        BAR_Y = meter_top - 0.55
        DMAX = 7.33              # bar full-scale

        bar_bg = RoundedRectangle(width=BAR_W, height=0.34, corner_radius=0.08,
                                  stroke_color="#3a4150", stroke_width=2,
                                  fill_color="#1a1e26", fill_opacity=1.0)
        bar_bg.move_to(np.array([BAR_X + BAR_W / 2, BAR_Y, 0.0]))

        def bar_fill_for(dist, color):
            frac = max(0.0, min(1.0, dist / DMAX))
            w = max(0.001, BAR_W * frac)
            r = RoundedRectangle(width=w, height=0.30, corner_radius=0.07,
                                 stroke_width=0, fill_color=color, fill_opacity=1.0)
            r.align_to(bar_bg, LEFT)
            r.shift(RIGHT * 0.02)
            r.set_y(BAR_Y)
            return r

        num_tracker = ValueTracker(7.33)
        num = always_redraw(lambda: Text(f"{num_tracker.get_value():.2f}",
                                         color=BELIEF_COLOR, weight=BOLD).scale(0.46)
                            .move_to(np.array([strip_x, BAR_Y - 0.6, 0.0])))

        self.play(FadeIn(meter_label), Create(bar_bg))
        bar = bar_fill_for(7.33, WORDLE_GRAY)
        self.add(num)
        self.play(GrowFromEdge(bar, LEFT), run_time=0.6)

        # ============================================================
        # 1. Belief dot appears at b0
        # ============================================================
        b0 = P(*DATA["belief_xy"][0])
        b1 = P(*DATA["belief_xy"][1])
        b2 = P(*DATA["belief_xy"][2])

        belief = Dot(point=b0, radius=0.13, color=BELIEF_COLOR)
        belief_glow = Circle(radius=0.28, color=BELIEF_COLOR, fill_opacity=0.25,
                             stroke_width=0).move_to(b0)
        belief_label = Text("belief", color=BELIEF_COLOR).scale(0.32)
        belief_label.next_to(belief, DOWN, buff=0.10)

        self.play(GrowFromCenter(belief_glow), GrowFromCenter(belief),
                  FadeIn(belief_label))

        caption = Text("The agent keeps a belief about where the answer is.",
                       color="#e8edf3").scale(0.40).to_edge(DOWN, buff=0.3)
        self.play(Write(caption))
        self.wait(0.5)

        # helper: a thin beam from belief to a target word dot
        def beam_to(target_pt, color=BEAM_COLOR):
            ln = Line(belief.get_center(), target_pt, color=color, stroke_width=2.5)
            ln.set_opacity(0.85)
            return ln

        # helper: move belief (dot+glow+label) to a new point
        def move_belief(new_pt, run_time=1.2):
            self.play(
                belief.animate.move_to(new_pt),
                belief_glow.animate.move_to(new_pt),
                belief_label.animate.next_to(new_pt, DOWN, buff=0.10),
                run_time=run_time,
            )

        # helper: update meter number + bar
        def update_meter(old_dist, new_dist, color, run_time=1.0):
            new_bar = bar_fill_for(new_dist, color)
            self.play(
                num_tracker.animate.set_value(new_dist),
                Transform(bar, new_bar),
                run_time=run_time,
            )

        # ============================================================
        # 2. GUESS 1 : SHARE  codes [0,1,1,0,0]   7.33 -> 5.32
        # ============================================================
        share_pt = P(*DATA["guess_xy"][0])
        share_label = Text("SHARE", color=BEAM_COLOR, weight=BOLD).scale(0.30)
        share_dot = Dot(point=share_pt, radius=0.08, color=BEAM_COLOR)
        share_label.next_to(share_dot, RIGHT, buff=0.10)

        new_cap = Text("Guess the nearest word; read the feedback.",
                       color="#e8edf3").scale(0.40).to_edge(DOWN, buff=0.3)
        self.play(Transform(caption, new_cap))

        beam1 = beam_to(share_pt)
        self.play(Create(beam1), GrowFromCenter(share_dot), FadeIn(share_label))

        row1, tiles1 = make_tile_row("SHARE", DATA["guesses"][0]["codes"], row_y[1])
        flip_in_row(row1, tiles1)
        self.wait(0.3)

        # arrow nudges belief b0 -> b1
        arrow1 = Arrow(start=b0, end=b1, color=BELIEF_COLOR, buff=0.12,
                       stroke_width=5, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(arrow1))
        move_belief(b1)
        self.play(FadeOut(arrow1), FadeOut(beam1),
                  share_label.animate.set_color("#7c8492").scale(0.9),
                  share_dot.animate.set_color("#5b626d"))
        update_meter(7.33, 5.32, WORDLE_YELLOW)
        self.wait(0.3)

        # ============================================================
        # 3. GUESS 2 : WATCH  codes [0,2,1,1,1]   5.32 -> 1.60
        # ============================================================
        watch_pt = P(*DATA["guess_xy"][1])
        watch_label = Text("WATCH", color=BEAM_COLOR, weight=BOLD).scale(0.30)
        watch_dot = Dot(point=watch_pt, radius=0.08, color=BEAM_COLOR)
        watch_label.next_to(watch_dot, RIGHT, buff=0.12)

        beam2 = beam_to(watch_pt)
        self.play(Create(beam2), GrowFromCenter(watch_dot), FadeIn(watch_label))

        row2, tiles2 = make_tile_row("WATCH", DATA["guesses"][1]["codes"], row_y[2])
        flip_in_row(row2, tiles2)
        self.wait(0.3)

        arrow2 = Arrow(start=b1, end=b2, color=BELIEF_COLOR, buff=0.12,
                       stroke_width=5, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(arrow2))
        move_belief(b2)
        self.play(FadeOut(arrow2), FadeOut(beam2),
                  watch_label.animate.set_color("#7c8492").scale(0.9),
                  watch_dot.animate.set_color("#5b626d"))
        update_meter(5.32, 1.60, WORDLE_YELLOW)
        self.wait(0.3)

        # ============================================================
        # 4. GUESS 3 / WIN : YACHT  all GREEN, belief snaps onto secret
        # ============================================================
        win_cap = Text("Now the nearest word IS the answer.",
                       color="#e8edf3").scale(0.40).to_edge(DOWN, buff=0.3)
        self.play(Transform(caption, win_cap))

        beam3 = beam_to(secret_pt, color=WORDLE_GREEN)
        self.play(Create(beam3))

        row3, tiles3 = make_tile_row("YACHT", DATA["guesses"][2]["codes"], row_y[3])
        flip_in_row(row3, tiles3)
        self.wait(0.2)

        # belief snaps onto the YACHT dot
        move_belief(secret_pt, run_time=0.8)
        update_meter(1.60, 0.0, WORDLE_GREEN, run_time=0.8)
        self.play(FadeOut(beam3))

        # celebrate
        self.play(
            Flash(secret_pt, color=WORDLE_GREEN, flash_radius=0.6, line_length=0.35,
                  num_lines=16),
            Indicate(secret_dot, color=WORDLE_GREEN, scale_factor=1.6),
            Indicate(VGroup(*tiles3), color=WORDLE_GREEN, scale_factor=1.15),
        )
        self.play(secret_glow.animate.scale(1.6).set_opacity(0.28),
                  rate_func=there_and_back, run_time=0.7)

        final_cap = Text("Solved in 3 - no search, just three nudges.",
                         weight=BOLD, color=WORDLE_GREEN).scale(0.46)
        final_cap.to_edge(DOWN, buff=0.3)
        self.play(Transform(caption, final_cap))

        # ============================================================
        # 5. hold
        # ============================================================
        self.wait(1.5)
