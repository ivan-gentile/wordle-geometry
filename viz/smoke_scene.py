"""Trivial scene to verify the Manim render path (text + MathTex + a shape)."""
from manim import *


class Smoke(Scene):
    def construct(self):
        title = Text("Wordle as Geometry", color=GREEN).scale(0.9)
        eq = MathTex(r"p \leftarrow (1-\eta)\,p + \eta\,T[g,f]").scale(0.9)
        eq.next_to(title, DOWN, buff=0.6)
        dot = Dot(color=YELLOW).next_to(eq, DOWN, buff=0.8)
        self.play(Write(title))
        self.play(Write(eq))
        self.play(Create(dot))
        self.wait(0.3)
