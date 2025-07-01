# Manim code generated with OpenAI GPT
# Command to generate animation: manim GenScene.py GenScene --format=mp4 --media_dir . --custom_folders video_dir

from manim import *
from math import *

class GenScene(Scene):
    def construct(self):

        sq = Square(side_length=2, color=RED)
        self.play(Create(sq))