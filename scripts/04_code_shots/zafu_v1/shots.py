"""Shot classes. A shot renders linear HDR into a target at its local time."""
from pathlib import Path
import numpy as np
import moderngl

ROOT = Path(__file__).parent


class FragShot:
    """one fullscreen fragment shader (shots/<file>)"""
    K = 4

    def __init__(self, eng, file, K=None, uniforms=None, textures=None):
        self.file = file
        self.K = K or self.K
        self.uniforms = uniforms or (lambda t: {})
        self.textures = textures or {}
        self.prog = eng.shot_program(file)

    def render(self, eng, t, tgt, jitter, tg):
        self.prog = eng.shot_program(self.file)   # hot reload
        u = dict(iTime=float(t), iGlobal=float(tg), iResolution=(float(tgt.w), float(tgt.h)), iJitter=jitter)
        u.update(self.uniforms(t))
        eng.run(self.prog, tgt, u, self.textures)
