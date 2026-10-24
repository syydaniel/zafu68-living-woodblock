"""ZAFU film renderer.
HDR shots (fragment shaders + rasterised layers) -> transitions -> jittered motion-blur accumulation
-> bloom / streak / halation / ACES grade / letterbox -> text layers -> 16-bit frames -> ffmpeg.

  still:  .venv/bin/python engine.py still 3.0 12.5 ...        [--w 1280] [--k 4]  -> frames/still_XXX.X.png
  sheet:  .venv/bin/python engine.py sheet 0 22 12             [--w 640]          -> frames/sheet_0-22.png (12 frames)
  video:  .venv/bin/python engine.py video 0 136               [--w 1920] [--k 6] [--out name]  -> renders/name.mov
"""
import argparse, math, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import moderngl
from PIL import Image

ROOT = Path(__file__).parent
FPS = 30
VS = """#version 410
in vec2 pos; out vec2 uv;
void main(){ uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0., 1.); }"""


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Engine:
    def __init__(self, W=1920, H=1080):
        self.W, self.H = W, H
        self.ctx = moderngl.create_standalone_context(require=410)
        self.ctx.enable_only(moderngl.NOTHING)
        self.vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
        self._progs, self._vaos, self._mtimes = {}, {}, {}
        self.common = (ROOT / 'shots' / 'common.glsl').read_text()
        src = (ROOT / 'post' / 'post.glsl').read_text()
        self.post = {}
        for part in src.split('//@@ ')[1:]:
            name, body = part.split('\n', 1)
            self.post[name.strip()] = self.program('post:' + name.strip(), '#version 410\n' + body)
        self.scene = self.target(W, H)
        self.A = self.target(W, H)
        self.B = self.target(W, H)
        self.acc = self.target(W, H, 'f4')
        self.final = self.target(W, H)
        self.levels = []
        w, h = W, H
        for i in range(7):
            w, h = max(1, w // 2), max(1, h // 2)
            self.levels.append(self.target(w, h))
        self.streak = self.target(W // 8, H // 8)
        self.streak2 = self.target(W // 8, H // 8)
        self.depth = self.ctx.depth_renderbuffer((W, H))
        self.tex_cache = {}

    # ---------------------------------------------------------------- GL helpers
    def target(self, w, h, dtype='f2'):
        tex = self.ctx.texture((w, h), 4, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return SimpleNamespace(tex=tex, fbo=self.ctx.framebuffer([tex]), w=w, h=h)

    def program(self, key, frag, vs=VS):
        p = self.ctx.program(vertex_shader=vs, fragment_shader=frag)
        self._progs[key] = p
        self._vaos[key] = self.ctx.vertex_array(p, [(self.vbo, '2f', 'pos')]) if 'pos' in p else None
        return p

    def shot_program(self, file):
        path = ROOT / 'shots' / file
        mt = path.stat().st_mtime
        key = 'shot:' + file
        if key in self._progs and self._mtimes.get(key) == mt:
            return self._progs[key]
        src = '#version 410\n' + self.common + '\n#line 1\n' + path.read_text()
        p = self.program(key, src)
        self._mtimes[key] = mt
        return p

    def vao(self, prog):
        for k, p in self._progs.items():
            if p is prog:
                return self._vaos[k]

    def run(self, prog, tgt, uniforms=None, textures=None, blend=None, clear=None):
        tgt.fbo.use()
        self.ctx.viewport = (0, 0, tgt.w, tgt.h)
        if clear is not None:
            tgt.fbo.clear(*clear)
        for i, (name, tex) in enumerate((textures or {}).items()):
            tex.use(i)
            if name in prog:
                prog[name].value = i
        for k, v in (uniforms or {}).items():
            if k in prog:
                prog[k].value = v
        if blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = blend
        else:
            self.ctx.disable(moderngl.BLEND)
        self.vao(prog).render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def texture_from_array(self, key, arr, mip=True):
        if key in self.tex_cache:
            return self.tex_cache[key]
        a = np.ascontiguousarray(arr[::-1] if arr.ndim == 3 else arr[::-1], dtype=np.float32)
        comps = a.shape[2] if a.ndim == 3 else 1
        tex = self.ctx.texture((a.shape[1], a.shape[0]), comps, a.astype(np.float16).tobytes(), dtype='f2')
        if mip:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        self.tex_cache[key] = tex
        return tex

    # ---------------------------------------------------------------- frame
    def render_scene(self, edl, t, jitter):
        act = edl.active(t)
        if len(act) == 1:
            s, lt = act[0]
            s.render(self, lt, self.scene, jitter, t)
        else:
            (sa, la), (sb, lb), mix, ttype, seed = act[0], act[1], act[2], act[3], act[4]
            sa.render(self, la, self.A, jitter, t)
            sb.render(self, lb, self.B, jitter, t)
            self.run(self.post['comp'], self.scene, dict(uMix=mix, uType=ttype, iResolution=(float(self.W), float(self.H)), uSeed=seed),
                     dict(uA=self.A.tex, uB=self.B.tex))

    def render_frame(self, edl, t, K=None):
        K = K or edl.samples(t)
        shutter = edl.shutter(t) / FPS
        self.acc.fbo.clear(0, 0, 0, 0)
        for k in range(K):
            tk = t + shutter * ((k + 0.5) / K - 0.5) if K > 1 else t
            jit = (halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5) if K > 1 else (0.0, 0.0)
            self.render_scene(edl, tk, jit)
            self.run(self.post['accum'], self.acc, dict(uWeight=1.0 / K), dict(uSrc=self.scene.tex), blend=(moderngl.ONE, moderngl.ONE))
        g = edl.grade(t)
        # bloom chain
        src = self.acc
        for i, lv in enumerate(self.levels):
            self.run(self.post['down'], lv, dict(uTexel=(1.0 / src.w, 1.0 / src.h), uFirst=1 if i == 0 else 0), dict(uSrc=src.tex))
            src = lv
        for i in range(len(self.levels) - 1, 0, -1):
            lo, hi = self.levels[i], self.levels[i - 1]
            self.run(self.post['up'], hi, dict(uTexel=(1.0 / lo.w, 1.0 / lo.h), uRadius=1.0), dict(uSrc=lo.tex), blend=(moderngl.ONE, moderngl.ONE))
        # anamorphic streak from level 2 (1/8)
        self.run(self.post['streak'], self.streak, dict(uTexel=(1.0 / self.levels[2].w, 1.0 / self.levels[2].h), uStep=2.0), dict(uSrc=self.levels[2].tex))
        self.run(self.post['streak'], self.streak2, dict(uTexel=(1.0 / self.streak.w, 1.0 / self.streak.h), uStep=5.0), dict(uSrc=self.streak.tex))
        u = dict(iResolution=(float(self.W), float(self.H)), uFrame=float(round(t * FPS)))
        u.update({k: v for k, v in g.items()})
        self.run(self.post['final'], self.final, u, dict(uScene=self.acc.tex, uBloom=self.levels[0].tex, uStreak=self.streak2.tex))
        # text
        for L in edl.texts_at(t):
            tex = self.texture_from_array(L['key'], L['arr'])
            self.run(self.post['text'], self.final, dict(
                iResolution=(float(self.W), float(self.H)), uCenter=L['center'], uSize=L['size'], uIn=L['pin'], uOut=L['pout'],
                uGlow=L.get('glow', 0.4), uBlur=L.get('blur', 4.0), uRise=L.get('rise', 0.15), uSeed=L.get('seed', 1.0),
                uAlpha=L.get('alpha', 1.0), uStyle=L['style'], uColor=L.get('color', (0.96, 0.95, 0.9)),
                uGlowColor=L.get('glow_color', (1.0, 0.85, 0.6)), uTime=t),
                dict(uText=tex), blend=(moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA))

    def read(self, bits=16):
        a = np.frombuffer(self.final.fbo.read(components=3, dtype='f2'), dtype=np.float16).reshape(self.H, self.W, 3)[::-1]
        a = np.clip(a.astype(np.float32), 0, 1)
        if bits == 8:
            return (a * 255 + 0.5).astype(np.uint8)
        return (a * 65535 + 0.5).astype('<u2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('args', nargs='*', type=float)
    ap.add_argument('--w', type=int, default=None)
    ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--h', type=int, default=None)
    ap.add_argument('--edl', default='edl')
    ap.add_argument('--notext', action='store_true')
    a = ap.parse_args()
    import importlib
    E = importlib.import_module(a.edl)
    if a.notext:
        E.Film.texts_at = lambda self, t: []
    if a.mode == 'still':
        W = a.w or 1280; H = a.h or W * 9 // 16
        eng = Engine(W, H); film = E.Film(eng)
        for t in a.args:
            t0 = time.time()
            eng.render_frame(film, t, a.k)
            Image.fromarray(eng.read(8)).save(ROOT / 'frames' / f'still{a.tag}_{t:06.2f}.png')
            print('still', t, f'{time.time() - t0:.2f}s', flush=True)
    elif a.mode == 'sheet':
        W = a.w or 640; H = a.h or W * 9 // 16
        t0, t1, n = a.args[0], a.args[1], int(a.args[2]) if len(a.args) > 2 else 12
        eng = Engine(W, H); film = E.Film(eng)
        cols = 4; rows = math.ceil(n / cols)
        sheet = Image.new('RGB', (cols * W + (cols + 1) * 6, rows * H + (rows + 1) * 6), (40, 40, 40))
        from PIL import ImageDraw
        for i in range(n):
            t = t0 + (t1 - t0) * i / max(1, n - 1)
            eng.render_frame(film, t, a.k or 2)
            im = Image.fromarray(eng.read(8))
            ImageDraw.Draw(im).text((8, 6), f'{t:.2f}', fill=(255, 80, 80))
            sheet.paste(im, (6 + (i % cols) * (W + 6), 6 + (i // cols) * (H + 6)))
        name = ROOT / 'frames' / f'sheet{a.tag}_{t0:g}-{t1:g}.png'
        sheet.save(name); print('sheet', name)
    elif a.mode == 'video':
        W = a.w or 1920; H = a.h or W * 9 // 16
        t0, t1 = a.args[0], a.args[1]
        eng = Engine(W, H); film = E.Film(eng)
        out = ROOT / 'renders' / f'{a.out or f"seg_{t0:g}-{t1:g}"}.mov'
        f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', f'{W}x{H}', '-r', str(FPS),
                               '-i', '-', '-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', '-vendor', 'apl0', str(out)],
                              stdin=subprocess.PIPE)
        ts = time.time()
        for f in range(f0, f1):
            eng.render_frame(film, f / FPS, a.k)
            ff.stdin.write(eng.read(16).tobytes())
            if (f - f0) % 30 == 0:
                el = time.time() - ts
                print(f'{f}/{f1}  {el:.0f}s  eta {el / max(1, f - f0 + 1) * (f1 - f):.0f}s', flush=True)
        ff.stdin.close(); ff.wait()
        print('done', out, f'{time.time() - ts:.0f}s')


if __name__ == '__main__':
    main()
