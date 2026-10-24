"""Shots 6-7 · 光 / 名: lights rise from the dark land into a tree of light (the opening root network, mirrored upward),
the tree collapses into a seed of light, the seed opens into a ginkgo leaf of light, the name appears."""
import math
from pathlib import Path
import numpy as np
import moderngl
from shot_roots import RIBBON_VS, RIBBON_FS, GLOW_VS, GLOW_FS, look_at, perspective, birth_time, ss
from camera import Path as CamPath

ROOT = Path(__file__).parent


class Times:
    """film times for this sequence (set from the edl so they follow the chosen score)"""
    start = 101.333      # climax hit
    grow_end = 117.0     # tree fully grown
    collapse = 122.667   # tree collapses into the seed
    leaf = 125.3         # the seed opens into the leaf
    title = 127.3        # final swell: name, letterbox opens
    end = 136.0


TRUNK = 2.2


def ginkgo_points(n, rng):
    """points inside a ginkgo leaf outline (leaf space: petiole base at the origin, blade up, ~1 tall)"""
    pts = []
    while len(pts) < n:
        q = rng.uniform([-0.62, -0.02], [0.62, 1.05], (n, 2))
        b = q - np.array([0.0, 0.42])
        r = np.hypot(b[:, 0], b[:, 1]); a = np.arctan2(b[:, 0], b[:, 1])
        R = 0.56 * (1 - 0.20 * np.exp(-a * a / 0.003)) * (0.93 + 0.07 * np.cos(a * 3))
        blade = (np.abs(a) < 1.08) & (r < R) & (r > 0.03)
        pet = (np.abs(q[:, 0]) < 0.010) & (q[:, 1] > 0.24) & (q[:, 1] < 0.44)
        ok = blade | (pet & (rng.uniform(0, 1, n) < 0.15))
        pts.extend(q[ok].tolist())
    pts = np.array(pts[:n])
    # a denser rim so the outline reads
    return pts


PART_VS = """#version 410
uniform mat4 uVP; uniform vec2 uRes; uniform float uT, uFocus, uAperture, uP11, uGain;
uniform float uGrowEnd, uCollapse, uLeafT, uTitle;
uniform vec3 uSeedPos; uniform vec3 uLeafO, uLeafX, uLeafY;
in vec2 corner;
in vec4 a0;   // start xyz, t0
in vec4 a1;   // node xyz, dur
in vec4 a2;   // leaf uv, rnd, size
out vec2 vQ; out float vI; out vec3 vC;
vec3 bez(vec3 a, vec3 b, vec3 c, float t){ return mix(mix(a, b, t), mix(b, c, t), t); }
float S(float a, float b, float t){ float x = clamp((t - a) / (b - a), 0., 1.); return x * x * (3. - 2. * x); }
void main(){
    float T = uT, t0 = a0.w, dur = a1.w, rnd = a2.z;
    if (T < t0) { gl_Position = vec4(3., 3., 3., 1.); return; }
    float u = clamp((T - t0) / dur, 0., 1.);
    float e = u * u * (3. - 2. * u);
    vec3 st = a0.xyz, nd = a1.xyz;
    vec3 mid = mix(st, nd, 0.5) + vec3(sin(rnd * 40.) * 2.0, 2.5 + 2.0 * rnd, cos(rnd * 37.) * 2.0);
    vec3 p = bez(st, mid, nd, e);
    // settled lights shimmer on the branches
    p += (1. - step(u, 0.999)) * 0.04 * vec3(sin(T * 1.3 + rnd * 50.), sin(T * 1.1 + rnd * 70.), cos(T * 1.2 + rnd * 30.));
    // collapse into the seed, spiralling
    float c = S(uCollapse + rnd * 0.4, uLeafT - 0.2 + rnd * 0.3, T);
    vec3 rel = p - uSeedPos;
    float ang = c * 3.0 * (0.5 + rnd);
    rel.xz = mat2(cos(ang), sin(ang), -sin(ang), cos(ang)) * rel.xz;
    p = uSeedPos + rel * (1. - c * c);
    // open into the leaf
    float l = S(uLeafT + rnd * 0.6, uTitle + rnd * 0.5, T);
    vec3 lp = uLeafO + uLeafX * a2.x + uLeafY * a2.y;
    p = mix(p, lp, l);
    vec4 cc = uVP * vec4(p, 1.);
    if (cc.w < 0.05) { gl_Position = vec4(3., 3., 3., 1.); return; }
    float w = cc.w;
    float px = a2.w * uRes.y * 0.5 * uP11 / w;
    float coc = min(uAperture * abs(w - uFocus) / w * uRes.y * 0.5 * uP11, uRes.y * 0.02);
    float wpx = max(px + coc, 0.8);
    vI = (px + 0.35) * (px + 0.35) / ((wpx + 0.35) * (wpx + 0.35));
    // rising lights are brighter, settled ones twinkle, the leaf glows evenly
    float rising = 1. - step(0.999, u);
    float tw = 0.6 + 0.4 * sin(T * (2. + 3. * rnd) + rnd * 60.);
    float bright = mix(mix(0.45 * tw, 1.2, rising), 0.55, l) * (1. + 2.0 * c * (1. - l));
    vI *= bright;
    vC = mix(mix(vec3(1.0, 0.76, 0.42), vec3(0.62, 1.0, 0.55), step(0.70, rnd) * (1. - l)), vec3(1.0, 0.70, 0.28), l);
    vec2 s = cc.xy / w * uRes * 0.5 + corner * wpx;
    gl_Position = vec4(s / (uRes * 0.5) * w, cc.z, w);
    vQ = corner;
}
"""
PART_FS = """#version 410
uniform float uGain;
in vec2 vQ; in float vI; in vec3 vC;
out vec4 fragColor;
void main(){
    float r = length(vQ);
    float disc = smoothstep(1.0, 0.75, r) * (0.8 + 0.2 * smoothstep(0.5, 0.95, r));
    fragColor = vec4(vC * disc * vI * uGain, 1.);
}
"""


class LightShot:
    K = 6

    def __init__(self, eng, times=Times, cam_keys=None):
        self.eng = eng
        self.T = times
        self.cam_keys = cam_keys
        ctx = eng.ctx
        z = np.load(ROOT / 'assets' / 'roots.npz')
        P, parent, born, rr = z['P'], z['parent'], z['born'], z['rr']
        # mirror the root network upward into a crown on top of a trunk
        C = P.copy()
        C[:, 1] = -P[:, 1] * 1.0 + TRUNK
        C[:, 0] *= 1.1; C[:, 2] *= 1.1
        itmax = born.max()
        bt = self.T.start + 1.2 + (self.T.grow_end - self.T.start - 1.2) * (born / itmax) ** 0.75
        i = np.arange(1, len(C)); j = parent[i]
        pd = np.zeros(len(C))
        inst = np.concatenate([C[j], C[i], rr[j, None] * 1.3, rr[i, None] * 1.3, bt[j, None], bt[i, None], pd[j, None], pd[i, None]], 1).astype('f4')
        # trunk
        k = np.arange(30)
        tp = np.stack([0.03 * np.sin(k * 0.4), k / 29 * TRUNK, 0.03 * np.cos(k * 0.33)], 1)
        tb = self.T.start + 0.3 + 1.0 * k / 29
        tinst = np.concatenate([tp[:-1], tp[1:], np.full((29, 1), 0.11), np.full((29, 1), 0.10), tb[:-1, None], tb[1:, None], np.zeros((29, 2))], 1).astype('f4')
        inst = np.concatenate([tinst, inst], 0)
        self.nseg = len(inst)
        self.C, self.bt = C, bt
        self.cvbo = ctx.buffer(np.array([0, -1, 1, -1, 0, 1, 1, 1], 'f4').tobytes())
        self.ivbo = ctx.buffer(inst.tobytes())
        self.rprog = ctx.program(vertex_shader=RIBBON_VS, fragment_shader=RIBBON_FS)
        self.rvao = ctx.vertex_array(self.rprog, [(self.cvbo, '2f', 'corner'), (self.ivbo, '3f 3f 4f 2f/i', 'p0', 'p1', 'rb', 'pd')])
        # particles: rise from the land to a node of the crown, then seed, then leaf
        rng = np.random.default_rng(11)
        n = 22000
        ang = rng.uniform(0, 2 * np.pi, n); rad = 2.5 + 13.0 * np.sqrt(rng.uniform(0, 1, n))
        st = np.stack([rad * np.cos(ang), rng.uniform(0.0, 0.3, n), np.minimum(rad * np.sin(ang), 4.0) - 3.0], 1)
        nodes = rng.integers(len(C) // 20, len(C), n)
        nd = C[nodes]
        t0 = np.maximum(bt[nodes] - rng.uniform(2.5, 4.5, n), self.T.start - 0.2 + rng.uniform(0, 1.5, n))
        dur = np.maximum(bt[nodes] - t0, 1.2) + rng.uniform(0.0, 0.8, n)
        # a third keep streaming up through the whole climax to branches that already exist
        late = rng.uniform(0, 1, n) < 0.33
        tl = rng.uniform(self.T.start + 4.0, self.T.collapse - 1.5, n)
        ok = late & (bt[nodes] < tl)
        t0 = np.where(ok, tl, t0); dur = np.where(ok, rng.uniform(2.2, 3.8, n), dur)
        leaf = ginkgo_points(n, rng)
        size = rng.uniform(0.008, 0.020, n)
        pinst = np.concatenate([st, t0[:, None], nd, dur[:, None], leaf, rng.uniform(0, 1, (n, 1)), size[:, None]], 1).astype('f4')
        self.npart = n
        self.pvbo = ctx.buffer(pinst.tobytes())
        self.pcorner = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
        self.pprog = ctx.program(vertex_shader=PART_VS, fragment_shader=PART_FS)
        self.pvao = ctx.vertex_array(self.pprog, [(self.pcorner, '2f', 'corner'), (self.pvbo, '4f 4f 4f/i', 'a0', 'a1', 'a2')])
        self.gprog = ctx.program(vertex_shader=GLOW_VS, fragment_shader=GLOW_FS)
        self.gvao = ctx.vertex_array(self.gprog, [(self.pcorner, '2f', 'corner')])
        self.bg = eng.shot_program('light_bg.glsl')
        Tm = self.T
        self.cam = CamPath(self.cam_keys(Tm) if self.cam_keys else [   # film time
            (Tm.start,        (0.0, 1.2, 17.0),  (0.0, 3.6, 0.0), 44, 0.025),
            (Tm.start + 6.0,  (5.5, 2.6, 15.5),  (0.0, 4.8, 0.0), 44, 0.025),
            (Tm.grow_end,     (-6.5, 4.0, 16.0), (0.0, 5.3, 0.0), 44, 0.02),
            (Tm.collapse,     (-3.5, 5.0, 18.0), (0.0, 5.6, 0.0), 42, 0.02),
            (Tm.title,        (0.0, 6.3, 12.5),  (0.0, 6.4, 0.0), 40, 0.02),
            (Tm.end,          (0.0, 6.3, 11.8),  (0.0, 6.4, 0.0), 40, 0.02),
        ])
        self.seed = np.array([0.0, 6.6, 0.0])
        self.leafO = np.array([0.0, 5.05, 0.4])       # leaf petiole base
        self.leafX = np.array([3.0, 0.0, 0.0])
        self.leafY = np.array([0.0, 3.0, 0.0])

    def matrices(self, t, W, H, jitter=(0, 0)):
        v = self.cam(t)
        eye, tgt, fov, ap = v[0:3], v[3:6], v[6], v[7]
        V = look_at(eye, tgt)
        Pm = perspective(fov, W / H, 0.1, 400.0)
        Pm[0, 2] += 2 * jitter[0] / W
        Pm[1, 2] += 2 * jitter[1] / H
        return Pm @ V, Pm[1, 1], float(np.linalg.norm(tgt - eye)), ap, eye, tgt, fov

    def render(self, eng, lt, tgt_fb, jitter, tg):
        t = tg                                        # this shot works in film time
        Tm = self.T
        ctx = eng.ctx
        W, H = tgt_fb.w, tgt_fb.h
        VP, p11, focus, ap, eye, look, fov = self.matrices(t, W, H, jitter)
        grow = ss(Tm.start, Tm.grow_end, t)
        collapse = ss(Tm.collapse, Tm.leaf, t)
        seedI = math.exp(-((t - Tm.leaf) / 0.6) ** 2) * 6.0 + collapse * (1 - ss(Tm.leaf, Tm.title, t)) * 2.0
        eng.run(self.bg, tgt_fb, dict(iTime=float(t), iResolution=(float(W), float(H)), iJitter=jitter, uEye=tuple(map(float, eye)),
                                      uTarget=tuple(map(float, look)), uFov=float(fov), uTree=float(grow * (1 - collapse)),
                                      uLeafGlow=float(ss(Tm.leaf, Tm.title + 1.0, t))))
        tgt_fb.fbo.use()
        ctx.viewport = (0, 0, W, H)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        vp = VP.T.astype('f4').tobytes()
        r = self.rprog
        r['uVP'].write(vp); r['uRes'].value = (float(W), float(H))
        r['uT'].value = float(t); r['uFocus'].value = focus; r['uAperture'].value = float(ap); r['uP11'].value = float(p11)
        r['uPulse'].value = (1.6, 0.25, 1.3333, Tm.start + 2.0); r['uWidthMul'].value = 1.0
        r['uGain'].value = float(0.75 * (1 - collapse))
        if collapse < 0.999:
            self.rvao.render(moderngl.TRIANGLE_STRIP, instances=self.nseg)
        pr = self.pprog
        pr['uVP'].write(vp); pr['uRes'].value = (float(W), float(H))
        pr['uT'].value = float(t); pr['uFocus'].value = focus; pr['uAperture'].value = float(ap); pr['uP11'].value = float(p11)
        pr['uGain'].value = 1.0
        pr['uCollapse'].value = Tm.collapse; pr['uLeafT'].value = Tm.leaf; pr['uTitle'].value = Tm.title
        pr['uSeedPos'].value = tuple(map(float, self.seed))
        pr['uLeafO'].value = tuple(map(float, self.leafO)); pr['uLeafX'].value = tuple(map(float, self.leafX)); pr['uLeafY'].value = tuple(map(float, self.leafY))
        self.pvao.render(moderngl.TRIANGLE_STRIP, instances=self.npart)
        if seedI > 0.01:
            g = self.gprog
            g['uVP'].write(vp); g['uRes'].value = (float(W), float(H))
            g['uPos'].value = tuple(map(float, self.seed)); g['uPx'].value = float(H * 0.06)
            g['uCol'].value = (1.0 * seedI, 0.82 * seedI, 0.5 * seedI)
            self.gvao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
