"""Shot 1 · 根: a seed of light falls into dark soil and a luminous root network grows (space colonisation, 3D ribbons, DOF motes)."""
import math
from pathlib import Path
import numpy as np
import moderngl
from scipy.spatial import cKDTree

ROOT = Path(__file__).parent
SEED_FALL, SEED_LAND = 1.5, 3.0
GROW_END = 19.0
SHOOT_T = 19.6          # the shoot starts rising
HIT = 21.3333           # cut to bamboo


def space_colonize(seed=7):
    rng = np.random.default_rng(seed)
    # attractors: dense near the surface, spreading wide, thinning with depth
    n = 7000
    y = -0.15 - rng.exponential(1.6, n)
    y = y[y > -7.0]
    n = len(y)
    R = 0.8 + 4.8 * (1 - np.exp((y + 0.1) / 1.2)) * np.exp((y + 2.5) / 9.0)
    ang = rng.uniform(0, 2 * np.pi, n)
    rr = R * np.sqrt(rng.uniform(0, 1, n))
    att = np.stack([rr * np.cos(ang), y, rr * np.sin(ang) * 0.75], 1)
    nodes = [np.array([0.0, 0.0, 0.0])]
    parent = [-1]
    it_born = [0]
    # a taproot first: straight down a little
    for k in range(6):
        nodes.append(nodes[-1] + np.array([0, -0.07, 0]) + rng.normal(0, 0.01, 3)); parent.append(len(nodes) - 2); it_born.append(k + 1)
    influence, kill, seg = 1.2, 0.18, 0.075
    alive = np.ones(len(att), bool)
    for it in range(7, 520):
        N = np.array(nodes)
        tree = cKDTree(N)
        A = att[alive]
        if len(A) < 5:
            break
        d, idx = tree.query(A, distance_upper_bound=influence)
        ok = np.isfinite(d)
        if not ok.any():
            break
        A, idx = A[ok], idx[ok]
        v = A - N[idx]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        acc = np.zeros_like(N)
        cnt = np.zeros(len(N))
        np.add.at(acc, idx, v)
        np.add.at(cnt, idx, 1)
        grow = np.nonzero(cnt > 0)[0]
        new = []
        for gi in grow:
            dv = acc[gi] / cnt[gi]
            dv += np.array([0, -0.35, 0]) + rng.normal(0, 0.18, 3)      # gravitropism + wander
            dv /= np.linalg.norm(dv) + 1e-9
            p = N[gi] + dv * seg
            new.append(p); parent.append(int(gi)); it_born.append(it)
        nodes.extend(new)
        tree2 = cKDTree(np.array(new))
        dd, _ = tree2.query(att, distance_upper_bound=kill)
        alive &= ~np.isfinite(dd)
    P = np.array(nodes)
    parent = np.array(parent)
    born = np.array(it_born, np.float64)
    # pipe model radii
    n = len(P)
    r = np.full(n, 0.0045)
    kids = [[] for _ in range(n)]
    for i in range(1, n):
        kids[parent[i]].append(i)
    order = np.argsort(-born, kind='stable')
    rr = np.zeros(n)
    for i in order:
        if kids[i]:
            rr[i] = (sum(rr[c] ** 2.3 for c in kids[i])) ** (1 / 2.3)
        else:
            rr[i] = 0.0045
    rr = np.minimum(rr, 0.075)
    # path distance from the seed
    pd = np.zeros(n)
    for i in np.argsort(born, kind='stable'):
        if parent[i] >= 0:
            pd[i] = pd[parent[i]] + np.linalg.norm(P[i] - P[parent[i]])
    return P, parent, born, rr, pd


def birth_time(it, itmax):
    x = it / itmax
    return SEED_LAND + 0.25 + (GROW_END - SEED_LAND - 0.25) * x ** 0.62


def look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy, aspect, near=0.05, far=200.0):
    f = 1 / math.tan(math.radians(fovy) / 2)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect; M[1, 1] = f
    M[2, 2] = (far + near) / (near - far); M[2, 3] = 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


def ss(a, b, t):
    x = min(1.0, max(0.0, (t - a) / (b - a)))
    return x * x * x * (x * (x * 6 - 15) + 10)


def camera(t):
    """eye, target, fov, focus distance, aperture: seed close-up -> dive among the roots -> pull back to the whole network -> whip up"""
    keys = [  # time, eye, target, fov, aperture
        (0.0,  (0.0, 0.25, 3.2),   (0.0, 0.05, 0.0),  36, 0.012),
        (3.0,  (0.0, 0.20, 3.0),   (0.0, 0.00, 0.0),  36, 0.012),
        (7.5,  (0.9, -0.55, 2.3),  (0.0, -0.75, 0.0), 40, 0.016),
        (13.0, (1.9, -2.3, 1.25),  (-0.3, -2.0, -0.4), 46, 0.020),
        (15.5, (1.4, -2.0, 4.8),   (0.0, -2.0, 0.0),  44, 0.012),
        (18.8, (0.6, -1.2, 10.5),  (0.0, -1.8, 0.0),  38, 0.006),
        (19.6, (0.5, -1.0, 10.8),  (0.0, -1.5, 0.0),  38, 0.006),
        (21.4, (0.3, 2.5, 7.5),    (0.0, 9.0, 0.0),   30, 0.004),
    ]
    for (t0, e0, g0, f0, a0), (t1, e1, g1, f1, a1) in zip(keys, keys[1:]):
        if t <= t1 or t1 == keys[-1][0]:
            x = ss(t0, t1, t)
            e = np.array(e0) + (np.array(e1) - np.array(e0)) * x
            g = np.array(g0) + (np.array(g1) - np.array(g0)) * x
            fov = f0 + (f1 - f0) * x; ap = a0 + (a1 - a0) * x
            break
    # a slow orbit on top of the keys while diving, and a gentle handheld float
    e = e + 0.02 * np.array([math.sin(t * 0.7), math.sin(t * 0.53 + 1), math.sin(t * 0.61 + 2)])
    focus = np.linalg.norm(e - g)
    return e, g, fov, focus, ap


RIBBON_VS = """#version 410
uniform mat4 uVP; uniform vec2 uRes; uniform float uT, uFocus, uAperture, uP11, uWidthMul;
in vec2 corner; in vec3 p0; in vec3 p1; in vec4 rb;   // r0 r1 b0 b1
in vec2 pd;    // path distance at both ends
out vec2 vQ; out float vL; out float vW; out float vI; out float vAge; out float vPd;
void main(){
    float b0 = rb.z, b1 = rb.w;
    float grow = clamp((uT - b0) / max(b1 - b0, 1e-3), 0., 1.);
    if (grow <= 0.) { gl_Position = vec4(3., 3., 3., 1.); return; }
    vec3 a = p0, b = mix(p0, p1, grow);
    vec4 ca = uVP * vec4(a, 1.), cb = uVP * vec4(b, 1.);
    if (ca.w < 0.15 || cb.w < 0.15) { gl_Position = vec4(3., 3., 3., 1.); return; }
    vec2 sa = ca.xy / ca.w * uRes * 0.5, sb = cb.xy / cb.w * uRes * 0.5;
    vec2 d = sb - sa; float L = length(d);
    vec2 dir = L > 1e-4 ? d / L : vec2(0., 1.);
    vec2 nrm = vec2(-dir.y, dir.x);
    float w = mix(ca.w, cb.w, 0.5);
    float age = clamp((uT - b1) / 5., 0., 1.);
    float r = mix(rb.x, rb.y, 0.5) * (0.45 + 0.55 * age) * uWidthMul;
    float px = r * uRes.y * 0.5 * uP11 / w;
    float coc = min(uAperture * abs(w - uFocus) / w * uRes.y * 0.5 * uP11, uRes.y * 0.018);
    float wpx = max(px + coc, 0.75);
    vI = (px + 0.4) / (wpx + 0.4) * min(1., (px + 0.3) / 0.9) * smoothstep(0.35, 1.3, w);
    // quad covering the capsule: along from -wpx to L + wpx, across +-wpx
    float s = mix(-wpx, L + wpx, corner.x);
    vec2 sp = sa + dir * s + nrm * wpx * corner.y;
    float wz = mix(ca.w, cb.w, clamp(s / max(L, 1e-4), 0., 1.));
    float z = mix(ca.z, cb.z, clamp(s / max(L, 1e-4), 0., 1.));
    gl_Position = vec4(sp / (uRes * 0.5) * wz, z, wz);
    vQ = vec2(s, wpx * corner.y); vL = L; vW = wpx;
    vAge = uT - mix(b0, b1, clamp(s / max(L, 1e-4), 0., 1.));
    vPd = mix(pd.x, pd.y, clamp(s / max(L, 1e-4), 0., 1.) * grow);
}
"""
RIBBON_FS = """#version 410
uniform float uT, uGain; uniform vec4 uPulse; // pulse speed, width, period, start
in vec2 vQ; in float vL; in float vW; in float vI; in float vAge; in float vPd;
out vec4 fragColor;
void main(){
    float dx = vQ.x - clamp(vQ.x, 0., vL);
    float dd = length(vec2(dx, vQ.y)) / vW;
    float prof = exp(-dd * dd * 3.2) * (1. - smoothstep(0.85, 1.0, dd));
    float young = exp(-max(vAge, 0.) / 1.6);
    vec3 gold = vec3(1.0, 0.62, 0.26), green = vec3(0.30, 0.62, 0.36), core = vec3(1.0, 0.92, 0.75);
    vec3 c = mix(green * 0.20, gold * 1.25, young) + core * 4. * exp(-max(vAge, 0.) / 0.22);
    // nutrient pulses travelling outward from the seed
    float ps = 0.;
    for (int k = 0; k < 8; k++){
        float tp = uPulse.w + float(k) * uPulse.z;
        float front = (uT - tp) * uPulse.x;
        if (front > 0.) ps += exp(-pow((vPd - front) / uPulse.y, 2.)) * exp(-front * 0.08);
    }
    c += vec3(1.0, 0.85, 0.55) * 3.0 * ps;
    fragColor = vec4(c * prof * vI * uGain, 1.);
}
"""
SPRITE_VS = """#version 410
uniform mat4 uVP; uniform vec2 uRes; uniform float uT, uFocus, uAperture, uP11, uSize;
uniform vec3 uSeed; uniform float uShock;
in vec2 corner; in vec4 pp;   // xyz, phase
out vec2 vQ; out float vI; out float vHot;
void main(){
    vec3 p = pp.xyz;
    float ph = pp.w;
    p += 0.05 * vec3(sin(uT * 0.21 + ph * 7.), sin(uT * 0.17 + ph * 11.) - 0.4 * uT * 0.02, sin(uT * 0.19 + ph * 5.));
    vec4 c = uVP * vec4(p, 1.);
    if (c.w < 0.05) { gl_Position = vec4(3., 3., 3., 1.); return; }
    float w = c.w;
    float px = uSize * (0.6 + 0.8 * fract(ph * 13.)) * uRes.y * 0.5 * uP11 / w;
    float coc = uAperture * abs(w - uFocus) / w * uRes.y * 0.5 * uP11;
    float wpx = max(px + coc, 0.8);
    vI = (px + 0.3) * (px + 0.3) / ((wpx + 0.3) * (wpx + 0.3));
    float dd = length(p - uSeed);
    vHot = exp(-pow((dd - uShock) / 0.25, 2.)) * step(0.01, uShock) + exp(-dd * 1.6) * 0.6;
    vec2 s = c.xy / w * uRes * 0.5 + corner * wpx;
    gl_Position = vec4(s / (uRes * 0.5) * w, c.z, w);
    vQ = corner;
}
"""
SPRITE_FS = """#version 410
uniform float uGain;
in vec2 vQ; in float vI; in float vHot;
out vec4 fragColor;
void main(){
    float r = length(vQ);
    float disc = smoothstep(1.0, 0.8, r) * (0.75 + 0.25 * smoothstep(0.5, 0.95, r));
    vec3 c = mix(vec3(0.55, 0.45, 0.32) * 0.35, vec3(1.0, 0.75, 0.4) * 3.0, clamp(vHot, 0., 1.));
    fragColor = vec4(c * disc * vI * uGain, 1.);
}
"""
GLOW_VS = """#version 410
uniform mat4 uVP; uniform vec2 uRes; uniform vec3 uPos; uniform float uPx;
in vec2 corner; out vec2 vQ;
void main(){
    vec4 c = uVP * vec4(uPos, 1.);
    vec2 s = c.xy / c.w * uRes * 0.5 + corner * uPx;
    gl_Position = vec4(s / (uRes * 0.5) * c.w, c.z, c.w);
    vQ = corner;
}
"""
GLOW_FS = """#version 410
uniform vec3 uCol; in vec2 vQ; out vec4 fragColor;
void main(){
    float r = length(vQ);
    float core = exp(-r * r * 60.) * 8. + exp(-r * 9.) * 0.8 + exp(-r * 3.) * 0.12;
    fragColor = vec4(uCol * core * smoothstep(1.0, 0.7, r), 1.);
}
"""


class RootsShot:
    K = 8

    def __init__(self, eng):
        self.eng = eng
        ctx = eng.ctx
        cache = ROOT / 'assets' / 'roots.npz'
        if cache.exists():
            z = np.load(cache); P, parent, born, rr, pd = z['P'], z['parent'], z['born'], z['rr'], z['pd']
        else:
            P, parent, born, rr, pd = space_colonize()
            np.savez(cache, P=P, parent=parent, born=born, rr=rr, pd=pd)
        self.P, self.parent, self.born, self.rr, self.pd = P, parent, born, rr, pd
        itmax = born.max()
        bt = np.array([birth_time(b, itmax) for b in born])
        self.bt = bt
        i = np.arange(1, len(P))
        j = parent[i]
        inst = np.concatenate([P[j], P[i], rr[j, None], rr[i, None], bt[j, None], bt[i, None], pd[j, None], pd[i, None]], 1).astype('f4')
        self.nseg = len(i)
        corners = np.array([0, -1, 1, -1, 0, 1, 1, 1], 'f4')
        self.cvbo = ctx.buffer(corners.tobytes())
        self.ivbo = ctx.buffer(inst.tobytes())
        self.rprog = ctx.program(vertex_shader=RIBBON_VS, fragment_shader=RIBBON_FS)
        self.rvao = ctx.vertex_array(self.rprog, [(self.cvbo, '2f', 'corner'), (self.ivbo, '3f 3f 4f 2f/i', 'p0', 'p1', 'rb', 'pd')])
        # shoot: a vertical wavy chain that rises from the seed
        k = np.arange(80)
        sp = np.stack([0.02 * np.sin(k * 0.3), k * 0.12, 0.02 * np.cos(k * 0.21)], 1)
        sb = SHOOT_T + 1.7 * (k / 79.0) ** 0.7
        sinst = np.concatenate([sp[:-1], sp[1:], np.full((79, 1), 0.035), np.full((79, 1), 0.03), sb[:-1, None], sb[1:, None],
                                np.zeros((79, 2))], 1).astype('f4')
        self.svbo = ctx.buffer(sinst.tobytes())
        self.svao = ctx.vertex_array(self.rprog, [(self.cvbo, '2f', 'corner'), (self.svbo, '3f 3f 4f 2f/i', 'p0', 'p1', 'rb', 'pd')])
        # motes
        rng = np.random.default_rng(3)
        m = 9000
        mp = np.stack([rng.uniform(-9, 9, m), rng.uniform(-8, 3.5, m), rng.uniform(-9, 9, m), rng.uniform(0, 1, m)], 1).astype('f4')
        self.mvbo = ctx.buffer(mp.tobytes())
        self.mcorner = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
        self.mprog = ctx.program(vertex_shader=SPRITE_VS, fragment_shader=SPRITE_FS)
        self.mvao = ctx.vertex_array(self.mprog, [(self.mcorner, '2f', 'corner'), (self.mvbo, '4f/i', 'pp')])
        self.gprog = ctx.program(vertex_shader=GLOW_VS, fragment_shader=GLOW_FS)
        self.gvao = ctx.vertex_array(self.gprog, [(self.mcorner, '2f', 'corner')])
        self.bg = eng.shot_program('roots_bg.glsl')
        # label anchors: nodes born just before each name reveal, spread over the frame
        self.labels = {}

    def seed_pos(self, t):
        if t < SEED_FALL:
            return None
        if t < SEED_LAND:
            x = (t - SEED_FALL) / (SEED_LAND - SEED_FALL)
            return np.array([0.0, 2.6 * (1 - x * x), 0.0])
        return np.array([0.0, 0.0, 0.0])

    def matrices(self, t, W, H, jitter=(0, 0)):
        eye, tgt, fov, focus, ap = camera(t)
        V = look_at(eye, tgt)
        Pm = perspective(fov, W / H)
        Pm[0, 2] += 2 * jitter[0] / W
        Pm[1, 2] += 2 * jitter[1] / H
        return Pm @ V, Pm[1, 1], focus, ap, eye

    def project(self, p, t, W=16, H=9):
        VP, *_ = self.matrices(t, W, H)
        c = VP @ np.append(p, 1.0)
        return c[:2] / c[3] * 0.5 * np.array([W / H, 1.0])   # screen units (y in [-.5,.5])

    def node_for(self, t_reveal, side):
        """a node born ~0.6 s before t_reveal, far enough from the centre on the requested side"""
        cand = np.nonzero((self.bt > t_reveal - 1.2) & (self.bt < t_reveal - 0.4))[0]
        best, bs = None, -1e9
        for c in cand[::3]:
            s = self.project(self.P[c], t_reveal)
            score = side * s[0] - abs(s[1] + 0.02) * 1.5 - (0 if abs(s[0]) < 0.62 else 10)
            if score > bs:
                best, bs = c, score
        return best

    def render(self, eng, t, tgt, jitter, tg):
        ctx = eng.ctx
        W, H = tgt.w, tgt.h
        VP, p11, focus, ap, eye = self.matrices(t, W, H, jitter)
        sp = self.seed_pos(t)
        ssc = (0.0, 0.0, -1.0)
        if sp is not None:
            c = VP @ np.append(sp, 1.0)
            ssc = (float(c[0] / c[3] * 0.5 * W / H), float(c[1] / c[3] * 0.5), 1.0)
        shock = max(0.0, t - SEED_LAND) * 3.2 if t > SEED_LAND else 0.0
        seedI = 0.0 if sp is None else (1.0 if t < SEED_LAND else 1.0 + 3.0 * math.exp(-(t - SEED_LAND) * 3.0))
        eng.run(self.bg, tgt, dict(iTime=float(t), iResolution=(float(W), float(H)), iJitter=jitter, uSeedScr=ssc, uSeedI=seedI,
                                   uShock=float(shock), uUp=float(ss(SHOOT_T, HIT, t))))
        tgt.fbo.use()
        ctx.viewport = (0, 0, W, H)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        vp = VP.T.astype('f4').tobytes()
        for prog in (self.rprog, self.mprog, self.gprog):
            prog['uVP'].write(vp)
            if 'uRes' in prog:
                prog['uRes'].value = (float(W), float(H))
        r = self.rprog
        r['uT'].value = float(t); r['uFocus'].value = float(focus); r['uAperture'].value = float(ap); r['uP11'].value = float(p11)
        r['uGain'].value = 0.8; r['uPulse'].value = (1.1, 0.10, 2.6667, 8.0); r['uWidthMul'].value = 0.75
        if t > SEED_LAND:
            self.rvao.render(moderngl.TRIANGLE_STRIP, instances=self.nseg)
        if t > SHOOT_T:
            r['uGain'].value = 1.6; r['uWidthMul'].value = 1.0; r['uPulse'].value = (0.0, 0.1, 100.0, 1e3)
            self.svao.render(moderngl.TRIANGLE_STRIP, instances=79)
        m = self.mprog
        m['uT'].value = float(t); m['uFocus'].value = float(focus); m['uAperture'].value = float(ap) * 1.3; m['uP11'].value = float(p11)
        m['uSize'].value = 0.006; m['uGain'].value = min(1.0, 0.25 + t / 3.0)
        m['uSeed'].value = tuple(float(x) for x in (sp if sp is not None else (0, 0, 0)))
        m['uShock'].value = float(shock)
        self.mvao.render(moderngl.TRIANGLE_STRIP, instances=9000)
        if sp is not None:
            g = self.gprog
            g['uPos'].value = tuple(float(x) for x in sp)
            g['uPx'].value = float(H * (0.045 + 0.05 * max(0.0, seedI - 1.0)))
            pulse = 1.0 + 0.25 * math.sin(t * 2 * math.pi / 2.6667) ** 8
            g['uCol'].value = tuple(float(x * seedI * pulse) for x in (1.0, 0.8, 0.5))
            self.gvao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
