// T1b: two gusts of maple leaves and samaras blow past the lens and cover the two season colour jumps.
// Transparent overlay, timeline frames 987..1030 at 25 fps (44 frames). Gust peaks at local frames ~11 and ~31.
const N = 44;
const GUSTS = [{ c: 11.2, n: 600 }, { c: 32.2, n: 640 }];
const FOV = 45;

function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

export async function create({ THREE, renderer, q }) {
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(FOV, 16 / 9, 0.05, 100);
  cam.position.set(0, 0, 0); cam.lookAt(0, 0, -1);
  const tanH = Math.tan(FOV / 2 * Math.PI / 180);
  const R = rng(+(q.get('seed') || 20261018));

  // colours sampled from T1b (gold, orange, red-orange, yellow, fresh green, deep green) and the samara straw
  const AUT = [[0.93, 0.66, 0.20], [0.91, 0.47, 0.15], [0.80, 0.29, 0.13], [0.95, 0.78, 0.36], [0.98, 0.58, 0.18]];
  const GRN = [[0.47, 0.62, 0.25], [0.27, 0.47, 0.30], [0.62, 0.70, 0.30]];
  const SAM = [[0.86, 0.72, 0.46], [0.78, 0.60, 0.36]];

  const rows = [];
  GUSTS.forEach((g, gi) => {
    for (let k = 0; k < g.n; k++) {
      const near = R() < 0.46;                                     // close to the lens: big, fast, blurred
      const z = near ? -(0.95 + R() * 1.5) : -(2.6 + R() * 6.5);
      const hh = -z * tanH, hw = hh * 16 / 9;
      const samara = !near && R() < 0.12;
      const size = samara ? 0.10 + R() * 0.08 : (near ? 0.30 + R() * 0.34 : 0.16 + R() * 0.16) * (0.8 + 0.4 * R());
      const dur = near ? 13 + R() * 7 : 22 + R() * 14;              // frames to cross the frame
      const t0 = g.c - dur * (0.35 + 0.3 * R()) + (R() - 0.5) * (near ? 9 : 12);
      const y0 = (R() * 2 - 1) * hh * 1.1, rise = (R() * 0.9 - 0.25) * hh;
      const p0 = [-(hw * 1.15 + size), y0 - rise * 0.5, z];
      const p1 = [hw * 1.15 + size, y0 + rise * 0.5 - (samara ? 0.4 * hh : 0), z - (near ? 0.3 : 1.5) * R()];
      const pal = samara ? SAM : (gi === 0 ? (R() < 0.82 ? AUT : GRN) : (R() < 0.55 ? AUT : GRN));
      const col = pal[Math.floor(R() * pal.length)].map(v => Math.min(1, v * (0.88 + 0.22 * R())));
      const ax = [R() * 2 - 1, R() * 2 - 1, R() * 2 - 1]; const al = Math.hypot(...ax) || 1;
      rows.push({ t0, dur, p0, p1, size, col, axis: ax.map(v => v / al), spin: (samara ? 0.05 : (near ? 0.04 + R() * 0.08 : 0.08 + R() * 0.16)) * (R() < 0.5 ? -1 : 1),
        roll: samara ? (0.55 + R() * 0.3) : (R() - 0.5) * (near ? 0.12 : 0.25), phase: R() * 6.283, flut: 0.03 + R() * 0.06, kind: samara ? 1 : 0, near: near ? 1 : 0 });
    }
  });

  const base = new THREE.PlaneGeometry(1, 1);
  const geo = new THREE.InstancedBufferGeometry();
  geo.index = base.index; geo.setAttribute('position', base.getAttribute('position')); geo.setAttribute('uv', base.getAttribute('uv'));
  geo.instanceCount = rows.length;
  const A = (n, f) => { const a = new Float32Array(rows.length * n); rows.forEach((r, i) => { const v = f(r); for (let j = 0; j < n; j++) a[i * n + j] = Array.isArray(v) ? v[j] : v; }); return new THREE.InstancedBufferAttribute(a, n); };
  geo.setAttribute('aTime', A(2, r => [r.t0, r.dur]));
  geo.setAttribute('aP0', A(3, r => r.p0));
  geo.setAttribute('aP1', A(3, r => r.p1));
  geo.setAttribute('aCol', A(3, r => r.col));
  geo.setAttribute('aAxis', A(4, r => [...r.axis, r.spin]));
  geo.setAttribute('aMisc', A(4, r => [r.size, r.roll, r.phase, r.flut]));
  geo.setAttribute('aKind', A(2, r => [r.kind, r.near]));

  const uni = { uT: { value: 0 } };
  const mat = new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms: uni, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    vertexShader: `
      in vec3 position; in vec2 uv; uniform mat4 projectionMatrix, viewMatrix; uniform float uT;
      in vec2 aTime; in vec3 aP0, aP1, aCol; in vec4 aAxis, aMisc; in vec2 aKind;
      out vec2 vUv; out vec3 vCol; out float vKind, vSoft, vFade, vFace;
      mat3 rot(vec3 a, float g){ float c = cos(g), s = sin(g), k = 1.0 - c;
        return mat3(c + a.x*a.x*k, a.y*a.x*k + a.z*s, a.z*a.x*k - a.y*s,
                    a.x*a.y*k - a.z*s, c + a.y*a.y*k, a.z*a.y*k + a.x*s,
                    a.x*a.z*k + a.y*s, a.y*a.z*k - a.x*s, c + a.z*a.z*k); }
      void main(){
        float tau = (uT - aTime.x) / aTime.y;
        if (tau < 0.0 || tau > 1.0 || uT <= 0.0 || uT >= 43.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
        float size = aMisc.x, roll = aMisc.y, ph = aMisc.z, fl = aMisc.w;
        vec3 p = mix(aP0, aP1, tau);
        p.y += sin(uT * 0.35 + ph) * fl * 3.0 * size; p.x += cos(uT * 0.27 + ph * 1.7) * fl * 2.0 * size;
        vec3 lp = vec3(position.xy * size, 0.0);
        mat3 R = rot(normalize(aAxis.xyz), ph + aAxis.w * uT) * rot(vec3(0.0, 0.0, 1.0), ph * 2.0 + roll * uT);
        vec3 wp = p + R * lp;
        vec3 nrm = R * vec3(0.0, 0.0, 1.0);
        vFace = abs(nrm.z);
        vUv = uv; vCol = aCol; vKind = aKind.x;
        // defocus: leaves right at the lens get soft edges, far ones stay crisp
        vSoft = aKind.y > 0.5 ? clamp((2.0 + p.z) * 0.04 + 0.028, 0.018, 0.07) : 0.010;
        vFade = smoothstep(0.0, 0.08, tau) * (1.0 - smoothstep(0.9, 1.0, tau)) * smoothstep(0.0, 4.0, uT) * (1.0 - smoothstep(39.0, 43.0, uT));
        gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
      }`,
    fragmentShader: `
      precision highp float; in vec2 vUv; in vec3 vCol; in float vKind, vSoft, vFade, vFace; out vec4 o;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float lobe(float a, float c, float w){ float d = a - c; return exp(-d * d / (w * w)); }
      void main(){
        vec2 p = vUv - vec2(0.5, 0.45);
        float r = length(p), a = atan(p.x, p.y);          // a = 0 points up the leaf
        float m, vein = 0.0;
        if (vKind < 0.5) {
          // maple leaf: five lobes, toothed edge, short stem
          float R = 0.20 + 0.19 * lobe(a, 0.0, 0.46) + 0.17 * (lobe(a, 1.15, 0.43) + lobe(a, -1.15, 0.43))
                  + 0.09 * (lobe(a, 2.2, 0.40) + lobe(a, -2.2, 0.40));
          R += (0.012 * sin(a * 34.0) + 0.006 * sin(a * 71.0)) * smoothstep(0.22, 0.3, R);
          m = 1.0 - smoothstep(R - vSoft, R + vSoft, r);
          float stem = (1.0 - smoothstep(0.008, 0.008 + vSoft, abs(p.x))) * step(p.y, 0.0) * step(-0.36, p.y);
          m = max(m, stem);
          float cs[5] = float[5](0.0, 1.12, -1.12, 2.15, -2.15);
          for (int k = 0; k < 5; k++) { float d = a - cs[k]; float dv = abs(sin(d)) * r;
            vein = max(vein, (1.0 - smoothstep(0.0, 0.007 + vSoft * 0.5, dv)) * step(0.0, cos(d)) * step(r, 0.34)); }
          vein *= clamp(1.0 - vSoft * 12.0, 0.1, 1.0);
        } else {
          // samara: a seed with one veined wing
          vec2 q = vUv - 0.5; q.y += 0.25;
          float seed = 1.0 - smoothstep(0.075 - vSoft, 0.075 + vSoft, length(q * vec2(1.0, 1.3)));
          vec2 w = q - vec2(0.03, 0.30); float wing = 1.0 - smoothstep(1.0 - vSoft * 8.0, 1.0 + vSoft * 8.0, length(w / vec2(0.15, 0.36)));
          m = max(seed, wing * step(-0.02, q.y));
          vein = (1.0 - smoothstep(0.0, 0.01, abs(fract(q.y * 14.0 + q.x * 4.0) - 0.5) * 0.08)) * wing * 0.5;
          if (seed > 0.5) vein = 0.6;
        }
        m *= vFade;
        if (m < 0.003) discard;
        // flat woodblock colour: lit side / shade side in two steps, darker veins, faint paper grain
        float lit = vFace > 0.45 ? 1.0 : 0.82;
        if (!gl_FrontFacing) lit *= 0.9;
        // a touch of the backlit warmth at the rim and a slightly deeper tone near the stem, like the leaves already in T1b
        float rim = vKind < 0.5 ? smoothstep(0.18, 0.36, r) : 0.0;
        vec3 c = vCol * lit * mix(0.86, 1.0, smoothstep(0.0, 0.3, r)) * (1.0 - 0.24 * vein);
        c = mix(c, c * vec3(1.08, 1.03, 0.92) + 0.04, rim * 0.5);
        c = mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, 0.9);
        c *= 0.97 + 0.05 * hash(floor(vUv * 220.0));
        o = vec4(c * m, m);
      }`,
  });
  const mesh = new THREE.Mesh(geo, mat); mesh.frustumCulled = false;
  scene.add(mesh);

  return {
    frames: N,
    render(t) { uni.uT.value = t; renderer.setClearColor(0x000000, 0); renderer.render(scene, cam); },
  };
}
