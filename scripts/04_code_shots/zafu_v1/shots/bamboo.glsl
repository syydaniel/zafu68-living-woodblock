// 竹 · a shoot breaks the soil and the camera rises with it through a moso bamboo forest in morning light
uniform vec3 uEye, uTarget; uniform float uFov, uAperture, uFocus;
uniform float uHeroH;      // hero culm height (m)
uniform float uBurst;      // seconds since the shoot broke the soil
uniform float uLeaves;     // 0..1 canopy fly-through overlay
uniform float uSeed;

const float CELL = 1.3;
const vec3 SUN = normalize(vec3(-0.50, 0.40, -0.77));
const vec3 SUNC = vec3(1.0, 0.80, 0.56) * 6.0;
const vec3 SKYC = vec3(0.42, 0.58, 0.70);
const float CAN0 = 12.0, CAN1 = 18.5;     // canopy slab

// ---------------------------------------------------------------- culms
struct Culm { vec2 c; float r; vec2 lean; float phase; float nodeL; float age; };
Culm culm(vec2 id){
    vec3 h = hash33(vec3(id, 7.1)), h2 = hash33(vec3(id, 13.7));
    Culm k;
    k.c = (id + 0.28 + 0.44 * h.xy) * CELL;
    k.r = mix(0.042, 0.082, h.z * h.z);
    k.lean = (h2.xy - 0.5) * 0.020;
    k.phase = h2.z;
    k.nodeL = mix(0.30, 0.44, hash12(id * 3.1));
    k.age = hash12(id * 5.7 + 2.);
    return k;
}
bool present(vec2 id){
    vec2 c = (id + 0.5) * CELL;
    return hash12(id + 91.7) < 0.80 && length(c) > 1.25;
}
float nodeDist(float y, float phase, float L){ float f = fract(y / L + phase); return min(f, 1. - f) * L; }
float culmR(float y, Culm k){ return k.r * (1. - 0.32 * clamp(y / 17., 0., 1.)); }
float sdCulm(vec3 p, Culm k){
    vec2 q = p.xz - k.lean * p.y - k.c;
    float r = culmR(p.y, k);
    float dn = nodeDist(p.y, k.phase, k.nodeL);
    r += 0.0035 * exp(-dn * dn / 0.00008) + 0.0015 * exp(-dn / 0.03);
    return length(q) - r;
}
// hero culm at the origin: grows from a sheathed shoot
float heroR(float y){
    float H = uHeroH;
    float r = mix(0.070, 0.050, clamp(y / 16., 0., 1.));
    float tip = clamp((H - y) / 1.1, 0., 1.);
    r *= pow(tip, 0.55);
    // young shoot is fat at the base
    r *= 1. + 1.2 * exp(-uHeroH * 0.9) * exp(-y * 3.);
    return r;
}
float sdHero(vec3 p){
    if (p.y > uHeroH + 0.05) return length(p - vec3(0., uHeroH, 0.));
    float r = heroR(p.y);
    float dn = nodeDist(p.y, 0.3, 0.36);
    r += 0.003 * exp(-dn * dn / 0.00008) * step(p.y, uHeroH - 2.2);
    // overlapping sheaths near the tip
    float sh = step(uHeroH - 2.6, p.y) * 0.004 * smoothstep(0.0, 0.5, fract(p.y / 0.22 + 0.5 * atan(p.z, p.x) / PI));
    return length(p.xz) - r - sh;
}

// analytic entry into a (sheared) vertical cylinder, then a few sphere-tracing steps on the exact culm SDF
float hitCulm(vec3 ro, vec3 rd, Culm k, float tmax){
    vec2 o = ro.xz - k.lean * ro.y - k.c;
    vec2 d = rd.xz - k.lean * rd.y;
    float R = k.r + 0.006;
    float a = dot(d, d), b = dot(o, d), c = dot(o, o) - R * R;
    float disc = b * b - a * c;
    if (disc < 0. || a < 1e-9) return -1.;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / a, t1 = (-b + sq) / a;
    if (t1 < 0.) return -1.;
    float t = max(t0, 0.0);
    for (int i = 0; i < 10; i++){
        vec3 p = ro + rd * t;
        float d0 = sdCulm(p, k);
        if (d0 < 0.0002 + 0.0003 * t) return (p.y < 0. || t > tmax) ? -1. : t;
        t += d0;
        if (t > t1) return -1.;
    }
    return -1.;
}

// grid traversal: nearest culm along the ray (returns t, id)
float traceCulms(vec3 ro, vec3 rd, float tmax, out vec2 hid){
    vec2 cell = floor(ro.xz / CELL);
    vec2 st = sign(rd.xz); st = vec2(st.x == 0. ? 1. : st.x, st.y == 0. ? 1. : st.y);
    vec2 inv = 1. / max(abs(rd.xz), vec2(1e-6));
    vec2 nb = (cell + max(st, 0.)) * CELL;
    vec2 tM = abs(nb - ro.xz) * inv;
    vec2 tD = CELL * inv;
    float best = 1e9; hid = vec2(0.);
    for (int i = 0; i < 70; i++){
        float tc = min(tM.x, tM.y);
        // test this cell and the neighbour across the nearer edge (lean can push a culm over)
        for (int j = 0; j < 2; j++){
            vec2 id = cell;
            if (j == 1) id += (tM.x < tM.y) ? vec2(st.x, 0.) : vec2(0., st.y);
            if (!present(id)) continue;
            float th = hitCulm(ro, rd, culm(id), tmax);
            if (th > 0. && th < best) { best = th; hid = id; }
        }
        if (best < tc + 0.3 || tc > tmax) break;
        if (tM.x < tM.y) { tM.x += tD.x; cell.x += st.x; } else { tM.y += tD.y; cell.y += st.y; }
    }
    return best < tmax ? best : -1.;
}

// ---------------------------------------------------------------- canopy + light
float canopyD(vec3 p){
    float h = smoothstep(CAN0, CAN0 + 2.5, p.y) * (1. - smoothstep(CAN1 - 3.0, CAN1, p.y));
    if (h <= 0.) return 0.;
    // feathery clumps near the culm tops, broken by fine leaf texture, with open sky between
    float clump = fbmv3(p * vec3(0.30, 0.50, 0.30) + vec3(0., 0., uSeed));
    float c = smoothstep(0.50, 0.70, clump);
    float fine = vnoise3(p * vec3(7.0, 10.0, 7.0)) * 0.6 + vnoise3(p * vec3(16.0, 22.0, 16.0)) * 0.4;
    float f = smoothstep(0.50, 0.535, fine);
    return c * f * 5.0 * h;
}
float sunVis(vec3 p){
    if (p.y > CAN1) return 1.;
    float ty0 = max(0., (CAN0 - p.y) / SUN.y), ty1 = (CAN1 - p.y) / SUN.y;
    float tau = 0.;
    float dt = (ty1 - ty0) / 6.;
    for (int i = 0; i < 6; i++) tau += canopyD(p + SUN * (ty0 + (float(i) + 0.5) * dt)) * dt;
    return exp(-tau * 0.9);
}
float shadowCulms(vec3 p){
    vec2 id;
    float t = traceCulms(p + SUN * 0.01, SUN, 22., id);
    return t > 0. ? 0. : 1.;
}
float phaseHG(float c, float g){ float g2 = g * g; return (1. - g2) / (4. * PI * pow(1. + g2 - 2. * g * c, 1.5)); }

float fogD(vec3 p){
    float d = 0.012 * exp(-p.y / 9.) + 0.05 * exp(-p.y / 1.2) * (0.5 + fbm3(p.xz * 0.15 + iTime * 0.02));
    // dust of the burst around the hero
    float dust = exp(-uBurst * 1.3) * step(0., uBurst) * exp(-dot(p - vec3(0., 0.15, 0.), p - vec3(0., 0.15, 0.)) / (0.08 + uBurst * 0.25));
    return d + dust * 1.5;
}

// ---------------------------------------------------------------- shading
vec3 culmColor(vec3 p, Culm k, bool hero){
    float dn = nodeDist(p.y, hero ? 0.3 : k.phase, hero ? 0.36 : k.nodeL);
    float y = p.y;
    float ang = hero ? atan(p.z, p.x) : atan(p.z - k.c.y, p.x - k.c.x);
    vec3 young = vec3(0.22, 0.42, 0.14), old = vec3(0.42, 0.44, 0.20);
    vec3 base = mix(young, old, hero ? 0.0 : smoothstep(0.35, 1.0, k.age));
    // vertical streaks
    base *= 0.85 + 0.15 * gnoise(vec2(ang * 3., y * 0.6 + k.phase * 9.));
    // white powder band just below each node, dark ring at the node
    float below = fract((y) / (hero ? 0.36 : k.nodeL) + (hero ? 0.3 : k.phase));
    float powder = smoothstep(0.80, 0.95, below) * (1. - smoothstep(0.97, 1.0, below));
    base = mix(base, vec3(0.62, 0.66, 0.58), powder * 0.55 * exp(-length(p - uEye) * 0.12));
    base *= 1. - 0.45 * exp(-dn * dn / 0.00004);
    // grey-white lichen patches on old culms
    if (!hero) {
        float li = smoothstep(0.55, 0.75, fbm3(vec2(ang * 1.5, y * 1.8) + k.phase * 20.)) * k.age;
        base = mix(base, vec3(0.55, 0.56, 0.50), li * 0.6);
    } else {
        // brown spotted sheaths near the shoot tip
        float sh = smoothstep(uHeroH - 2.8, uHeroH - 2.2, y);
        vec3 sheath = vec3(0.34, 0.22, 0.12) * (0.7 + 0.3 * fbm3(vec2(ang * 4., y * 9.)));
        float spots = smoothstep(0.62, 0.74, vnoise(vec2(ang * 22., y * 48.)));
        float edges = smoothstep(0.85, 1.0, fract(y / 0.21 + 0.3 * sin(ang)));
        sheath *= (1. - 0.45 * spots) * (1. - 0.35 * edges);
        base = mix(base, sheath, sh);
    }
    return base;
}

vec3 shadeSurface(vec3 p, vec3 n, vec3 rd, vec3 alb, float rough, float spec){
    float sv = sunVis(p) * shadowCulms(p);
    float ndl = max(dot(n, SUN), 0.);
    // sky light from above, blocked by the canopy
    float skyOcc = 0.25 + 0.75 * exp(-max(0., CAN0 + 2. - p.y) * 0.05);
    vec3 amb = SKYC * 0.55 * (0.6 + 0.4 * n.y) * skyOcc + vec3(0.26, 0.34, 0.14) * 0.45 * (0.6 - 0.4 * n.y);
    vec3 c = alb * (SUNC * ndl * sv + amb);
    vec3 h = normalize(SUN - rd);
    float nh = max(dot(n, h), 0.);
    float a2 = rough * rough * rough * rough;
    float D = a2 / (PI * pow(nh * nh * (a2 - 1.) + 1., 2.));
    float F = 0.04 + 0.96 * pow(1. - max(dot(-rd, n), 0.), 5.);
    float fade = exp(-length(p - uEye) * 0.06);
    c += SUNC * D * F * ndl * sv * spec * 0.25 * fade;
    c += SKYC * min(F, 0.35) * 0.30 * spec * skyOcc * (0.25 + 0.75 * fade);      // glossy sky sheen along the culm edges
    // backlit rim where the sun grazes the silhouette
    float rim = pow(1. - max(dot(-rd, n), 0.), 3.) * pow(max(dot(rd, SUN), 0.), 2.);
    c += SUNC * rim * sv * 0.22 * spec * (0.15 + 0.85 * fade);
    return c;
}

// ---------------------------------------------------------------- burst debris
vec4 debris(vec3 ro, vec3 rd){
    // returns (t, albedo shade) of nearest debris particle
    float best = 1e9; float shade = 0.;
    if (uBurst < 0. || uBurst > 2.2) return vec4(-1.);
    for (int i = 0; i < 36; i++){
        float fi = float(i);
        vec3 h = hash33(vec3(fi, 3.3, 9.1));
        vec3 v = vec3(cos(h.x * TAU), 0., sin(h.x * TAU)) * (0.3 + 1.0 * h.y) + vec3(0., 1.2 + 2.2 * h.z, 0.);
        float tt = uBurst;
        vec3 c = vec3(0., 0.12, 0.) + v * tt + vec3(0., -4.9, 0.) * tt * tt;
        if (c.y < 0.) continue;
        float r = 0.006 + 0.016 * hash11(fi * 7.1);
        vec3 oc = ro - c;
        float b = dot(oc, rd), cc = dot(oc, oc) - r * r;
        float disc = b * b - cc;
        if (disc > 0.) { float t = -b - sqrt(disc); if (t > 0. && t < best) { best = t; shade = h.z; } }
    }
    return best < 1e8 ? vec4(best, shade, 0., 0.) : vec4(-1.);
}

// ---------------------------------------------------------------- leaves overlay for the canopy fly-through
vec4 leafLayer(vec2 p, float depth, float t){
    // bamboo leaves: long lanceolate blades, moving down the frame as the camera rises
    vec4 acc = vec4(0.);
    float sc = mix(2.2, 7.0, depth);
    vec2 q = p * sc + vec2(depth * 17.3, -t * mix(9.0, 3.5, depth) * uLeaves);
    vec2 id = floor(q), f = fract(q) - 0.5;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        vec2 cid = id + vec2(i, j);
        vec3 h = hash33(vec3(cid, depth * 31.));
        if (h.z > 0.55 + 0.4 * (1. - uLeaves)) continue;
        vec2 o = f - vec2(i, j) - (h.xy - 0.5) * 0.7;
        o = rot(h.x * 3.0 - 1.5) * o;
        float L = 0.50 + 0.30 * h.y, Wd = 0.060;
        float u = o.x / L + 0.5;
        float uu = clamp(u, 0., 1.);
        float prof = Wd * pow(max(sin(PI * pow(uu, 0.55)), 0.), 1.25) * (1. - 0.25 * uu);
        float d = abs(o.y + 0.05 * sin(u * 3.)) - prof;
        float soft = mix(0.035, 0.008, depth);
        float m = (1. - smoothstep(-soft, soft, d)) * step(0., u) * step(u, 1.);
        // mostly dark silhouettes; a few leaves catch the sun and glow through
        float lit = smoothstep(0.78, 0.95, h.y);
        vec3 lc = mix(vec3(0.020, 0.040, 0.012), vec3(0.45, 0.66, 0.10) * 1.8, lit);
        lc *= 0.75 + 0.5 * smoothstep(0.0, -0.02, abs(o.y) - 0.004) * lit;                    // midrib
        lc += vec3(0.8, 0.9, 0.4) * 1.5 * lit * smoothstep(0.012, 0.0, abs(d));               // bright rim
        acc.rgb = mix(acc.rgb, lc, m * (1. - acc.a));
        acc.a = max(acc.a, m);
    }
    return acc;
}

void main(){
    vec2 p = screenP();
    vec3 ro = uEye;
    mat3 cam = lookAt(uEye, uTarget, 0.);
    float fl = 0.5 / tan(radians(uFov) * 0.5);
    vec3 rd = normalize(cam * vec3(p, fl));
    // thin-lens depth of field from the subframe jitter
    if (uAperture > 0.) {
        vec2 lj = (iJitter + 0.5);
        float ang = lj.x * TAU, rr = sqrt(fract(lj.y * 7.13 + lj.x * 3.1));
        vec2 lens = vec2(cos(ang), sin(ang)) * rr * uAperture;
        vec3 fp = ro + rd * uFocus / dot(rd, cam[2]);
        ro += cam * vec3(lens, 0.);
        rd = normalize(fp - ro);
    }
    float noiseJ = hash12(gl_FragCoord.xy + fract(iTime * 7.3) * 91. + iJitter * 37.);

    // ---------------- primary hits
    float tmax = 90.;
    float tg = rd.y < 0. ? -ro.y / rd.y : 1e9;
    vec2 hid;
    float tc = traceCulms(ro, rd, min(tmax, tg), hid);
    // hero
    float th = -1.;
    {
        float t = 0.;
        vec2 o = ro.xz; vec2 d = rd.xz;
        float R = 0.16;
        float a = dot(d, d), b = dot(o, d), c = dot(o, o) - R * R;
        float disc = b * b - a * c;
        if (disc > 0. && a > 1e-9) {
            float t0 = max(0., (-b - sqrt(disc)) / a), t1 = (-b + sqrt(disc)) / a;
            t = t0;
            for (int i = 0; i < 48; i++){
                if (t > t1) break;
                vec3 q = ro + rd * t;
                float d0 = sdHero(q);
                if (d0 < 0.0002 + 0.0002 * t) { if (q.y >= -0.02) th = t; break; }
                t += d0 * 0.8;
            }
        }
    }
    vec4 db = debris(ro, rd);

    float tHit = 1e9; int kind = 0;   // 0 sky/canopy, 1 ground, 2 culm, 3 hero, 4 debris
    if (tg < tHit && tg > 0.) { tHit = tg; kind = 1; }
    if (tc > 0. && tc < tHit) { tHit = tc; kind = 2; }
    if (th > 0. && th < tHit) { tHit = th; kind = 3; }
    if (db.x > 0. && db.x < tHit) { tHit = db.x; kind = 4; }

    vec3 col;
    vec3 pos = ro + rd * min(tHit, tmax);
    if (kind == 1) {
        // leaf litter
        vec2 g = pos.xz;
        float n1 = fbm(g * 1.3), n2 = fbm(g * 9.0 + 3.), n3 = vnoise(g * 40.);
        vec3 alb = mix(vec3(0.20, 0.13, 0.07), vec3(0.34, 0.25, 0.13), smoothstep(-0.3, 0.4, n1));
        alb = mix(alb, vec3(0.14, 0.10, 0.06), smoothstep(0.2, 0.6, n2) * 0.6);
        // scattered fallen bamboo leaves (elongated streaks)
        vec2 lq = rot(floor(n1 * 6.) * 0.9) * g * vec2(3., 22.);
        float lf = smoothstep(0.62, 0.8, vnoise(lq));
        alb = mix(alb, vec3(0.42, 0.33, 0.16), lf * 0.7);
        alb *= 0.85 + 0.3 * n3;
        // the soil cracked open around the shoot
        float rg = length(g);
        float crack = exp(-rg * 6.) * smoothstep(0., 0.4, uBurst + 0.2);
        alb = mix(alb, vec3(0.10, 0.07, 0.04), crack * 0.8);
        vec3 n = normalize(vec3(-0.3 * (fbm3(g * 9. + vec2(0.01, 0.)) - fbm3(g * 9.)) / 0.01, 1., 0.) + vec3(0., 0., 0.));
        n = normalize(vec3(n2 * 0.25, 1., n1 * 0.25));
        col = shadeSurface(pos, n, rd, alb, 0.8, 0.2);
    } else if (kind == 2) {
        Culm k = culm(hid);
        vec2 q = pos.xz - k.lean * pos.y - k.c;
        vec3 n = normalize(vec3(q.x, 0., q.y));
        // node ridges tilt the normal
        float y = pos.y; float L = k.nodeL;
        float f = fract(y / L + k.phase); float dn = min(f, 1. - f) * L; float sgn = f < 0.5 ? 1. : -1.;
        n = normalize(n + vec3(0., sgn * 0.35 * exp(-dn * dn / 0.00008), 0.));
        col = shadeSurface(pos, n, rd, culmColor(pos, k, false), 0.32, 1.0);
    } else if (kind == 3) {
        vec2 q = pos.xz;
        vec3 n = normalize(vec3(q.x, 0.25 * smoothstep(uHeroH - 1.3, uHeroH, pos.y), q.y));
        Culm k; k.c = vec2(0.); k.r = 0.07; k.lean = vec2(0.); k.phase = 0.3; k.nodeL = 0.36; k.age = 0.;
        bool sheathed = pos.y > uHeroH - 2.4;
        col = shadeSurface(pos, n, rd, culmColor(pos, k, true), sheathed ? 0.7 : 0.28, sheathed ? 0.3 : 1.2);
        // fine hairs on the sheath catch the backlight
        col += SUNC * 0.03 * step(uHeroH - 2.4, pos.y) * pow(max(dot(rd, SUN), 0.), 3.) * sunVis(pos);
    } else if (kind == 4) {
        vec3 alb = mix(vec3(0.08, 0.05, 0.03), vec3(0.25, 0.18, 0.1), db.y);
        col = alb * (SUNC * 0.3 * sunVis(pos) + SKYC * 0.2);
    } else {
        // sky, seen through the canopy
        col = sunSky(rd, SUN, vec3(0.30, 0.48, 0.70) * 1.3, vec3(0.95, 0.88, 0.72) * 1.6, SUNC * 0.25, 0.0004);
        tHit = 400.;
    }

    // ---------------- canopy volume between the camera and the hit (dense leaves, lit through)
    {
        float y0 = CAN0, y1 = CAN1 + 0.5;
        float ta = (y0 - ro.y) / rd.y, tb = (y1 - ro.y) / rd.y;
        if (rd.y < 0.) { float tmp = ta; ta = tb; tb = tmp; }
        if (abs(rd.y) < 1e-4) { ta = (ro.y > y0 && ro.y < y1) ? 0. : 1e9; tb = 60.; }
        ta = max(ta, 0.); tb = min(tb, min(tHit, 70.));
        if (tb > ta) {
            const int N = 40;
            float dt = (tb - ta) / float(N);
            vec3 accC = vec3(0.); float T = 1.;
            for (int i = 0; i < N; i++){
                float t = ta + (float(i) + noiseJ) * dt;
                vec3 q = ro + rd * t;
                float d = canopyD(q);
                if (d > 0.001) {
                    float sv = sunVis(q + SUN * 0.3);
                    float back = pow(max(dot(rd, SUN), 0.), 3.);
                    float up = clamp(rd.y, 0., 1.);
                    // reflected light on the underside + light transmitted through the thin leaves
                    float tint = vnoise3(q * 0.4);
                    vec3 trans = mix(vec3(0.26, 0.48, 0.07), vec3(0.50, 0.58, 0.10), tint);
                    vec3 leafC = vec3(0.05, 0.09, 0.03) * (SKYC * 0.8 + SUNC * sv * 0.12)
                               + trans * (SKYC * 0.32 * up + SUNC * sv * (0.02 + 0.30 * back));
                    float a = 1. - exp(-d * dt * 2.4);
                    accC += T * a * leafC;
                    T *= 1. - a;
                    if (T < 0.02) break;
                }
            }
            col = col * T + accC;
        }
    }

    // ---------------- fog with shafts of sun through the canopy
    {
        float tend = min(tHit, 60.);
        const int N = 36;
        vec3 scat = vec3(0.); float T = 1.;
        float cosT = dot(rd, SUN);
        float ph = phaseHG(cosT, 0.62) * 0.8 + phaseHG(cosT, -0.1) * 0.2;
        float prev = 0.;
        for (int i = 0; i < N; i++){
            float x = (float(i) + noiseJ) / float(N);
            float t = tend * x * x;
            float dt = t - prev; prev = t;
            vec3 q = ro + rd * t;
            if (q.y > CAN0 + 1.) break;
            float d = fogD(q);
            float sv = sunVis(q);
            // dust specks glinting in the shafts
            vec3 cq = q * 5.0;
            vec3 ci = floor(cq);
            float sp = hash13(ci);
            float speck = step(0.9985, sp) * smoothstep(0.35, 0.0, length(fract(cq) - 0.5 - (hash33(ci) - 0.5) * 0.5)) * 30.;
            vec3 Li = SUNC * sv * ph * 1.1 + SKYC * 0.05;
            scat += T * (d * Li + speck * sv * SUNC * 0.02) * dt;
            T *= exp(-d * dt);
        }
        col = col * T + scat;
    }

    // ---------------- canopy fly-through: leaves rushing past the lens
    if (uLeaves > 0.001) {
        for (int k = 0; k < 3; k++){
            float dep = float(k) / 2.;
            vec4 L = leafLayer(p, dep, iTime);
            float blurA = L.a * mix(0.95, 0.75, dep);
            col = mix(col, L.rgb * mix(1.0, 0.7, dep), blurA * smoothstep(0., 0.25, uLeaves));
        }
    }
    fragColor = vec4(col, 1.);
}
