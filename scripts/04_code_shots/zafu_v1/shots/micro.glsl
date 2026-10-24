// 格物 · into the leaf, confocal-fluorescence style: glowing cell walls (cyan), chloroplasts (chlorophyll red),
// CO2 drifting in and fixed in gold; then a fast pull back for the data beats
uniform vec3 uEye, uTarget; uniform float uFov;
uniform float uCO2;        // 0..1 intensity of the CO2 stream
uniform float uFix;        // 0..1 how much of the CO2 is being fixed (gold sparks)
uniform float uPull;       // 0..1 pull-back phase (rising streaks)
uniform float uDepthFade;  // depth falloff of the tissue glow
uniform float uSection;    // focal distance of the optical section (0 = off)

const vec3 WALL = vec3(0.18, 0.78, 1.00);
const vec3 CHL  = vec3(1.00, 0.16, 0.34);
const vec3 GOLD = vec3(1.00, 0.72, 0.30);
const vec3 HERO = vec3(2.55, -6.95, 3.55);        // the chloroplast we visit

// 3D cellular: (F1, F2) and the id of the nearest feature
vec2 cells(vec3 p, out vec3 cid, float elong){
    p.y /= elong;
    vec3 i = floor(p), f = fract(p);
    float d1 = 8., d2 = 8.;
    for (int z = -1; z <= 1; z++) for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
        vec3 g = vec3(x, y, z);
        vec3 o = 0.15 + 0.7 * hash33(i + g);
        vec3 r = g + o - f;
        r.y *= elong;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; cid = i + g; } else if (d < d2) d2 = d;
    }
    return vec2(sqrt(d1), sqrt(d2));
}
// 2D jigsaw pavement cells on the leaf surface
vec3 pavement(vec2 p){
    vec2 w = p + 0.18 * vec2(gnoise(p * 2.3), gnoise(p * 2.3 + 7.));
    vec2 i = floor(w), f = fract(w);
    float d1 = 8., d2 = 8.; vec2 id = vec2(0.);
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){
        vec2 g = vec2(x, y);
        vec2 o = 0.2 + 0.6 * hash22(i + g);
        float d = length(g + o - f);
        if (d < d1) { d2 = d1; d1 = d; id = i + g; } else if (d < d2) d2 = d;
    }
    return vec3(d2 - d1, id);
}

vec3 tissue(vec3 p, out float absorb){
    vec3 em = vec3(0.);
    absorb = 0.;
    if (p.y > 0.05) return em;
    if (p.y > -0.9) {
        vec3 pv = pavement(p.xz * 0.9);
        float wall = exp(-pv.x / 0.018) * exp(-pow((p.y + 0.35) / 0.45, 2.));
        float top = exp(-abs(p.y + 0.02) / 0.03);
        em += WALL * (wall * 2.8 + top * 0.04 * (0.5 + 0.5 * vnoise(p.xz * 8.)));
        vec2 sc = (pv.yz + 0.5) / 0.9;
        vec3 h = hash33(vec3(pv.yz, 3.));
        if (h.x < 0.22) {
            vec2 q = rot(h.y * 6.28) * (p.xz - sc);
            vec2 e = q / vec2(0.26, 0.17);
            float r = length(e);
            float ring = exp(-pow((r - 0.62) / 0.16, 2.)) * exp(-pow((p.y + 0.12) / 0.2, 2.));
            float slit = exp(-pow(q.y / 0.025, 2.)) * step(abs(q.x), 0.16);
            em += CHL * ring * 2.6 * (1. - slit) + WALL * 1.2 * exp(-pow((r - 1.0) / 0.08, 2.)) * exp(-pow((p.y + 0.12) / 0.2, 2.));
        }
        absorb = 0.5;
        return em;
    }
    vec3 cid;
    vec2 F = cells(p * vec3(1.1, 1., 1.1), cid, 2.6);
    float bd = (F.y - F.x) * 0.5;
    float wall = exp(-bd / 0.012);
    em += WALL * wall * 1.8;
    vec3 q = p * vec3(8.0, 4.5, 8.0) + vec3(0., iTime * 0.12 + hash13(cid) * 9., 0.);
    float blob = smoothstep(0.74, 0.83, vnoise3(q)) * smoothstep(0.12, 0.03, bd);
    em += CHL * blob * 1.6;
        absorb = 0.10 + 0.6 * blob;
    return em;
}

vec4 heroChl(vec3 p){
    vec3 q = p - HERO;
    q.xz = rot(0.6) * q.xz;
    vec3 e = q / vec3(0.42, 0.20, 0.24);
    float r = length(e);
    if (r > 1.25) return vec4(0.);
    vec3 em = vec3(0.);
    float shell = exp(-pow((r - 1.0) / 0.045, 2.));
    em += CHL * shell * 1.0 + vec3(1.0, 0.5, 0.6) * shell * 0.25;
    if (r < 1.0) {
        vec2 gp = q.xz / vec2(0.11, 0.10);
        vec2 gi = floor(gp), gf = fract(gp) - 0.5;
        vec2 jit = (hash22(gi) - 0.5) * 0.3;
        float rr = length(gf - jit) * 0.105;
        float disc = smoothstep(0.042, 0.034, rr);
        float layers = pow(0.5 + 0.5 * cos(q.y * 360.), 3.) * step(abs(q.y), 0.085 + 0.03 * hash12(gi));
        float stroma = exp(-pow((mod(q.y + q.x * 0.3, 0.05) - 0.025) / 0.004, 2.)) * 0.25;
        em += vec3(0.35, 1.0, 0.55) * (disc * layers * 3.0 + stroma * 1.5 * (1. - disc));
        em += CHL * 0.06;
    }
    return vec4(em, 0.1 * step(r, 1.0));
}

vec3 co2(vec2 p, float t){
    vec3 acc = vec3(0.);
    for (int k = 0; k < 4; k++){
        float fk = float(k);
        float sc = 6. + fk * 4.;
        float z = fract(t * 0.18 + fk * 0.25);
        float zoom = exp2(z * 2.);
        vec2 q = p * sc / zoom + fk * 13.1;
        vec2 id = floor(q), f = fract(q) - 0.5;
        vec3 h = hash33(vec3(id, fk));
        if (h.z > 0.16) continue;
        vec2 c = (h.xy - 0.5) * 0.6;
        vec2 dir = normalize(vec2(cos(h.x * 9.), sin(h.x * 9.)));
        float fade = smoothstep(0., 0.3, z) * smoothstep(1., 0.7, z);
        float sz = 0.045 / zoom * 3.;
        float m = exp(-dot(f - c, f - c) / (sz * sz * 0.6));
        m += 0.6 * exp(-dot(f - c - dir * sz * 2.2, f - c - dir * sz * 2.2) / (sz * sz * 0.35));
        m += 0.6 * exp(-dot(f - c + dir * sz * 2.2, f - c + dir * sz * 2.2) / (sz * sz * 0.35));
        vec3 cc = mix(vec3(0.85, 0.92, 1.0), GOLD * 1.8, uFix * step(h.y, 0.5 + 0.5 * uFix));
        acc += cc * m * fade * 0.6;
    }
    return acc;
}

void main(){
    vec2 p = screenP();
    vec3 ro = uEye;
    mat3 cam = lookAt(uEye, uTarget, 0.);
    float fl = 0.5 / tan(radians(uFov) * 0.5);
    vec3 rd = normalize(cam * vec3(p, fl));
    float nj = hash12(gl_FragCoord.xy + iJitter * 71. + fract(iTime) * 13.);

    vec3 col = vec3(0.002, 0.004, 0.008);
    float T = 1.;
    float ta = 0., tb = 30.;
    if (rd.y != 0.) {
        float t0 = (0.1 - ro.y) / rd.y, t1 = (-14. - ro.y) / rd.y;
        ta = max(0., min(t0, t1)); tb = min(30., max(t0, t1));
    }
    if (ro.y > 0.1 && rd.y >= 0.) tb = ta;
    float t = ta + nj * 0.03;
    float sec = uSection > 0. ? uSection : (1.3 * S(4.5, 6.5, iTime) - 0.7 * S(12.0, 14.5, iTime)) * (1. - S(16.0, 17.5, iTime));
    for (int i = 0; i < 110; i++){
        if (t > tb || T < 0.02) break;
        float dt = 0.022 + 0.010 * t;
        vec3 pos = ro + rd * t;
        float ab;
        vec3 em = tissue(pos, ab);
        vec4 hc = heroChl(pos);
        em += hc.rgb * 1.6; ab += hc.a;
        float fade = exp(-t * uDepthFade);
        if (sec > 0.) fade *= 0.12 + 0.88 * exp(-pow((t - sec) / (0.35 + 0.25 * sec), 2.));
        col += T * em * dt * 0.9 * fade;
        T *= exp(-ab * dt * 2.0);
        t += dt;
    }
    col += co2(p, iTime) * uCO2;
    if (uFix > 0.) {
        vec3 rel = transpose(cam) * (HERO - ro);
        if (rel.z > 0.05) {
            vec2 hs = rel.xy / rel.z * fl;
            float d = length(p - hs);
            float n = gnoise(vec2(atan(p.y - hs.y, p.x - hs.x) * 6., iTime * 3.));
            col += GOLD * uFix * 0.25 * exp(-d * 5.) * (0.6 + 0.4 * n);
        }
    }
    if (uPull > 0.) {
        float x = p.x * 40.;
        float id = floor(x);
        float h = hash11(id);
        float y = fract(p.y * 0.8 - iTime * (0.5 + h) * 0.7 + h * 7.);
        float streak = exp(-pow((fract(x) - 0.5) / 0.08, 2.)) * smoothstep(0.0, 0.04, y) * smoothstep(0.10, 0.04, y) * step(0.90, h);
        col += mix(WALL, GOLD, h) * streak * 0.35 * uPull;
    }
    fragColor = vec4(col, 1.);
}
