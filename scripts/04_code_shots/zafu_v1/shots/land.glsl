// 山水 · out of the bamboo onto a ridge: a sea of clouds, peaks layered to the horizon, egrets; then it all becomes an ink painting
uniform vec3 uEye, uTarget; uniform float uFov, uRoll;
uniform float uInk;       // 0..1 progress of the ink transformation
uniform float uLeaves;    // leaves rushing past at the start (continuity with the bamboo shot)
uniform float uLeaf;      // ginkgo leaf fall progress (-1 off), 0..1
uniform float uRipple;    // seconds since the leaf touched the painting (-1 off)
uniform vec2  uLeafPos;   // screen position where the leaf lands
uniform float uBirds;     // seconds into the egret crossing (-1 off)

const vec3 SUN = normalize(vec3(-0.45, 0.13, 0.88));
const vec3 SUNC = vec3(1.0, 0.76, 0.50) * 5.0;

// ---------------------------------------------------------------- terrain: peaks rising from a cloud sea
float peaksH(vec2 x){
    vec2 q = x / 1500.;
    q += 0.28 * vec2(fbm3(q * 1.7 + 2.), fbm3(q * 1.7 + 5.));
    float f = fbm(q + 1.3);
    float r = 1. - abs(gnoise(q * 2.2 + 1.));
    float h = 980. * pow(smoothstep(-0.30, 0.70, f), 1.5) * (0.70 + 0.30 * r * r);
    h += 35. * fbm3(x / 220.);
    // keep the flight corridor open
    float corr = smoothstep(180., 560., abs(x.x - 60.)) + smoothstep(4400., 5200., x.y);
    h *= clamp(corr, 0.15, 1.);
    // the near ridge we start on (covered in bamboo)
    h = max(h, 268. * exp(-dot((x - vec2(0., -60.)) / vec2(520., 300.), (x - vec2(0., -60.)) / vec2(520., 300.))) - 12.);
    // distant high range
    h += smoothstep(5500., 9500., x.y) * 420. * smoothstep(-0.2, 0.6, fbm3(x / 2600. + 4.));
    return h - 30.;
}
float terrain(vec2 x){ return peaksH(x) + 3.5 * gnoise(x / 40.) + 1.2 * gnoise(x / 13.); }
float canopyTex(vec2 x){ return 1.6 * vnoise(x / 3.0) + 0.8 * vnoise(x / 1.2) + 0.35 * vnoise(x / 0.5); }
float cloudH(vec2 x){ return 18. + 26. * fbm(x / 420. + vec2(iGlobal * 0.006, 0.)) + 30. * fbm3(x / 1600.); }

float surfH(vec2 x, float t){
    float h = terrain(x);
    if (t < 300.) h += canopyTex(x) * (1. - smoothstep(120., 300., t));
    return h;
}
// returns t; kind 1 = ground, 2 = cloud
float march(vec3 ro, vec3 rd, float tmax, out int kind){
    float t = 0.3;
    kind = 0;
    for (int i = 0; i < 240; i++){
        vec3 p = ro + rd * t;
        float hg = surfH(p.xz, t);
        float hc = cloudH(p.xz);
        float d = p.y - max(hg, hc);
        if (d < 0.0015 * t) { kind = hg > hc ? 1 : 2; return t; }
        t += max(d * 0.5, 0.02 + t * 0.0012);
        if (t > tmax) break;
    }
    return -1.;
}
vec3 gNormal(vec2 x, float t){
    float e = max(0.25, t * 0.0012);
    float h0 = surfH(x, t), hx = surfH(x + vec2(e, 0.), t), hz = surfH(x + vec2(0., e), t);
    return normalize(vec3(h0 - hx, e, h0 - hz));
}
vec3 cNormal(vec2 x){
    float e = 6.;
    float h0 = cloudH(x), hx = cloudH(x + vec2(e, 0.)), hz = cloudH(x + vec2(0., e));
    return normalize(vec3(h0 - hx, e * 0.6, h0 - hz));
}
float softShadow(vec3 p){
    float res = 1., t = 2.;
    for (int i = 0; i < 26; i++){
        vec3 q = p + SUN * t;
        float d = q.y - terrain(q.xz);
        res = min(res, 9. * d / t);
        if (res < 0.01) break;
        t += max(5., t * 0.2);
        if (t > 4000.) break;
    }
    return clamp(res, 0., 1.);
}

vec3 skyCol(vec3 rd){
    float h = max(rd.y, 0.);
    vec3 zen = vec3(0.26, 0.42, 0.70), hor = vec3(0.86, 0.80, 0.72);
    vec3 c = mix(hor, zen, pow(h, 0.5)) * 0.85;
    float mu = max(dot(rd, SUN), 0.);
    c += vec3(1.0, 0.66, 0.36) * (0.45 * pow(mu, 6.) + 1.0 * pow(mu, 60.));
    c += SUNC * 30. * smoothstep(0.99955, 0.99975, mu);
    vec2 cp = rd.xz / max(rd.y, 0.04) * 0.5 + iGlobal * 0.003;
    float cl = smoothstep(0.15, 0.75, fbm(cp * 1.1 + 3.)) * smoothstep(0.0, 0.2, rd.y);
    c = mix(c, vec3(1.0, 0.90, 0.80) * (0.95 + 0.8 * pow(mu, 3.)), cl * 0.45);
    return c;
}
vec3 aerial(vec3 col, vec3 rd, float t){
    float mu = max(dot(rd, SUN), 0.);
    vec3 fc = mix(vec3(0.46, 0.55, 0.70), vec3(0.98, 0.80, 0.58), pow(mu, 8.)) * 0.78;
    float f = 1. - exp(-t * 0.00017);
    return mix(col, fc, f);
}

// ---------------------------------------------------------------- leaves (continuity from the canopy)
vec4 leafLayer(vec2 p, float depth, float t){
    vec4 acc = vec4(0.);
    float sc = mix(2.2, 7.0, depth);
    vec2 q = p * sc + vec2(depth * 17.3, -t * mix(9.0, 3.5, depth));
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
        float lit = smoothstep(0.78, 0.95, h.y);
        vec3 lc = mix(vec3(0.020, 0.040, 0.012), vec3(0.45, 0.66, 0.10) * 1.8, lit);
        acc.rgb = mix(acc.rgb, lc, m * (1. - acc.a));
        acc.a = max(acc.a, m);
    }
    return acc;
}

// ---------------------------------------------------------------- egrets
float egret(vec2 q, float flap){
    // a flying bird as two tapering brush strokes arching from a small body (flying left)
    float d = 1e3;
    for (int k = 0; k < 2; k++){
        float sd = k == 0 ? -1. : 1.;
        vec2 tip = vec2(0.85 * sd, 0.55 * flap - 0.25 * (1. - flap));
        vec2 mid = vec2(0.38 * sd, 0.26 + 0.22 * flap);
        vec2 prev = vec2(0.);
        for (int i = 1; i <= 8; i++){
            float t = float(i) / 8.;
            vec2 p = mix(mix(vec2(0.), mid, t), mix(mid, tip, t), t);
            d = min(d, sdSeg(q, prev, p) - mix(0.075, 0.012, t));
            prev = p;
        }
    }
    d = min(d, length((q - vec2(-0.02, -0.03)) / vec2(1.8, 1.0)) - 0.07);
    d = min(d, sdSeg(q, vec2(-0.12, -0.01), vec2(-0.27, 0.05)) - 0.022);
    return d;
}

// ---------------------------------------------------------------- ginkgo leaf (2D, painted)
float ginkgo(vec2 q, out float vein){
    vec2 b = q - vec2(0., 0.55);
    float r = length(b), a = atan(b.x, b.y);
    float spread = 1.05;
    float R = 0.62 * (1. - 0.22 * exp(-a * a / 0.004)) * (0.92 + 0.08 * cos(a * 3.));
    float fan = step(abs(a), spread) * (1. - smoothstep(R - 0.01, R + 0.01, r)) * smoothstep(0.02, 0.06, r + 0.06 * (1. - abs(a) / spread));
    fan *= 1. - smoothstep(R - 0.03, R, r) * step(0.5, fract(a * 9.)) * 0.3;
    float pet = (1. - smoothstep(0.008, 0.014, abs(q.x + 0.03 * q.y * q.y))) * step(0.18, q.y) * step(q.y, 0.58);
    vein = 0.5 + 0.5 * cos(a * 70.);
    return max(fan, pet);
}

void main(){
    vec2 p = screenP();
    vec3 ro = uEye;
    mat3 cam = lookAt(uEye, uTarget, uRoll);
    float fl = 0.5 / tan(radians(uFov) * 0.5);
    vec3 rd = normalize(cam * vec3(p, fl));

    int kind;
    float t = march(ro, rd, 16000., kind);
    vec3 col;
    float depth = 16000.;
    float inkD = 0.;          // ink density for the painting
    float isPeak = 0.;
    if (t > 0.) {
        vec3 pos = ro + rd * t;
        depth = t;
        if (kind == 1) {
            vec3 n = gNormal(pos.xz, t);
            float slope = 1. - n.y;
            float nC = fbm3(pos.xz / 80.), nD = vnoise(pos.xz / 7.);
            vec3 bamboo = vec3(0.24, 0.36, 0.09), forest = vec3(0.07, 0.14, 0.06), rock = vec3(0.30, 0.29, 0.26);
            vec3 alb = mix(bamboo, forest, smoothstep(200., 420., pos.y + 70. * nC) * 0.8);
            alb *= 0.8 + 0.4 * nD;
            alb = mix(alb, rock, smoothstep(0.42, 0.65, slope + 0.2 * nC));
            float dif = max(dot(n, SUN), 0.);
            float sh = softShadow(pos + n * 1.5);
            vec3 amb = vec3(0.36, 0.46, 0.62) * (0.5 + 0.5 * n.y);
            col = alb * (SUNC * dif * sh * 0.5 + amb * 0.9);
            col += alb * SUNC * 0.3 * pow(max(dot(rd, SUN), 0.), 3.) * sh * (1. - smoothstep(60., 400., t));   // backlit leaf sheen
            // peaks sink softly into the cloud sea
            float ch = cloudH(pos.xz);
            float sink = smoothstep(ch + 70., ch, pos.y);
            vec3 cloudC = vec3(0.72, 0.72, 0.76);
            col = mix(col, cloudC, sink * 0.9);
            col = aerial(col, rd, t);
            // painting: ink heavy near the top of each peak, dissolving downward into cloud; far peaks pale
            float above = smoothstep(ch + 10., ch + 260., pos.y);
            inkD = mix(0.22, 1.0, exp(-t / 6500.)) * (0.22 + 0.78 * above) * (0.72 + 0.28 * (1. - dif * sh));
            isPeak = 1.;
        } else {
            vec3 n = cNormal(pos.xz);
            float lit = smoothstep(-0.1, 0.5, dot(n, SUN) * 2.5 + 0.2);
            vec3 cc = mix(vec3(0.40, 0.47, 0.62) * 0.7, vec3(1.0, 0.91, 0.80) * 0.95, lit);
            float mu = max(dot(rd, SUN), 0.);
            cc += vec3(1.0, 0.72, 0.42) * pow(mu, 8.) * 0.8;      // sun glow across the cloud tops
            col = aerial(cc, rd, t * 0.7);
            inkD = 0.03 * (1. - lit);
        }
    } else {
        col = skyCol(rd);
    }

    // ---------------- egrets crossing
    if (uBirds >= 0.) {
        for (int i = 0; i < 7; i++){
            float fi = float(i);
            vec3 h = hash33(vec3(fi, 1.3, 7.7));
            float side = mod(fi, 2.) * 2. - 1.; float rank = floor((fi + 1.) / 2.);
            vec3 bp = vec3(250. - uBirds * 24. + rank * 12., 262. + side * rank * 4. + 2. * sin(uBirds * 0.6 + fi), 3990. + rank * 8. * side + h.x * 6.);
            vec3 rel = transpose(cam) * (bp - ro);
            if (rel.z < 1.) continue;
            vec2 sp = rel.xy / rel.z * fl;
            float sz = 8.0 / rel.z * fl;
            vec2 q = (p - sp) / sz;
            if (dot(q, q) > 2.) continue;
            float flap = 0.5 + 0.5 * sin(uBirds * 5.2 + fi * 1.7 + h.y * 3.);
            float d = egret(q, flap);
            float m = 1. - smoothstep(-0.01, 0.03, d);
            vec3 bc = vec3(0.95, 0.94, 0.90) * (SUNC * 0.12 + 0.6);
            col = mix(col, bc, m);
            inkD = max(inkD, m * 0.9);
        }
    }

    // ---------------- ink painting (水墨)
    if (uInk > 0.) {
        vec3 paper = vec3(0.925, 0.905, 0.855);
        float fib = 0.5 * gnoise(vec2(p.x * 900., p.y * 170.)) + 0.5 * gnoise(vec2(p.x * 150., p.y * 720.) + 3.);
        paper *= 1. + 0.012 * fib + 0.02 * fbm3(p * 2.6 + 11.);
        // silhouettes: depth discontinuities give a dry brush line along each ridge
        float ed = fwidth(log(depth + 1.)) * 3.0;
        float edge = smoothstep(0.05, 0.2, ed) * isPeak * mix(0.35, 1.0, exp(-depth / 4200.));
        float brk = smoothstep(-0.35, 0.35, gnoise(p * vec2(48., 9.) + 3.));
        // hemp-fibre strokes down the flanks
        float cun = smoothstep(0.25, 0.75, gnoise(vec2(p.x * 150. + 25. * gnoise(p * 4.), p.y * 10.))) * 0.17 * inkD
                  * smoothstep(-0.25, 0.45, gnoise(p * vec2(9., 34.) + 5.));
        // wash texture
        float wash = 0.85 + 0.3 * fbm3(p * 9. + 2.);
        float d = clamp(max(pow(inkD, 0.85) * wash + cun, edge * brk * 0.9), 0., 1.);
        vec3 inkC = vec3(0.12, 0.13, 0.14);
        vec3 painting = paper * mix(vec3(1.), inkC / paper, d);
        // the sun becomes a vermilion seal-like disc
        vec3 sunS = transpose(cam) * SUN;
        if (sunS.z > 0.) {
            vec2 ss = sunS.xy / sunS.z * fl;
            float sd = length(p - ss);
            float disc = 1. - smoothstep(0.034, 0.037, sd + 0.002 * gnoise(p * 200.));
            painting = mix(painting, vec3(0.78, 0.26, 0.16) * (0.92 + 0.08 * fbm3(p * 60.)), disc * 0.85);
        }
        // wet bloom spreading from the centre
        float n = fbm(p * 3.0 + 7.) * 0.9 + 0.1 * fbm(p * 18.);
        float dd = length(p * vec2(0.8, 1.0)) * 0.95 + (n - 0.5) * 0.6;
        float x = uInk * 1.55 - dd;
        float k = smoothstep(-0.03, 0.07, x);
        float front = exp(-abs(x) * 26.) * (1. - k) * step(0.001, uInk) * step(uInk, 0.999);
        vec3 pl = pow(painting, vec3(2.2)) * 1.12;
        col = mix(col, pl, k);
        col *= 1. - 0.3 * front;
    }

    // ---------------- ginkgo leaf and ripples
    if (uLeaf >= 0.) {
        float lt = uLeaf;
        vec2 lp = mix(uLeafPos + vec2(-0.30, 0.66), uLeafPos, 1. - (1. - lt) * (1. - lt)) + vec2(0.06 * sin(lt * 7.), 0.);
        float ang = 0.9 * sin(lt * 5.2) - 0.3 + 0.4 * lt;
        float sc = 0.10 * (1. - 0.15 * lt);
        vec2 q = rot(-ang) * (p - lp) / sc;
        q.x /= mix(0.55, 1.0, 0.5 + 0.5 * cos(lt * 4.3 * (1. - lt * 0.6)));   // the fan turns, never edge-on
        float vein;
        float m = ginkgo(q, vein);
        float vein2; float inner = ginkgo((q - vec2(0., 0.55)) * 1.08 + vec2(0., 0.55), vein2);
        float rim = clamp(m - inner, 0., 1.);
        float wash = 0.88 + 0.12 * fbm3(q * 6. + 3.);
        vec3 gold = mix(vec3(1.0, 0.78, 0.22), vec3(0.95, 0.60, 0.10), smoothstep(0.2, 0.9, q.y)) * 1.1;
        vec3 lc = gold * (0.86 + 0.14 * vein) * wash * (1. - 0.35 * rim);
        col = mix(col, lc, m * smoothstep(0., 0.05, lt));
    }
    if (uRipple >= 0.) {
        vec2 rq = (p - uLeafPos) * vec2(1., 2.6);
        float r = length(rq);
        float rings = 0.;
        for (int i = 0; i < 4; i++){
            float fi = float(i);
            float R = (uRipple - fi * 0.35) * 0.17;
            if (R > 0.) rings += exp(-pow((r - R) / 0.005, 2.)) * exp(-R * 4.);
        }
        col *= 1. - 0.55 * clamp(rings, 0., 1.);
    }

    // ---------------- leaves at the start
    if (uLeaves > 0.001) {
        for (int k = 0; k < 3; k++){
            float dep = float(k) / 2.;
            vec4 L = leafLayer(p, dep, iTime + 16.);
            col = mix(col, L.rgb, L.a * mix(0.95, 0.75, dep) * uLeaves);
        }
    }
    fragColor = vec4(col, 1.);
}
