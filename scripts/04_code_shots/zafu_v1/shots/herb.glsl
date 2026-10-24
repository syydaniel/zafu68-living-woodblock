// 草木 · a floating herbarium: five leaves backlit in the dark, veins glowing; the camera tracks past them with rack focus
uniform vec3 uEye, uTarget; uniform float uFov, uAperture, uFocus;

const int NL = 5;

// leaf placements: position, yaw, pitch, size
vec3 leafPos(int i){
    return i == 0 ? vec3(0.0, 0.10, 0.0) : i == 1 ? vec3(1.65, 0.12, 0.35) : i == 2 ? vec3(3.30, 0.08, -0.10)
         : i == 3 ? vec3(4.95, 0.10, 0.25) : vec3(6.60, 0.08, 0.0);
}
float leafSize(int i){ return i == 0 ? 0.52 : i == 1 ? 0.66 : i == 2 ? 0.56 : i == 3 ? 0.54 : 0.50; }
mat3 leafBasis(int i, float t){
    float fi = float(i);
    float yaw = 0.25 * sin(t * 0.35 + fi * 1.7) + (hash11(fi) - 0.5) * 0.5;
    float pitch = 0.12 * sin(t * 0.41 + fi * 2.3);
    float roll = (hash11(fi + 7.) - 0.5) * 0.6 + 0.06 * sin(t * 0.3 + fi);
    mat2 ry = rot(yaw), rx = rot(pitch), rz = rot(roll);
    vec3 X = vec3(1., 0., 0.), Y = vec3(0., 1., 0.), Z = vec3(0., 0., 1.);
    X.xy = rz * X.xy; Y.xy = rz * Y.xy;
    X.yz = rx * X.yz; Y.yz = rx * Y.yz; Z.yz = rx * Z.yz;
    X.xz = ry * X.xz; Y.xz = ry * Y.xz; Z.xz = ry * Z.xz;
    return mat3(X, Y, Z);
}

// ---------------------------------------------------------------- species (leaf space: petiole base at origin, blade up, ~1 long)
// returns x = inside (0..1 AA), y = vein strength, z = midrib
vec3 ginkgoL(vec2 q, float aa){
    vec2 b = q - vec2(0., 0.42);
    float r = length(b), a = atan(b.x, b.y);
    float spread = 1.08;
    float R = 0.56 * (1. - 0.20 * exp(-a * a / 0.003)) * (0.93 + 0.07 * cos(a * 3.));
    R *= 1. - 0.018 * sin(a * 11. + 0.5) - 0.010 * sin(a * 29.);          // gently wavy outer margin
    float d = max(r - R, abs(a) - spread);
    d = max(d, -(r - 0.03));
    float inside = 1. - smoothstep(-aa, aa, d * (r > 0.1 ? 1. : 1.));
    float pet = 1. - smoothstep(-aa, aa, abs(q.x + 0.02 * q.y * q.y) - 0.0055 * (1. + q.y));
    pet *= step(0., q.y) * step(q.y, 0.44);
    float vein = pow(0.5 + 0.5 * cos(a * 80. + 3. * r), 8.) * smoothstep(0.05, 0.2, r) * 0.6;
    return vec3(max(inside, pet), vein * inside, pet);
}
vec3 bambooL(vec2 q, float aa){
    float y = q.y;
    if (y < -0.05 || y > 1.02) return vec3(0.);
    float u = clamp(y, 0., 1.);
    float w = 0.105 * pow(max(sin(PI * pow(u, 0.62)), 0.), 1.15) * (1. - 0.2 * u);
    float d = abs(q.x + 0.02 * sin(u * 2.)) - w;
    float inside = (1. - smoothstep(-aa, aa, d)) * step(0., y);
    float xn = (q.x + 0.02 * sin(u * 2.)) / max(w, 1e-3);
    float par = pow(0.5 + 0.5 * cos(xn * 7. * PI), 10.) * 0.7;
    float cross = pow(0.5 + 0.5 * cos(y * 180.), 30.) * 0.25;
    float mid = exp(-xn * xn * 60.);
    float pet = (1. - smoothstep(-aa, aa, abs(q.x) - 0.008)) * step(-0.08, y) * step(y, 0.02);
    return vec3(max(inside, pet), (par + cross) * inside, mid * inside + pet);
}
vec3 lirioL(vec2 q, float aa){
    // tulip-tree leaf (马褂木): broad truncated tip with a shallow notch, two pointed lobes on each side
    vec2 b = q - vec2(0., 0.46);
    float r = length(b), a = atan(b.x, b.y);
    float R = 0.30 + 0.15 * pow(abs(sin(2. * a)), 1.6);
    float ca = cos(a);
    if (ca > 0.) R = min(R, 0.31 / max(ca, 1e-3));                    // flat top
    R -= 0.07 * exp(-a * a / 0.012);                                    // notch
    if (ca < -0.3) R = min(R, 0.30 / max(-ca, 1e-3));                  // truncated base
    float d = r - R;
    float inside = 1. - smoothstep(-aa, aa, d);
    float mid = exp(-b.x * b.x * 1600.) * step(-0.40, b.y) * step(b.y, 0.26);
    float lat = pow(0.5 + 0.5 * cos(a * 7. - r * 8.), 16.) * smoothstep(0.03, 0.10, r) * 0.7;
    float pet = (1. - smoothstep(-aa, aa, abs(q.x) - 0.006)) * step(-0.12, q.y) * step(q.y, 0.18);
    return vec3(max(inside, pet), lat * inside, mid * inside + pet);
}
vec3 mapleL(vec2 q, float aa){
    // palmate, seven pointed lobes, finely toothed
    vec2 b = q - vec2(0., 0.34);
    float r = length(b), a = atan(b.x, b.y);
    float lobes = max(0., 1. - pow(abs(sin(3.5 * a)), 0.72));
    float R = 0.13 + 0.40 * lobes;
    R *= 1. - 0.45 * smoothstep(2.3, 2.9, abs(a));
    R *= 1. + 0.010 * sin(a * 160.) * smoothstep(0.3, 0.8, lobes);
    float d = r - R;
    float inside = 1. - smoothstep(-aa, aa, d);
    // main veins run to each lobe tip
    float k = fract(a * 3.5 / PI + 0.5) - 0.5;
    float vein = exp(-k * k * 120.) * smoothstep(0.02, 0.08, r);
    float sec = pow(0.5 + 0.5 * cos(a * 40. + r * 30.), 12.) * 0.3;
    float pet = (1. - smoothstep(-aa, aa, abs(q.x) - 0.0045)) * step(-0.14, q.y) * step(q.y, 0.34);
    return vec3(max(inside, pet), (vein + sec) * inside, pet);
}
vec3 teaL(vec2 q, float aa){
    // elliptic, serrate margin, pinnate veins looping near the edge
    vec2 b = (q - vec2(0., 0.50)) / vec2(0.19, 0.46);
    float r = length(b);
    float ang = atan(b.x, b.y);
    float serr = 0.018 * smoothstep(0.1, 0.6, abs(sin(ang))) * (0.5 + 0.5 * sin(ang * 46.));
    float d = (r - 1. + serr) * 0.19;
    float inside = 1. - smoothstep(-aa, aa, d);
    float mid = exp(-q.x * q.x * 2500.) * step(0.05, q.y) * step(q.y, 0.94);
    float yy = q.y - abs(q.x) * 1.6;
    float lat = pow(0.5 + 0.5 * cos(yy * 55.), 14.) * smoothstep(0.01, 0.03, abs(q.x)) * (1. - smoothstep(0.75, 0.95, r));
    float pet = (1. - smoothstep(-aa, aa, abs(q.x) - 0.009)) * step(-0.06, q.y) * step(q.y, 0.06);
    return vec3(max(inside, pet), lat * inside, mid * inside + pet);
}
vec3 species(int i, vec2 q, float aa){
    if (i == 0) return ginkgoL(q, aa);
    if (i == 1) return bambooL(q, aa);
    if (i == 2) return lirioL(q, aa);
    if (i == 3) return mapleL(q, aa);
    return teaL(q, aa);
}
vec3 leafTint(int i){
    return i == 0 ? vec3(1.00, 0.70, 0.12) : i == 1 ? vec3(0.42, 0.72, 0.16) : i == 2 ? vec3(0.70, 0.76, 0.18)
         : i == 3 ? vec3(0.95, 0.22, 0.06) : vec3(0.16, 0.44, 0.10);
}

void main(){
    vec2 p = screenP();
    vec3 ro = uEye;
    mat3 cam = lookAt(uEye, uTarget, 0.);
    float fl = 0.5 / tan(radians(uFov) * 0.5);
    vec3 rd = normalize(cam * vec3(p, fl));
    if (uAperture > 0.) {
        vec2 lj = iJitter + 0.5;
        float ang = lj.x * TAU, rr = sqrt(fract(lj.y * 7.13 + lj.x * 3.1));
        vec2 lens = vec2(cos(ang), sin(ang)) * rr * uAperture;
        vec3 fp = ro + rd * uFocus / dot(rd, cam[2]);
        ro += cam * vec3(lens, 0.);
        rd = normalize(fp - ro);
    }

    // background: deep green-black with a soft light behind the display
    vec3 L = normalize(vec3(0.25, 0.35, 1.0));       // light behind the leaves, toward +z
    float glow = pow(max(dot(rd, L), 0.), 6.);
    vec3 col = vec3(0.010, 0.016, 0.012) + vec3(0.10, 0.12, 0.08) * glow * 0.6 + vec3(0.02, 0.03, 0.02) * (0.5 + 0.5 * rd.y);
    col *= 0.8 + 0.4 * fbm3(rd.xy * 3. + iTime * 0.02);
    // floating dust in the light
    {
        for (int k = 0; k < 3; k++){
            float z = 1.2 + float(k) * 1.1;
            float tt = (z - ro.z) / rd.z;
            if (tt < 0.) continue;
            vec3 pp = ro + rd * tt;
            vec2 g = pp.xy * 7. + vec2(0., iTime * 0.05 * (1. + float(k)));
            vec2 id = floor(g), f = fract(g) - 0.5;
            vec3 h = hash33(vec3(id, float(k)));
            if (h.z > 0.93) {
                float d = length(f - (h.xy - 0.5) * 0.6);
                float coc = abs(tt - uFocus) * uAperture * 8. + 0.02;
                col += vec3(1.0, 0.9, 0.7) * 0.25 * exp(-d * d / (coc * coc * 0.5 + 0.002)) * (0.03 / (coc + 0.03));
            }
        }
    }

    // leaves: nearest plane hit
    float best = 1e9;
    for (int i = 0; i < NL; i++){
        vec3 c = leafPos(i);
        mat3 B = leafBasis(i, iTime);
        vec3 n = B[2];
        float den = dot(rd, n);
        if (abs(den) < 1e-4) continue;
        float t = dot(c - ro, n) / den;
        if (t < 0. || t > best) continue;
        vec3 hp = ro + rd * t;
        float s = leafSize(i);
        vec2 q = vec2(dot(hp - c, B[0]), dot(hp - c, B[1])) / s + vec2(0., 0.45);
        if (abs(q.x) > 0.7 || q.y < -0.2 || q.y > 1.1) continue;
        float aa = 1.2 * t / (s * iResolution.y * fl) ;
        vec3 sp = species(i, q, aa);
        if (sp.x < 0.002) continue;
        best = t;
        vec3 tint = leafTint(i);
        // base-to-tip colour drift (greener at the base for ginkgo and maple), gentle desaturation
        if (i == 0) tint = mix(vec3(0.72, 0.70, 0.16), tint, smoothstep(0.35, 0.8, q.y));
        if (i == 3) tint = mix(vec3(0.80, 0.42, 0.06), tint, smoothstep(0.2, 0.6, q.y));
        tint = mix(vec3(dot(tint, vec3(0.3, 0.55, 0.15))), tint, 0.82);
        // margin: coverage just inside the outline, sampled slightly shrunk
        vec3 inner = species(i, (q - vec2(0., 0.45)) * 1.025 + vec2(0., 0.45), aa * 2.);
        float margin = clamp(sp.x - inner.x, 0., 1.) * 0.6;
        // transmission through the blade: veins lighter and warmer, mottled mesophyll
        float cells = 0.80 + 0.14 * vnoise(q * 300.) + 0.12 * fbm3(q * 9. + float(i) * 3.);
        float thick = 1. - 0.30 * sp.y - 0.50 * sp.z;
        vec3 trans = tint * tint * 1.6 * cells * thick * (1. - 0.45 * margin)
                   + vec3(1.0, 0.88, 0.55) * (0.55 * sp.y + 0.9 * sp.z) * tint;
        float back = max(dot(rd, L), 0.);
        vec3 lc = trans * (0.25 + 1.6 * pow(back, 2.)) ;
        // front key from above-left: a faint sheen on the surface
        vec3 K = normalize(vec3(-0.5, 0.7, -0.5));
        float spec = pow(max(dot(reflect(rd, n * sign(-den)), K), 0.), 40.);
        lc += vec3(0.8, 0.85, 0.9) * spec * 0.25;
        // darker, drier margin
        col = mix(col, lc, sp.x);
    }
    fragColor = vec4(col, 1.);
}
