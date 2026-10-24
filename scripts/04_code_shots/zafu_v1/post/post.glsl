//@@ accum
// add one HDR subframe into the accumulation buffer (sanitised)
uniform sampler2D uSrc; uniform float uWeight;
in vec2 uv; out vec4 fragColor;
void main(){
    vec4 c = texture(uSrc, uv);
    if (any(isnan(c)) || any(isinf(c))) c = vec4(0.);
    fragColor = vec4(clamp(c.rgb, 0., 3e3) * uWeight, uWeight);
}

//@@ comp
// transition between two shots
uniform sampler2D uA; uniform sampler2D uB; uniform float uMix; uniform int uType; uniform vec2 iResolution; uniform float uSeed;
in vec2 uv; out vec4 fragColor;
#define PI_ 3.14159265
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3. - 2. * f);
    return mix(mix(h12(i), h12(i + vec2(1, 0)), u.x), mix(h12(i + vec2(0, 1)), h12(i + vec2(1, 1)), u.x), u.y); }
float fb(vec2 p){ float s = 0., a = .5; for (int i = 0; i < 5; i++){ s += a * vn(p); p = p * 2.03 + 7.1; a *= .5; } return s; }
void main(){
    vec3 a = texture(uA, uv).rgb, b = texture(uB, uv).rgb;
    float m = clamp(uMix, 0., 1.);
    vec3 c;
    vec2 p = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.);
    if (uType == 0) {            // dissolve in a perceptual space
        c = exp(mix(log(a + 1e-3), log(b + 1e-3), m)) - 1e-3;
        c = mix(c, mix(a, b, m), 0.5);
    } else if (uType == 1) {     // through white light
        float up = smoothstep(0., 0.5, m), dn = smoothstep(0.5, 1., m);
        float fl = sin(PI_ * m);
        c = mix(a, b, smoothstep(0.35, 0.65, m)) + vec3(1.0, 0.96, 0.88) * 6. * pow(fl, 3.);
    } else if (uType == 2) {     // ink bloom: B spreads through noisy wet edges from the centre
        float n = fb(p * 3.0 + uSeed) * 0.9 + 0.1 * fb(p * 18. + uSeed);
        float d = length(p * vec2(0.85, 1.0)) * 0.9 + (n - 0.5) * 0.55;
        float x = m * 1.45 - d;
        float k = smoothstep(-0.02, 0.06, x);
        float edge = exp(-abs(x - 0.0) * 30.) * (1. - k) * step(0.001, m) * step(m, 0.999);
        c = mix(a, b, k) * (1. - 0.35 * edge);
    } else if (uType == 3) {     // rising wipe with soft organic edge (bottom to top)
        float n = fb(vec2(p.x * 4., uSeed)) - 0.5;
        float y = uv.y + 0.08 * n;
        float k = smoothstep(m * 1.3 - 0.15, m * 1.3 - 0.0, 1.0 - y);
        c = mix(a, b, 1. - k);
    } else {                     // luma key of A: bright parts of B first
        float l = dot(b, vec3(0.2126, 0.7152, 0.0722));
        float k = smoothstep(1. - m * 1.4, 1.2 - m * 1.4, clamp(l / (1. + l) + 0.2 * fb(p * 5.), 0., 1.));
        c = mix(a, b, max(k, smoothstep(0.7, 1.0, m)));
    }
    fragColor = vec4(c, 1.);
}

//@@ down
// 13-tap downsample (Jimenez 2014); Karis average on the first level against fireflies
uniform sampler2D uSrc; uniform vec2 uTexel; uniform int uFirst;
in vec2 uv; out vec4 fragColor;
vec3 k(vec3 c){ return c / (1. + dot(c, vec3(0.2126, 0.7152, 0.0722)) * 0.25); }
void main(){
    vec2 t = uTexel;
    vec3 a = texture(uSrc, uv + t * vec2(-2, 2)).rgb, b = texture(uSrc, uv + t * vec2(0, 2)).rgb, c = texture(uSrc, uv + t * vec2(2, 2)).rgb;
    vec3 d = texture(uSrc, uv + t * vec2(-2, 0)).rgb, e = texture(uSrc, uv).rgb, f = texture(uSrc, uv + t * vec2(2, 0)).rgb;
    vec3 g = texture(uSrc, uv + t * vec2(-2, -2)).rgb, h = texture(uSrc, uv + t * vec2(0, -2)).rgb, i = texture(uSrc, uv + t * vec2(2, -2)).rgb;
    vec3 j = texture(uSrc, uv + t * vec2(-1, 1)).rgb, l = texture(uSrc, uv + t * vec2(1, 1)).rgb;
    vec3 m = texture(uSrc, uv + t * vec2(-1, -1)).rgb, n = texture(uSrc, uv + t * vec2(1, -1)).rgb;
    vec3 o;
    if (uFirst == 1) {
        o = (k(j) + k(l) + k(m) + k(n)) * 0.125 + (k(a) + k(b) + k(d) + k(e)) * 0.03125 + (k(b) + k(c) + k(e) + k(f)) * 0.03125
          + (k(d) + k(e) + k(g) + k(h)) * 0.03125 + (k(e) + k(f) + k(h) + k(i)) * 0.03125;
    } else {
        o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + l + m + n) * 0.125;
    }
    fragColor = vec4(o, 1.);
}

//@@ up
// 9-tap tent upsample, added onto the next larger level
uniform sampler2D uSrc; uniform vec2 uTexel; uniform float uRadius;
in vec2 uv; out vec4 fragColor;
void main(){
    vec2 t = uTexel * uRadius;
    vec3 s = texture(uSrc, uv).rgb * 4.;
    s += (texture(uSrc, uv + vec2(-t.x, 0)).rgb + texture(uSrc, uv + vec2(t.x, 0)).rgb + texture(uSrc, uv + vec2(0, -t.y)).rgb + texture(uSrc, uv + vec2(0, t.y)).rgb) * 2.;
    s += texture(uSrc, uv - t).rgb + texture(uSrc, uv + t).rgb + texture(uSrc, uv + vec2(-t.x, t.y)).rgb + texture(uSrc, uv + vec2(t.x, -t.y)).rgb;
    fragColor = vec4(s / 16., 1.);
}

//@@ streak
// horizontal anamorphic streak from a small bloom level
uniform sampler2D uSrc; uniform vec2 uTexel; uniform float uStep;
in vec2 uv; out vec4 fragColor;
void main(){
    vec3 s = vec3(0.); float w = 0.;
    for (int i = -12; i <= 12; i++){
        float fi = float(i); float k = exp(-abs(fi) * 0.16);
        vec3 c = texture(uSrc, uv + vec2(fi * uStep * uTexel.x, 0.)).rgb;
        s += max(c - 0.6, 0.) * k; w += k;
    }
    fragColor = vec4(s / w, 1.);
}

//@@ final
// exposure, bloom, halation, ACES, grade, chromatic aberration, vignette, grain, letterbox
uniform sampler2D uScene; uniform sampler2D uBloom; uniform sampler2D uStreak;
uniform vec2 iResolution; uniform float uFrame;
uniform float uExposure, uBloomAmt, uHalation, uStreakAmt, uSat, uContrast, uTemp, uTint, uVignette, uGrain, uAspect, uFade, uFlash, uCA, uZoom, uBloomNorm;
uniform vec3 uShadow, uHigh;
in vec2 uv; out vec4 fragColor;
vec3 RRTAndODTFit(vec3 v){ vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
vec3 aces(vec3 c){
    const mat3 I = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    const mat3 O = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    return clamp(O * RRTAndODTFit(I * c), 0., 1.);
}
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main(){
    vec2 zuv = (uv - 0.5) / max(uZoom, 0.5) + 0.5;      // beat zoom punch
    vec2 c = zuv - 0.5;
    float r2 = dot(c, c);
    // lateral chromatic aberration grows toward the edges (and with the punch)
    vec2 off = c * r2 * (uCA + (uZoom - 1.) * 0.6);
    vec3 col;
    col.r = texture(uScene, zuv - off).r;
    col.g = texture(uScene, zuv).g;
    col.b = texture(uScene, zuv + off).b;
    vec3 bl = texture(uBloom, zuv).rgb / max(uBloomNorm, 1.0);   // the up-chain sums 7 levels
    col += bl * uBloomAmt;
    // halation: warm red fringe around strong highlights (film)
    col += vec3(1.0, 0.32, 0.12) * max(bl - 1.0, 0.) * uHalation;
    col += texture(uStreak, zuv).rgb * vec3(0.55, 0.75, 1.0) * uStreakAmt;
    col *= exp2(uExposure);
    // white balance (temperature / tint)
    col *= vec3(1. + uTemp * 0.10, 1. + uTint * 0.06, 1. - uTemp * 0.10);
    col = aces(col);
    // grade in display space
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(l), col, uSat);
    col = clamp((col - 0.5) * uContrast + 0.5, 0., 1.);
    float sh = 1. - smoothstep(0.0, 0.45, l), hi = smoothstep(0.45, 1.0, l);
    col += uShadow * sh * 0.08 + uHigh * hi * 0.06;
    col = pow(max(col, 0.), vec3(1. / 2.2));
    // vignette
    col *= 1. - uVignette * smoothstep(0.15, 0.85, r2 * 2.2);
    // flash / fade
    col = mix(col, vec3(1.), clamp(uFlash, 0., 1.));
    col *= 1. - clamp(uFade, 0., 1.);
    // grain, stronger in the mids
    float g = h12(gl_FragCoord.xy + fract(uFrame * 0.6180339) * 1000.) + h12(gl_FragCoord.xy * 1.37 + fract(uFrame * 0.3819) * 713.) - 1.;
    float lm = dot(col, vec3(0.333));
    col += g * uGrain * (0.35 + 0.65 * 4. * lm * (1. - lm));
    // letterbox
    float ar = iResolution.x / iResolution.y;
    float barH = max(0., 0.5 - 0.5 * ar / uAspect);
    float y = uv.y;
    float inside = smoothstep(barH - 0.0008, barH + 0.0008, y) * smoothstep(barH - 0.0008, barH + 0.0008, 1. - y);
    col *= inside;
    col += (h12(gl_FragCoord.xy + uFrame) - 0.5) / 1023.;
    fragColor = vec4(clamp(col, 0., 1.), 1.);
}

//@@ text
// one text layer over the display image (premultiplied)
uniform sampler2D uText; uniform vec2 iResolution;
uniform vec2 uCenter; uniform vec2 uSize;        // screen units (y in [-.5,.5])
uniform float uIn, uOut, uGlow, uBlur, uRise, uSeed, uAlpha;
uniform int uStyle; uniform vec3 uColor; uniform vec3 uGlowColor; uniform float uTime;
in vec2 uv; out vec4 fragColor;
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3. - 2. * f);
    return mix(mix(h12(i), h12(i + vec2(1, 0)), u.x), mix(h12(i + vec2(0, 1)), h12(i + vec2(1, 1)), u.x), u.y); }
float fb(vec2 p){ float s = 0., a = .5; for (int i = 0; i < 4; i++){ s += a * vn(p); p = p * 2.03 + 7.1; a *= .5; } return s; }
void main(){
    vec2 p = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    vec2 q = (p - uCenter) / uSize + 0.5;
    // rise: glyphs drift up into place while revealing
    float pin = clamp(uIn, 0., 1.), pout = clamp(uOut, 0., 1.);
    vec2 tq = q;
    if (uStyle == 2) tq.y -= uRise * (1. - pin) * (1. - pin);
    if (tq.x < -0.05 || tq.x > 1.05 || tq.y < -0.1 || tq.y > 1.1) { fragColor = vec4(0.); return; }
    vec2 tuv = tq;
    float lod = uBlur * (1. - pin) + uBlur * pout;
    vec4 T = textureLod(uText, tuv, lod);
    float inside = step(0., tq.x) * step(tq.x, 1.) * step(0., tq.y) * step(tq.y, 1.);
    float cov = T.r * inside, sdf = T.g, tf = T.b;
    float a = 0.; float glowA = 0.;
    if (uStyle == 0) {                 // focus: blur to sharp, fade
        a = cov * pin * (1. - pout);
        glowA = smoothstep(0.25, 0.5, sdf) * pin * (1. - pout);
    } else if (uStyle == 1 || uStyle == 3) {   // ink / brush: reveal follows the time field, wet noisy edge
        float n = fb(tq * vec2(40., 40. * uSize.y / uSize.x) + uSeed) - 0.5;
        float x = pin * 1.08 - tf + 0.05 * n;
        float rev = smoothstep(-0.015, 0.01, x);
        float wet = exp(-max(x, 0.) * 40.) * rev;
        a = cov * rev * (1. - pout);
        a *= 0.92 + 0.08 * fb(tq * 180. + uSeed);         // dry-brush texture
        glowA = smoothstep(0.3, 0.5, sdf) * rev * (1. - pout) + wet * cov * 0.8 * (1. - pout);
        if (uStyle == 1) a = max(a, smoothstep(0.42, 0.5, sdf) * wet * 0.4 * (1. - pout));
    } else {                           // rise: per-glyph stagger through the time field
        float g = clamp(pin * 1.6 - tf * 0.6, 0., 1.);
        a = cov * smoothstep(0., 1., g) * (1. - pout);
        glowA = smoothstep(0.25, 0.5, sdf) * g * (1. - pout);
    }
    a *= uAlpha; glowA *= uAlpha;
    vec3 col = uColor * a + uGlowColor * glowA * uGlow * (1. - a);
    float al = clamp(a + glowA * uGlow * 0.5 * (1. - a), 0., 1.);
    fragColor = vec4(col, al);
}

//@@ show
uniform sampler2D uSrc; in vec2 uv; out vec4 fragColor;
void main(){ fragColor = texture(uSrc, uv); }
