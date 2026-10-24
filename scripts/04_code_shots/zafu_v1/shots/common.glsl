// shared helpers, prepended to every shot shader (after #version)
uniform vec2  iResolution;
uniform float iTime;      // shot-local time (s)
uniform float iGlobal;    // film time (s)
uniform vec2  iJitter;    // sub-pixel jitter of this subframe (pixels)
in vec2 uv;
out vec4 fragColor;

#define PI 3.14159265
#define TAU 6.28318531

float hash11(float p){ p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2  hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.xx + p3.yz) * p3.zy); }
vec3  hash32(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yzz) * p3.zyx); }
float hash13(vec3 p){ p = fract(p * .1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec3  hash33(vec3 p){ p = fract(p * vec3(.1031, .1030, .0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx); }

float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3. - 2. * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float gnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    float a = dot(hash22(i) * 2. - 1., f), b = dot(hash22(i + vec2(1, 0)) * 2. - 1., f - vec2(1, 0));
    float c = dot(hash22(i + vec2(0, 1)) * 2. - 1., f - vec2(0, 1)), d = dot(hash22(i + vec2(1, 1)) * 2. - 1., f - vec2(1, 1));
    return 1.4 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float vnoise3(vec3 p){
    vec3 i = floor(p), f = fract(p); vec3 u = f * f * (3. - 2. * f);
    float n000 = hash13(i), n100 = hash13(i + vec3(1, 0, 0)), n010 = hash13(i + vec3(0, 1, 0)), n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1)), n101 = hash13(i + vec3(1, 0, 1)), n011 = hash13(i + vec3(0, 1, 1)), n111 = hash13(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
const mat2 ROT2 = mat2(0.8, 0.6, -0.6, 0.8);
float fbm(vec2 p){ float s = 0., a = .5; for (int i = 0; i < 5; i++){ s += a * gnoise(p); p = ROT2 * p * 2.03 + 17.1; a *= .5; } return s; }
float fbm3(vec2 p){ float s = 0., a = .5; for (int i = 0; i < 3; i++){ s += a * gnoise(p); p = ROT2 * p * 2.03 + 17.1; a *= .5; } return s; }
const mat3 ROT3 = mat3(0.00, 0.80, 0.60, -0.80, 0.36, -0.48, -0.60, -0.48, 0.64);
float fbmv3(vec3 p){ float s = 0., a = .5; for (int i = 0; i < 4; i++){ s += a * vnoise3(p); p = ROT3 * p * 2.02 + 3.1; a *= .5; } return s; }

float S(float a, float b, float t){ float x = clamp((t - a) / (b - a), 0., 1.); return x * x * x * (x * (x * 6. - 15.) + 10.); }
float easeOut(float a, float b, float t){ float x = clamp((t - a) / (b - a), 0., 1.); return 1. - pow(1. - x, 3.); }
float easeIn(float a, float b, float t){ float x = clamp((t - a) / (b - a), 0., 1.); return x * x * x; }
float gauss(float x, float c, float w){ float d = (x - c) / w; return exp(-d * d); }
mat2  rot(float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float smin(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (b - a) / k, 0., 1.); return mix(b, a, h) - k * h * (1. - h); }
float smax(float a, float b, float k){ return -smin(-a, -b, k); }
float sdSeg(vec2 p, vec2 a, vec2 b){ vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.); return length(pa - ba * h); }

// camera: returns ray dir for screen point p (y in [-.5,.5]) with focal length fl
mat3 lookAt(vec3 ro, vec3 ta, float roll){
    vec3 w = normalize(ta - ro), u = normalize(cross(w, vec3(sin(roll), cos(roll), 0.))), v = cross(u, w);
    return mat3(u, v, w);
}
vec2 screenP(){ return (gl_FragCoord.xy + iJitter - 0.5 * iResolution.xy) / iResolution.y; }

// physically-inspired sky with a warm sun
vec3 sunSky(vec3 rd, vec3 sd, vec3 zen, vec3 hor, vec3 sunCol, float sunSize){
    float h = clamp(rd.y, -0.2, 1.);
    vec3 c = mix(hor, zen, pow(max(h, 0.), 0.55));
    float mu = max(dot(rd, sd), 0.);
    c += sunCol * (0.30 * pow(mu, 6.) + 0.60 * pow(mu, 64.));
    c += sunCol * 60. * smoothstep(1. - sunSize, 1. - sunSize * 0.6, mu);
    return c;
}
