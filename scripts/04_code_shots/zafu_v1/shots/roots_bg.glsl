// 根 background: dark humus volume, faintly lit by the seed; shock ring at landing; light from above at the end
uniform vec3 uSeedScr;   // screen pos (y in [-.5,.5]), z < 0 = no seed
uniform float uSeedI, uShock, uUp;

void main(){
    vec2 p = screenP();
    // deep warm black with slow drifting soil haze
    float n1 = fbm(p * 2.2 + vec2(0.0, iTime * 0.01));
    float n2 = fbm(p * 7.0 - vec2(iTime * 0.006, 0.0) + 4.);
    vec3 col = vec3(0.010, 0.008, 0.006) * (0.7 + 0.6 * n1);
    // seed light scattering in the haze
    if (uSeedScr.z > 0.) {
        float d = length(p - uSeedScr.xy);
        float haze = 0.5 + 0.5 * n2;
        col += vec3(1.0, 0.62, 0.30) * uSeedI * (0.020 * exp(-d * 3.5) + 0.05 * exp(-d * 12.)) * haze;
        // shock ring spreading through the soil
        float ring = exp(-pow((d - uShock * 0.16) / (0.012 + 0.03 * uShock * 0.16), 2.)) * exp(-uShock * 0.9) * step(0.001, uShock);
        col += vec3(1.0, 0.75, 0.45) * ring * 0.35 * (0.6 + 0.8 * n2);
    }
    // near the end the soil above thins: shafts of daylight reach down toward the rising shoot, warm green light from the top
    float up = uUp;
    float shafts = smoothstep(0.35, 0.95, 0.5 + 0.5 * gnoise(vec2(p.x * 9. + 0.3 * p.y, 3.1)) + 0.25 * gnoise(vec2(p.x * 23., 7.)));
    float fromTop = exp(-(0.5 - p.y) * mix(6., 1.6, up));
    col += vec3(0.95, 0.88, 0.62) * up * up * shafts * fromTop * (0.6 + 0.4 * n2) * 1.4;
    col += vec3(0.45, 0.70, 0.40) * up * up * up * fromTop * 1.2;
    col += vec3(1.0, 0.96, 0.86) * pow(up, 7.) * smoothstep(-0.1, 0.5, p.y) * 5.;
    fragColor = vec4(col, 1.);
}
