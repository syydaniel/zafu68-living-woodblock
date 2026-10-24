// 光 background: blue-hour sky over dark forested ridges, low mist lit by the tree of light
uniform vec3 uEye, uTarget; uniform float uFov;
uniform float uTree;       // 0..1 tree brightness (lights the mist)
uniform float uLeafGlow;   // 0..1 warm glow of the leaf at the end

float ridge(float az, float k){
    return 0.04 + 0.05 * fbm3(vec2(az * (3. + k * 2.) + k * 7., k)) + 0.02 * gnoise(vec2(az * 22. + k * 3., k));
}

void main(){
    vec2 p = screenP();
    mat3 cam = lookAt(uEye, uTarget, 0.);
    float fl = 0.5 / tan(radians(uFov) * 0.5);
    vec3 rd = normalize(cam * vec3(p, fl));
    float h = rd.y;
    // sky: deep navy above, faint teal glow at the horizon
    vec3 col = mix(vec3(0.020, 0.040, 0.070), vec3(0.004, 0.008, 0.022), smoothstep(0.0, 0.6, h));
    col += vec3(0.05, 0.10, 0.12) * exp(-max(h, 0.) * 9.) * 0.8;
    // stars
    vec3 sd = rd * 260.;
    vec3 ci = floor(sd);
    float st = hash13(ci);
    if (st > 0.9975) {
        vec3 fp = fract(sd) - 0.5;
        float tw = 0.6 + 0.4 * sin(iTime * (1. + 3. * hash13(ci + 3.)) + st * 90.);
        col += vec3(0.8, 0.85, 1.0) * exp(-dot(fp, fp) * 40.) * tw * 0.9 * smoothstep(0.02, 0.2, h);
    }
    // layered ridges on the horizon
    float az = atan(rd.x, rd.z);
    for (int k = 0; k < 3; k++){
        float fk = float(k);
        float top = ridge(az, fk) - 0.012 * fk;
        float m = smoothstep(top + 0.002, top - 0.002, h);
        vec3 rc = mix(vec3(0.010, 0.020, 0.030), vec3(0.002, 0.004, 0.006), fk / 2.);
        col = mix(col, rc, m);
    }
    // ground: dark land, mist near the ground glowing under the tree
    if (rd.y < 0.) {
        float t = -uEye.y / rd.y;
        vec3 gp = uEye + rd * t;
        float d = length(gp.xz);
        col = vec3(0.004, 0.006, 0.008) * (0.8 + 0.4 * fbm3(gp.xz * 0.4));
        col += vec3(1.0, 0.78, 0.45) * uTree * 0.10 * exp(-d * 0.25);
        col += vec3(1.0, 0.75, 0.35) * uLeafGlow * 0.06 * exp(-d * 0.2);
    }
    // low mist band
    float mist = exp(-abs(h + 0.01) * 22.) * (0.5 + 0.5 * fbm3(vec2(az * 6. + iTime * 0.02, 1.)));
    col += vec3(0.06, 0.09, 0.11) * mist * 0.6 + vec3(1.0, 0.8, 0.5) * mist * uTree * 0.10;
    // the tree's light scattering in the air around it
    vec3 toC = normalize(vec3(0., 5.5, 0.) - uEye);
    float g = pow(max(dot(rd, toC), 0.), 12.);
    col += vec3(1.0, 0.80, 0.50) * g * (uTree * 0.10 + uLeafGlow * 0.12);
    fragColor = vec4(col, 1.);
}
