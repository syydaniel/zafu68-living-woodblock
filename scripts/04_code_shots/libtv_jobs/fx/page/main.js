// Offline three.js renderer for the LibTV repair effects.
// URL: index.html?fx=e3a_page|t1b_leaves&w=1920&h=1080&K=8&shutter=0.5
// window.renderFrame(i) renders frame i as the average of K sub-frames (motion blur) and returns the
// premultiplied RGBA bytes (bottom-up rows, as read by gl.readPixels) as base64.
import * as THREE from 'three';

const q = new URLSearchParams(location.search);
const W = +(q.get('w') || 1920), H = +(q.get('h') || 1080);
const K = +(q.get('K') || 8), SHUTTER = +(q.get('shutter') || 0.5);
const FX = q.get('fx');

const canvas = document.createElement('canvas');
canvas.width = W; canvas.height = H;
document.body.appendChild(canvas);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, premultipliedAlpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // raw shaders write display values directly
renderer.setClearColor(0x000000, 0);
const gl = renderer.getContext();

async function loadTex(url) {
  const t = await new THREE.TextureLoader().loadAsync(url);
  t.colorSpace = THREE.NoColorSpace;
  t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter; t.generateMipmaps = false;
  t.flipY = true;
  return t;
}

try {
  const mod = await import(`./${FX}.js`);
  const fx = await mod.create({ THREE, renderer, W, H, loadTex, q });
  const N = fx.frames;
  const px = new Uint8Array(W * H * 4);
  const acc = new Float32Array(W * H * 4);
  window.FRAMES = N;
  window.renderFrame = (i) => {
    // first and last frame: one exact sample so they match the cut points pixel for pixel
    const k = (i === 0 || i === N - 1) ? 1 : K;
    acc.fill(0);
    for (let s = 0; s < k; s++) {
      const off = k === 1 ? 0 : ((s + 0.5) / k - 0.5) * SHUTTER;
      const t = Math.min(N - 1, Math.max(0, i + off));
      fx.render(t);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
      for (let j = 0; j < px.length; j++) acc[j] += px[j];
    }
    for (let j = 0; j < px.length; j++) px[j] = Math.round(acc[j] / k);
    let bin = '';
    const CH = 0x8000;
    for (let j = 0; j < px.length; j += CH) bin += String.fromCharCode.apply(null, px.subarray(j, j + CH));
    return btoa(bin);
  };
  window.READY = true;
} catch (e) {
  window.BOOT_ERROR = String(e && e.stack || e);
}
