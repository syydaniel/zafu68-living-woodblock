"""Relative depth (Depth Anything V2 Large, non-commercial use) for photo plates -> 16-bit PNG disparity in [0,1]
(1 = nearest). The 3D-photo cards turn it into geometry: 1/depth = lerp(1/far, 1/near, disparity).
  .venv_ai/bin/python tools/depth.py SRC [SRC...] [--out work/layers/depth]
"""
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent.parent
REPO = 'depth-anything/Depth-Anything-V2-Large-hf'


def main():
    args = [a for a in sys.argv[1:]]
    out = Path(args[args.index('--out') + 1]) if '--out' in args else ROOT / 'work/layers/depth'
    srcs = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] != '--out')]
    out.mkdir(parents=True, exist_ok=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    proc = AutoImageProcessor.from_pretrained(REPO)
    model = AutoModelForDepthEstimation.from_pretrained(REPO).to(dev).eval()
    for s in srcs:
        im = Image.open(s).convert('RGB')
        # run at a high working resolution (long side 1036, multiple of 14) for crisp edges
        scale = 1036 / max(im.size)
        wk = im.resize((round(im.width * scale / 14) * 14, round(im.height * scale / 14) * 14), Image.BICUBIC)
        inp = proc(images=wk, return_tensors='pt', do_resize=False).to(dev)
        with torch.no_grad():
            d = model(**inp).predicted_depth[0]
        d = torch.nn.functional.interpolate(d[None, None], size=(im.height, im.width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
        lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        dst = out / (Path(s).stem + '_d.png')
        Image.fromarray((d * 65535).astype(np.uint16)).save(dst)
        # geometry copy for the 3D-photo cards: float16, bottom row first (matches three.js uv), long side <= 1024
        k = 1024 / max(d.shape)
        g = np.asarray(Image.fromarray(d.astype(np.float32)).resize((round(d.shape[1] * k), round(d.shape[0] * k)), Image.BILINEAR))
        g = np.ascontiguousarray(np.flipud(g)).astype(np.float16)
        (out / (Path(s).stem + '_d.bin')).write_bytes(g.tobytes())
        (out / (Path(s).stem + '_d.json')).write_text(f'{{"w": {g.shape[1]}, "h": {g.shape[0]}}}')
        print(dst, d.shape, flush=True)


if __name__ == '__main__':
    main()
