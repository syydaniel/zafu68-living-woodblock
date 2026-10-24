"""Composite an overlay's raw frames over background PNGs for QC.  python comp.py <fx> <bgdir> <w> <h> <out.jpg> <frames...>"""
import sys, os, numpy as np
from PIL import Image
fx, bgdir, w, h, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fr = [int(x) for x in sys.argv[6:]]
tiles = []
for j in fr:
    a = np.frombuffer(open(f'out/{fx}/raw/{j:05d}.rgba', 'rb').read(), np.uint8).reshape(h, w, 4)[::-1].astype(np.float32) / 255
    bg = np.asarray(Image.open(f'{bgdir}/bg_{j:03d}.png').convert('RGB').resize((w, h))).astype(np.float32) / 255
    c = a[..., :3] + bg * (1 - a[..., 3:4])
    tiles.append(Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((640, 360)))
cols = 4; rows = (len(tiles) + cols - 1) // cols
o = Image.new('RGB', (cols * 640, rows * 360))
for i, t in enumerate(tiles): o.paste(t, ((i % cols) * 640, (i // cols) * 360))
o.save(out, quality=88)
