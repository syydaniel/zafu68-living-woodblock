"""Raw RGBA frames from render.mjs -> PNG (straight alpha), ProRes masters and QC previews.
  python post.py png <fx> <w> <h> [--alpha]        raw -> out/<fx>/png/%05d.png
  python post.py prores <fx> <name> [--alpha]       png -> out/<name>.mov (422 HQ, or 4444 with alpha), 25 fps
"""
import sys, os, glob, subprocess
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def png(fx, w, h, alpha):
    src = os.path.join(HERE, 'out', fx, 'raw'); dst = os.path.join(HERE, 'out', fx, 'png'); os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, '*.rgba'))):
        a = np.frombuffer(open(f, 'rb').read(), np.uint8).reshape(h, w, 4)[::-1].astype(np.float32)
        if alpha:
            al = a[..., 3:4]
            rgb = np.where(al > 0, a[..., :3] * 255.0 / np.maximum(al, 1), 0)      # premultiplied -> straight
            im = np.concatenate([np.clip(rgb, 0, 255), al], -1).round().astype(np.uint8)
            Image.fromarray(im, 'RGBA').save(os.path.join(dst, os.path.basename(f)[:-5] + '.png'))
        else:
            Image.fromarray(a[..., :3].round().astype(np.uint8), 'RGB').save(os.path.join(dst, os.path.basename(f)[:-5] + '.png'))


def prores(fx, name, alpha):
    src = os.path.join(HERE, 'out', fx, 'png', '%05d.png'); dst = os.path.join(HERE, 'out', name + '.mov')
    if alpha:
        enc = ['-c:v', 'prores_ks', '-profile:v', '4444', '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16']
    else:
        enc = ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le']
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', '25', '-i', src, *enc, '-vendor', 'apl0', dst], check=True)
    print(dst)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'png': png(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), '--alpha' in sys.argv)
    elif cmd == 'prores': prores(sys.argv[2], sys.argv[3], '--alpha' in sys.argv)
