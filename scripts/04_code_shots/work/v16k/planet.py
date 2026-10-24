"""Opening 0-5.6 s (小星球, little planet): the ink print of Tianmu (T1_tianmu_ink, = AI-1's first frame) and its mirror make one
seamless 3840-wide scroll; bent into a little planet it turns slowly on xuan paper while the question rises; a samara spins down
onto it; the planet unrolls (curvature 1 -> 0) while the camera dives, and the last frame IS the flat print, so AI-1 continues it.
  Planet(t) -> float32 RGB 1080p frame"""
import math
from pathlib import Path
import numpy as np, cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W, H = 1920, 1080
PAPER = np.array([0.935, 0.905, 0.835], np.float32)


def ease(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def ramp(t, a, b): return min(1.0, max(0.0, (t - a) / (b - a)))


class Planet:
    T_UNROLL0, T_UNROLL1, END = 3.3, 5.6, 5.6

    def __init__(self, print_png=ROOT / 'work/v16t/T1/T1_tianmu_ink_1080.png'):
        a = np.asarray(Image.open(print_png).convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
        self.flat = a
        pano = np.concatenate([a, a[:, ::-1]], 1)                       # seamless both ways round
        forest = a[740:1080, 0:620]                                      # dense carved forest from the print's lower left
        tile = np.concatenate([forest, forest[:, ::-1]], 1)
        tile = np.concatenate([tile, tile[::-1]], 0)
        reps = (1500 // tile.shape[0] + 1, pano.shape[1] // tile.shape[1] + 1)
        under = np.tile(tile, (reps[0], reps[1], 1))[:1500, :pano.shape[1]]   # the planet body is forest all the way down
        self.pano = np.concatenate([pano, under], 0)                     # rows 0..H-1 the print, below: more forest
        self.Wp = pano.shape[1]
        pap = np.asarray(Image.open(ROOT / 'data/v14/paper.png').convert('RGB').resize((W, H), Image.LANCZOS), np.float32) / 255
        self.paper = pap / pap.reshape(-1, 3).mean(0) * PAPER
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy

    def warp(self, k, s, piv_screen, spin):
        """inverse map: screen -> scroll coordinates for curvature k (1 planet .. 0 flat), scale s"""
        xc = W / 2 + spin
        if k < 1e-3:
            px = (self.xx - piv_screen[0]) / s + xc; py = (self.yy - piv_screen[1]) / s + H
            return px, py
        rho0 = self.Wp / (2 * math.pi * k)
        cy = H + rho0                                                    # centre of curvature, below the print
        px = (self.xx - piv_screen[0]) / s + xc; py = (self.yy - piv_screen[1]) / s + cy
        dx = px - xc; dy = cy - py
        r = np.sqrt(dx * dx + dy * dy); al = np.arctan2(dx, dy)
        x = xc + al * self.Wp / (2 * math.pi * k)
        y = H - (r - rho0)                                               # height above the print's bottom edge -> row
        return x, y

    def frame(self, t):
        u = ease(ramp(t, self.T_UNROLL0, self.T_UNROLL1))
        k = (1 - u) ** 1.6                                               # curvature relaxes, fastest at the end
        rho0 = self.Wp / (2 * math.pi * max(k, 1e-3))
        # planet state: whole planet (radius rho0 + H at k=1) fits in ~0.42 of the frame height; flat state: s = 1
        s_planet = 0.42 * H / (self.Wp / (2 * math.pi) + H)
        s = math.exp(math.log(s_planet) * (1 - u))                       # geometric zoom reads as a steady dive
        # control where the middle of the print's bottom edge sits on screen: planet state -> the planet is centred,
        # flat state -> it is the bottom-centre of the frame; the pivot (centre of curvature) follows from it
        rho1 = self.Wp / (2 * math.pi)
        by = (H / 2 + 20 - s_planet * rho1) * (1 - u) + H * u
        piv = np.array([W / 2, by + s * rho0]) if k > 1e-3 else np.array([W / 2, H])
        spin = -260 * (1 - u) * t                                        # slow turn while it is a planet, stops as it unrolls
        x, y = self.warp(k, s, piv, spin)
        x = np.mod(x, self.Wp).astype(np.float32); y = y.astype(np.float32)
        img = cv2.remap(self.pano, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
        outside = (img[..., 0] < 0)[..., None]
        img = np.where(outside, self.paper, img)
        # paper halo round the planet, fading as it unrolls (weight 0.0: no visible halo)
        if u < 1:
            halo = cv2.GaussianBlur(outside.astype(np.float32)[..., 0], (0, 0), 6)[..., None]
            img = img * (1 - 0.0 * halo)
        if t >= self.END - 1e-6: img = self.flat
        return np.clip(img, 0, 1).astype(np.float32)


if __name__ == '__main__':
    import subprocess, sys
    p = Planet(); out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'work/v16k/planet_test.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                            '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in range(int(5.6 * 25) + 1):
        im = p.frame(f / 25); enc.stdin.write((im * 255 + 0.5).astype(np.uint8).tobytes())
        if f in (0, 50, 90, 110, 125, 140): Image.fromarray((im * 255).astype(np.uint8)).save(out.replace('.mp4', f'_{f:03d}.jpg'), quality=88)
    enc.stdin.close(); enc.wait(); print('->', out)
