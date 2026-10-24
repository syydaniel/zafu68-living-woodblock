"""print look for the zafu_v1 bamboo hero shot: a gradient map toward the film's palette (ink-green shadows, mineral green mids, warm paper
highlights) mixed with the desaturated original, plus faint carved ink edges, so the CG bamboo reads as part of the woodblock film"""
import numpy as np, cv2
SH = np.array([0.10, 0.17, 0.14], np.float32); MID = np.array([0.42, 0.57, 0.31], np.float32)
HI = np.array([0.94, 0.90, 0.80], np.float32); GOLD = np.array([0.86, 0.74, 0.46], np.float32)


def look(rgb):
    x = np.clip(rgb.astype(np.float32) * 1.08, 0, 1)
    L = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    t = np.clip(L, 0, 1)
    gm = np.where(t < 0.45, SH + (MID - SH) * (t / 0.45), np.where(t < 0.8, MID + (GOLD - MID) * ((t - 0.45) / 0.35), GOLD + (HI - GOLD) * ((t - 0.8) / 0.2)))
    des = L + (x - L) * 0.78
    out = 0.45 * gm + 0.55 * des
    g = cv2.GaussianBlur(L[..., 0], (0, 0), 1.2)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    ink = np.clip((e - 0.10) / 0.25, 0, 1) * 0.35
    out = out * (1 - ink[..., None]) + SH * ink[..., None]
    return np.clip(out, 0, 1)


if __name__ == '__main__':
    from PIL import Image
    cap = cv2.VideoCapture('~/Movies/ZAFU/renders/v18_bamboo.mov'); ims = []
    for f in (40, 160, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        ims.append(np.hstack([cv2.resize(rgb, (640, 360)), cv2.resize(look(rgb), (640, 360))]))
    Image.fromarray((np.vstack(ims) * 255).astype(np.uint8)).save('work/v18/_bamboo_look.jpg', quality=88)
