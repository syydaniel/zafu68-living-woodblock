"""Text textures: R = coverage, G = signed distance (0.5 at the edge, inside > 0.5), B = reveal-time field (0..1).
Time fields: 'lr' left->right, 'tb' top->bottom, 'glyph' one glyph after another, 'brush' ink flowing through each
stroke group (geodesic distance inside the glyph from its top-left start, groups in reading order)."""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).parent
CACHE = ROOT / 'assets' / 'text'
AD = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/'
FONTS = {
    'song_light': ('/System/Library/Fonts/Supplemental/Songti.ttc', 3),
    'song': ('/System/Library/Fonts/Supplemental/Songti.ttc', 6),
    'song_bold': ('/System/Library/Fonts/Supplemental/Songti.ttc', 1),
    'song_black': ('/System/Library/Fonts/Supplemental/Songti.ttc', 0),
    'xingkai': (AD + '13b8ce423f920875b28b551f9406bf1014e0a656.asset/AssetData/Xingkai.ttc', 0),
    'kai': (AD + '88d6cc32a907955efa1d014207889413890573be.asset/AssetData/Kaiti.ttc', 0),
    'optima': ('/System/Library/Fonts/Optima.ttc', 0),
    'optima_bold': ('/System/Library/Fonts/Optima.ttc', 1),
    'avenir_light': ('/System/Library/Fonts/Avenir Next.ttc', 11),
    'didot': ('/System/Library/Fonts/Supplemental/Didot.ttc', 0),
    'didot_italic': ('/System/Library/Fonts/Supplemental/Didot.ttc', 1),
}


def font(name, px):
    path, idx = FONTS[name]
    return ImageFont.truetype(path, px, index=idx)


def _layout(text, fnt, px, vertical, tracking, line_gap):
    """returns canvas size and a list of (char, x, y) top-left positions, glyph boxes"""
    lines = text.split('\n')
    asc, desc = fnt.getmetrics()
    glyphs = []
    if not vertical:
        y = 0
        width = 0
        for li, line in enumerate(lines):
            x = 0
            for ch in line:
                w = fnt.getlength(ch)
                glyphs.append((ch, x, y, w))
                x += w + tracking * px
            width = max(width, x - tracking * px)
            y += (asc + desc) * (1 + line_gap)
        height = y - (asc + desc) * line_gap
        return int(np.ceil(width)), int(np.ceil(height)), glyphs
    # vertical: columns right to left, glyphs centred in a px-wide column
    col_w = px * (1 + line_gap)
    step = px * (1 + tracking)
    ncols = len(lines)
    height = max(len(l) for l in lines) * step - tracking * px
    width = ncols * col_w - line_gap * px
    for ci, line in enumerate(lines):
        x0 = width - (ci + 1) * col_w + line_gap * px
        for k, ch in enumerate(line):
            w = fnt.getlength(ch)
            glyphs.append((ch, x0 + (px - w) / 2, k * step - (asc - px * 0.88), w))
    return int(np.ceil(width)), int(np.ceil(height)), glyphs


def make_text(text, fontname='song', px=300, vertical=False, tracking=0.05, line_gap=0.4, field='lr', spread=None, pad=None):
    key = hashlib.md5(json.dumps([text, fontname, px, vertical, tracking, line_gap, field, spread, pad, 4]).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f'{key}.npy'
    if cf.exists():
        return np.load(cf)
    fnt = font(fontname, px)
    spread = spread or max(8, px // 10)
    pad = pad or int(spread * 1.6)
    W, H, glyphs = _layout(text, fnt, px, vertical, tracking, line_gap)
    W += 2 * pad; H += 2 * pad
    asc, desc = fnt.getmetrics()
    cov = np.zeros((H, W), np.float32)
    gid = np.full((H, W), -1, np.int32)
    for i, (ch, x, y, w) in enumerate(glyphs):
        if ch.strip() == '':
            continue
        im = Image.new('L', (W, H), 0)
        ImageDraw.Draw(im).text((x + pad, y + pad), ch, font=fnt, fill=255)
        a = np.asarray(im, np.float32) / 255.
        gid[(a > 0.5) & (gid < 0)] = i
        cov = np.maximum(cov, a)
    inside = cov > 0.5
    d_out = ndimage.distance_transform_edt(~inside)
    d_in = ndimage.distance_transform_edt(inside)
    sd = d_in - d_out
    g = np.clip(0.5 + sd / (2 * spread), 0, 1)
    # time field
    ys, xs = np.nonzero(inside)
    tf = np.zeros((H, W), np.float32)
    if len(xs):
        if field == 'lr':
            tf[inside] = (xs - xs.min()) / max(1, xs.max() - xs.min())
        elif field == 'tb':
            tf[inside] = (ys - ys.min()) / max(1, ys.max() - ys.min())
        else:
            order = [i for i, gl in enumerate(glyphs) if gl[0].strip()]
            n = len(order)
            for k, i in enumerate(order):
                m = gid == i
                if not m.any():
                    continue
                t0, t1 = k / n, (k + 1) / n
                if field == 'glyph':
                    gy, gx = np.nonzero(m)
                    loc = ((gx - gx.min()) / max(1, gx.max() - gx.min()) * 0.5 + (gy - gy.min()) / max(1, gy.max() - gy.min()) * 0.5)
                    tf[m] = t0 + (t1 - t0) * loc
                else:  # brush
                    from skimage.graph import MCP_Geometric
                    lab, nl = ndimage.label(m)
                    comps = []
                    for c in range(1, nl + 1):
                        cy, cx = np.nonzero(lab == c)
                        if len(cy) < 30:
                            continue
                        # reading order: top first, then right-to-left like a brush would come in from the upper left
                        comps.append((cy.min() + 0.25 * cx.min(), c, cy, cx))
                    comps.sort()
                    nc = max(1, len(comps))
                    sizes = np.array([len(c[2]) for c in comps], np.float64)
                    wts = np.sqrt(sizes) / np.sqrt(sizes).sum() if len(comps) else np.array([1.])
                    acc = t0
                    for (key_, c, cy, cx), wgt in zip(comps, wts):
                        span = (t1 - t0) * wgt
                        costs = np.where(lab == c, 1.0, 1e6)
                        start = np.argmin(cy + 0.6 * cx)
                        mcp = MCP_Geometric(costs)
                        cum, _ = mcp.find_costs([(cy[start], cx[start])])
                        v = cum[cy, cx]
                        v = v / max(1e-6, v.max())
                        tf[cy, cx] = acc + span * v
                        acc += span
                    # tiny leftovers
                    left = m & (tf == 0)
                    tf[left] = t1
    # propagate the time field outward so glows reveal with their glyph
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    tf = tf[iy, ix]
    out = np.stack([cov, g.astype(np.float32), tf.astype(np.float32), np.ones_like(cov)], -1)
    np.save(cf, out)
    return out


if __name__ == '__main__':
    import sys
    a = make_text(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'song', 300, field=sys.argv[3] if len(sys.argv) > 3 else 'lr')
    Image.fromarray((a[..., :3] * 255).astype(np.uint8)).save(ROOT / 'frames' / 'text_test.png')
    print(a.shape)
