"""Smooth camera paths: Catmull-Rom through keys (time, eye, target, fov, aperture), eased time mapping per span."""
import numpy as np


def _cr(p0, p1, p2, p3, x):
    x2, x3 = x * x, x * x * x
    return 0.5 * ((2 * p1) + (-p0 + p2) * x + (2 * p0 - 5 * p1 + 4 * p2 - p3) * x2 + (-p0 + 3 * p1 - 3 * p2 + p3) * x3)


class Path:
    def __init__(self, keys, ease_ends=True):
        self.t = np.array([k[0] for k in keys], float)
        self.v = [np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in k[1:]]) for k in keys]
        self.ease = ease_ends

    def __call__(self, t):
        T, V = self.t, self.v
        if t <= T[0]:
            return V[0]
        if t >= T[-1]:
            return V[-1]
        # global ease-in/out on the whole path time so the move starts and ends softly
        i = int(np.searchsorted(T, t) - 1)
        x = (t - T[i]) / (T[i + 1] - T[i])
        if self.ease:
            if i == 0 and i == len(T) - 2:
                x = x * x * (3 - 2 * x)
            elif i == 0:
                x = x * x * (1.5 - 0.5 * x)                        # ease in, leaves with a matching slope
            elif i == len(T) - 2:
                x = 1 - (1 - x) * (1 - x) * (1.5 - 0.5 * (1 - x))  # ease out
        p0 = V[max(0, i - 1)]; p1 = V[i]; p2 = V[i + 1]; p3 = V[min(len(V) - 1, i + 2)]
        return _cr(p0, p1, p2, p3, x)


def cam_uniforms(path, t):
    v = path(t)
    eye, tgt = v[0:3], v[3:6]
    fov = v[6] if len(v) > 6 else 45.0
    ap = v[7] if len(v) > 7 else 0.0
    return dict(uEye=tuple(float(x) for x in eye), uTarget=tuple(float(x) for x in tgt), uFov=float(fov), uAperture=float(ap),
                uFocus=float(np.linalg.norm(np.asarray(tgt) - np.asarray(eye))))
