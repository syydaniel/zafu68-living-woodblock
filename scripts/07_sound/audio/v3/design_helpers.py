def one(name, t, a=0.0, b=None, peak=None, act=None, sp=1.0, lp=None, hp=None, x=0.0, az=None, dist=2.0, rv=-14.0, sync="onset", fi=0.003, fo=0.05,
        note="", stem="FX", lfe=None, room="auto", clip=None, doppler=False, tilt=None):
    """one sound from the library (or a prepared clip), its attack on film time t, at screen position x (-1 .. 1) or
    azimuth az (may be a path [(seconds from the clip start, az, dist), ...])"""
    if t is None:
        return None
    c = clip if clip is not None else S(name, a, b)
    c = speed(c, sp)
    if hp: c = hpf(c, hp)
    if lp: c = lpf(c, lp)
    if tilt: c = shelf_hi(c, 2500, tilt)
    c = fade(c, fi, fo)
    if peak is not None: c = to_peak(c, peak)
    elif act is not None: c = to_act(c, act)
    off = onset(c) if sync == "onset" else (float(sync) if sync else 0.0)
    M.add(c, t - off, stem=stem, az=(az if az is not None else x * SX), dist=dist, rv=rv, lfe=lfe, room=room, name=name or note, note=note, doppler=doppler)
    if note is not None: cue(t, stem, name or "synth", note)
    return c


def film_time(x, t):
    """the recording is read at the film's clock, so the next shot of the same place goes on where this
    one stopped: one continuous air under the cuts, only its level and colour change"""
    return np.roll(x, -(int(max(0.0, t) * SR) % len(x)), axis=0) if len(x) > SR else x


def bed(name, t0, t1, level=None, a=0.0, lp=None, hp=None, fi=0.5, fo=0.5, pre=0.25, post=0.25, pts=None, width=None, az=0.0, rear=-4.0, stem="AMB", surround=True,
        rv=None, room="auto", clip=None, note=None):
    """a bed between two film times from a library recording (stereo, or 5.1 when the file is). level = dB of its
    active level; pts = [(film time, dB), ...] an absolute level curve instead. Stereo recordings get a second,
    different stretch of themselves behind the listener in 5.1 (rear dB, None = no)."""
    n = int((t1 - t0 + pre + post) * SR)
    if n <= 0: return
    x = clip if clip is not None else film_time(S(name, a, None, mono=False, surround=surround), t0 - pre)
    x = loop_to(x, n)
    if hp: x = hpf(x, hp)
    if lp: x = lpf(x, lp)
    ref = lev(x)
    g = db((level if level is not None else 0.0) - ref)
    x = x * g
    if pts: x = x * env(t0 - pre, n, pts)[:, None]
    x = tame(x, (max(p[1] for p in pts) if pts else level) + 13.0)
    x = fade(x, fi, fo)
    M.add(x, t0 - pre, stem=stem, az=az, width=width, rv=rv, room=room, name=name)
    if rear is not None and x.shape[1] == 2 and clip is None:
        y = S(name, 0.0, None, mono=False, surround=False); k = len(y) // 2; y = loop_to(film_time(np.concatenate([y[k:], y[:k]])[:, ::-1], t0 - pre), n)
        if hp: y = hpf(y, hp)
        y = lpf(y, min(lp or 6000, 6000)) * g * db(rear)
        if pts: y = y * env(t0 - pre, n, pts)[:, None]
        M.add(fade(tame(y, (max(p[1] for p in pts) if pts else level) + 13.0 + rear), fi, fo), t0 - pre, stem=stem, az=180.0, width=70.0, rv=None, name=name + " (rear)", only="5.1")
    if note: cue(t0, stem, name, note)


def steps(name, t0, t1, rate, peak, x0=0.0, x1=None, d0=3.0, d1=None, sp=1.0, lp=None, hp=None, jit=0.03, k0=0, rv=-14.0, times=None, note="footsteps", thr=-32.0):
    """footsteps from the takes inside a library file; times = explicit contact times instead of a rate"""
    if t0 is None: return
    sg = segs(name, thr=thr, maxlen=0.6, minlen=0.06); assert sg, name
    x1 = x0 if x1 is None else x1; d1 = d0 if d1 is None else d1; tt = []
    if times is None:
        t = t0; k = k0
        while t < t1:
            tt.append(t); t += (1.0 / rate) * (1 + jit * (((k * 53) % 11) / 5 - 1)); k += 1
    else:
        tt = list(times)
    for k, t in enumerate(tt):
        u = (t - t0) / max(1e-6, t1 - t0); a, b = sg[((k + k0) * 5 + 1) % len(sg)]
        one(name, t, a=a, b=b, peak=peak - (0 if k % 2 == 0 else 1.5), sp=sp * (0.97 + 0.06 * ((k * 37) % 10) / 10), lp=lp, hp=hp, x=x0 + (x1 - x0) * u + 0.03 * (1 if k % 2 else -1),
            dist=d0 + (d1 - d0) * u, rv=rv, fo=0.04, note=None)
    cue(tt[0], "FX", name, f"{note}: {len(tt)} steps to {tt[-1]:.2f}")


NOTE = dict(As5=932.33, F6=1396.91, As6=1864.66, D5=587.33, E5=659.26, Fs5=739.99, G5=783.99, A5=880.0, B5=987.77, Cs6=1108.73, D6=1174.66, E6=1318.51, Fs6=1479.98, G6=1567.98, A6=1760.0, B6=1975.53, D7=2349.32, Gs5=830.61)


def bell(f, dur=2.4, bright=0.5, attack=0.012):
    """a breath of light rather than a struck (marimba-like) note: the note and its octave, each doubled a few cents apart so that
    it shimmers slowly, a little air round it; it swells in (never faster than 0.2 s) and dies away"""
    n = int(dur * SR); t = np.arange(n) / SR; att = max(0.20, attack, 0.16 * dur); x = np.zeros(n)
    for m, am in ((1.0, 1.0), (2.0, 0.10 + 0.28 * bright), (3.0, 0.08 * bright)):
        for det in (-1.0, 1.0): x += 0.5 * am * np.sin(2 * np.pi * f * m * (1 + det * 0.0017) * t + rng.uniform(0, 6.283))
    air = bpf(rng.standard_normal(n).astype(np.float32), f * 1.9, min(f * 2.4, 16000.0)); x += air / (np.abs(air).max() + 1e-9) * 0.08 * bright
    e = np.sin(np.minimum(1.0, t / att) * np.pi / 2) ** 2 * np.exp(-np.maximum(0.0, t - att) * 5.0 / dur)
    return fade((x * e).astype(np.float32), 0.0, 0.4)


