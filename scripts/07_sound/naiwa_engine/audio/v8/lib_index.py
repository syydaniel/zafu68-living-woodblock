"""CLAP index of the whole local sound library: every file of the Final Cut Pro Sound Effects and the
Fairlight Sound Library gets one embedding (mean over up to three 10 s windows: start, middle, end), so sounds can be
searched by what they sound like:   find("small pebbles plop into water", 8)
Build:  HF_HUB_DISABLE_XET=1 ~/ACE-Step-1.5/.venv/bin/python audio/v8/lib_index.py build
Add:    ... lib_index.py add <folder> ...                     (append more folders, see add())
Query:  ... lib_index.py q "text one" "text two" ...        (top 8 each)
Files: audio/v8/lib_index.npz (emb, float16) + lib_index.json (path, dur, ch, sr, rms)."""
import os, sys, json, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOTS = ["/Library/Audio/Apple Loops/Apple/Final Cut Pro Sound Effects", "~/Movies/Fairlight Sound Library/Media Files"]
EXT = (".caf", ".wav", ".aif", ".aiff", ".m4a", ".mp3", ".flac")


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels,sample_rate:format=duration", "-of", "json", p], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout); return float(j["format"]["duration"]), int(j["streams"][0]["channels"]), int(j["streams"][0]["sample_rate"])
    except Exception:
        return None


def dec(p, a, d):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", p, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def build():
    from clap_tool import Clap
    c = Clap(); files = []
    for rt in ROOTS:
        for dp, dn, fn in os.walk(rt):
            for f in sorted(fn):
                if f.lower().endswith(EXT) and not f.startswith("."):
                    files.append(os.path.join(dp, f))
    meta = []; embs = []; batch = []; owner = []
    def flush():
        if not batch: return
        e = c.embed_audio(batch)
        for k, i in enumerate(owner):
            embs[i].append(e[k])
        batch.clear(); owner.clear()
    for n, p in enumerate(files):
        pr = probe(p)
        if pr is None or pr[0] < 0.05:
            continue
        dur, ch, sr = pr; wins = [(0.0, min(10.0, dur))]
        if dur > 25: wins.append((dur / 2 - 5, 10.0))
        if dur > 45: wins.append((dur - 12, 10.0))
        i = len(meta); xs = [dec(p, a, d) for a, d in wins]; xs = [x for x in xs if len(x) > 2000]
        if not xs: continue
        meta.append(dict(path=p, name=os.path.splitext(os.path.basename(p))[0], dur=round(dur, 2), ch=ch, sr=sr, rms=round(float(20 * np.log10(np.sqrt(np.mean(np.concatenate(xs) ** 2)) + 1e-9)), 1),
                         cat=os.path.basename(os.path.dirname(p)), lib="FCP" if "Final Cut" in p else "FL")); embs.append([])
        for x in xs:
            batch.append(x); owner.append(i)
        if len(batch) >= 16: flush()
        if n % 200 == 0: print(n, "/", len(files), flush=True)
    flush()
    E = np.stack([np.mean(e, 0) / (np.linalg.norm(np.mean(e, 0)) + 1e-9) for e in embs]).astype(np.float16)
    np.savez(f"{HERE}/lib_index.npz", emb=E); json.dump(meta, open(f"{HERE}/lib_index.json", "w"), ensure_ascii=False)
    print("INDEX DONE", len(meta), "files", flush=True)


_IDX = {}
def find(text, n=8, c=None, maxdur=None, mindur=None, lib=None, cat=None):
    if "E" not in _IDX:
        _IDX["E"] = np.load(f"{HERE}/lib_index.npz")["emb"].astype(np.float32); _IDX["M"] = json.load(open(f"{HERE}/lib_index.json"))
    if c is None:
        if "c" not in _IDX:
            from clap_tool import Clap
            _IDX["c"] = Clap()
        c = _IDX["c"]
    t = c.embed_text([text])[0]; s = _IDX["E"] @ t; out = []
    for i in np.argsort(-s):
        m = _IDX["M"][i]
        if maxdur and m["dur"] > maxdur: continue
        if mindur and m["dur"] < mindur: continue
        if lib and m["lib"] != lib: continue
        if cat and cat.lower() not in m["cat"].lower(): continue
        out.append((round(float(s[i]), 3), m["name"], m["dur"], m["ch"], m["cat"], m["lib"]))
        if len(out) >= n: break
    return out


def add(roots, lib="SON"):
    """index more folders (e.g. the Sonniss GDC 2026 bundle) and APPEND them to lib_index.json / .npz (files already in the index are skipped)"""
    from clap_tool import Clap
    E0 = np.load(f"{HERE}/lib_index.npz")["emb"]; M0 = json.load(open(f"{HERE}/lib_index.json")); have = {m["path"] for m in M0}
    c = Clap(); files = []
    for rt in roots:
        for dp, dn, fn in os.walk(rt):
            for f in sorted(fn):
                if f.lower().endswith(EXT) and not f.startswith(".") and os.path.join(dp, f) not in have:
                    files.append(os.path.join(dp, f))
    meta = []; embs = []
    for n, p in enumerate(files):
        pr = probe(p)
        if pr is None or pr[0] < 0.05: continue
        dur, ch, sr = pr; wins = [(0.0, min(10.0, dur))]
        if dur > 25: wins.append((dur / 2 - 5, 10.0))
        if dur > 45: wins.append((dur - 12, 10.0))
        xs = [dec(p, a_, d_) for a_, d_ in wins]; xs = [x for x in xs if len(x) > 2000]
        if not xs: continue
        e = c.embed_audio(xs); v = np.mean(e, 0); v = v / (np.linalg.norm(v) + 1e-9)
        meta.append(dict(path=p, name=os.path.splitext(os.path.basename(p))[0], dur=round(dur, 2), ch=ch, sr=sr, rms=round(float(20 * np.log10(np.sqrt(np.mean(np.concatenate(xs) ** 2)) + 1e-9)), 1),
                         cat=os.path.basename(os.path.dirname(p)), lib=lib)); embs.append(v)
        if n % 20 == 0: print(n, "/", len(files), flush=True)
    if meta:
        E = np.concatenate([E0, np.stack(embs).astype(np.float16)]); np.savez(f"{HERE}/lib_index.npz", emb=E); json.dump(M0 + meta, open(f"{HERE}/lib_index.json", "w"), ensure_ascii=False)
    print("ADD DONE", len(meta), "files added,", len(M0) + len(meta), "in all", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "add":
        add(sys.argv[2:])
    else:
        for q in sys.argv[2:]:
            print("Q:", q)
            for r in find(q, 8): print("   ", r)
