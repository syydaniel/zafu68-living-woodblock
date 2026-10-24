"""An "ear" for the sound work: LAION CLAP (larger_clap_general) scores how well a piece of audio matches
text descriptions. Used to pick library samples by what they sound like (not by file name), to check that each placed
effect reads as what it should be, and to compare music cues against mood descriptions.
Run with a Python that has torch + transformers (here the ACE-Step 1.5 environment, ~/ACE-Step-1.5/.venv/bin/python;
weights in the Hugging Face cache).
  from clap_tool import Clap; c = Clap(); c.score(wav_48k_array_or_path, ["footsteps on dry dirt", "glass breaking"]) -> probabilities
  c.embed_audio(list of arrays) / c.embed_text(list of str) -> unit vectors (cosine similarity = dot product)"""
import os, numpy as np, torch, soundfile as sf
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import ClapModel, ClapProcessor
from scipy.signal import resample_poly


def load48(p, a=0.0, b=None):
    info = sf.info(p); i = int(a * info.samplerate); j = info.frames if b is None else int(b * info.samplerate)
    x, sr = sf.read(p, start=i, stop=j, dtype="float32", always_2d=True); x = x.mean(1)
    if sr != 48000:
        x = resample_poly(x, 48000, sr).astype(np.float32)
    return x


class Clap:
    def __init__(self, name="laion/larger_clap_general", device=None):
        self.dev = device or os.environ.get("CLAP_DEV") or ("mps" if torch.backends.mps.is_available() else "cpu")      # CLAP_DEV=cpu leaves the GPU to other jobs (e.g. a Blender render)
        self.m = ClapModel.from_pretrained(name).to(self.dev).eval(); self.p = ClapProcessor.from_pretrained(name)

    @torch.no_grad()
    def embed_audio(self, clips):
        out = []
        for k in range(0, len(clips), 8):
            xs = [load48(c) if isinstance(c, str) else np.asarray(c, np.float32) for c in clips[k:k + 8]]
            xs = [x if len(x) >= 48000 else np.tile(x, int(np.ceil(48000 / max(1, len(x)))))[:48000] for x in xs]
            inp = self.p(audio=xs, sampling_rate=48000, return_tensors="pt", padding=True)
            e = self.m.get_audio_features(**{k_: v.to(self.dev) for k_, v in inp.items()})
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed_text(self, texts):
        inp = self.p(text=list(texts), return_tensors="pt", padding=True)
        e = self.m.get_text_features(**{k_: v.to(self.dev) for k_, v in inp.items()})
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def score(self, clip, texts, temp=20.0):
        a = self.embed_audio([clip])[0]; t = self.embed_text(texts); s = t @ a; p = np.exp(temp * (s - s.max())); p /= p.sum()
        return [(texts[i], float(s[i]), float(p[i])) for i in np.argsort(-s)]


if __name__ == "__main__":
    c = Clap(); L = "~/Movies/NAIWA_BEYOND/audio/v7/lib"
    T = ["wind blowing", "rain falling", "a stream of running water", "crickets at night", "footsteps on dirt", "glass breaking", "an explosion", "a jet engine", "orchestral music", "a person speaking", "a bell ringing"]
    for n in ("Wind_2", "Rain_1", "Water_Stream_River_01", "Crickets_FX_01", "Footsteps_Sneaker_Dirt_Walk_1", "FL_Glass_Break_1", "Explosion_2", "Turbine", "Bell_Small"):
        r = c.score(f"{L}/{n}.wav", T); print(n, "->", [(t, round(s, 2)) for t, s, _ in r[:3]])
