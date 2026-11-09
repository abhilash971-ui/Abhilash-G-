"""
Minimal offline SFZ sampler + one-shot sample banks, in pure numpy.

Supports the subset of SFZ used by Sonatina Symphonic Orchestra:
headers <control>/<global>/<master>/<group>/<region>, #include,
key ranges, velocity ranges, round robin (seq_length/seq_position),
CC1 dynamics crossfades (xfin/xfout_locc1/hicc1, gain_cc1, cutoff_cc1),
velocity crossfades, volume/group_volume/master_volume, pan, tune,
transpose, pitch_keycenter, offset(+random), loop_mode/loop_start/loop_end,
ADSR (delay/attack/hold/decay/sustain/release, vel2attack), amp_veltrack,
2-pole / 1-pole filters with veltrack/keytrack and amplitude LFO.
"""
import glob
import os
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100
NOTE_RE = re.compile(r"^([a-gA-G])([#b]?)(-?\d+)$")
OFFS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def note_num(v):
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        m = NOTE_RE.match(v)
        if not m:
            raise
        n = OFFS[m.group(1).lower()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
        return n + (int(m.group(3)) + 1) * 12


@lru_cache(maxsize=None)
def load_sample(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x[:, :2], sr


class SFZ:
    def __init__(self, path, cc1=None):
        self.path = path
        self.root = os.path.dirname(path)
        self.control = {}
        self.regions = []
        self._parse(path)
        self.cc1 = cc1 if cc1 is not None else float(self.control.get("set_cc1", 100))
        self.seq = defaultdict(int)
        self.rng = np.random.default_rng(11)
        for r in self.regions:
            if "key" in r:
                k = note_num(r["key"])
                r.setdefault("lokey", str(k))
                r.setdefault("hikey", str(k))
                r.setdefault("pitch_keycenter", str(k))
            r["_lo"] = note_num(r.get("lokey", "0"))
            r["_hi"] = note_num(r.get("hikey", "127"))
            r["_kc"] = note_num(r.get("pitch_keycenter", "60"))

    # ------------------------------------------------------------ parsing
    def _lines(self, path):
        with open(path, errors="ignore") as f:
            for raw in f:
                line = raw.split("//")[0].strip()
                if not line:
                    continue
                m = re.match(r'#include\s+"(.+)"', line)
                if m:
                    yield from self._lines(os.path.join(self.root, m.group(1)))
                else:
                    yield line

    def _parse(self, path):
        levels = {"global": {}, "master": {}, "group": {}}
        cur, cur_kind = None, None

        def flush():
            if cur_kind == "region" and cur is not None and "sample" in cur:
                r = {}
                for lv in ("global", "master", "group"):
                    r.update(levels[lv])
                r.update(cur)
                self.regions.append(r)

        for line in self._lines(path):
            parts = re.split(r"<(\w+)>", line)
            # parts: [text, hdr, text, hdr, text ...]
            chunks = [(None, parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
            for hdr, text in chunks:
                if hdr:
                    flush()
                    cur_kind = hdr
                    if hdr in ("global", "master", "group"):
                        levels[hdr] = {}
                        if hdr == "global":
                            levels["master"], levels["group"] = {}, {}
                        if hdr == "master":
                            levels["group"] = {}
                        cur = levels[hdr]
                    elif hdr == "region":
                        cur = {}
                    elif hdr == "control":
                        cur = self.control
                    else:
                        cur = {}
                ops = list(re.finditer(r"([A-Za-z0-9_]+)=", text))
                for i, m in enumerate(ops):
                    end = ops[i + 1].start() if i + 1 < len(ops) else len(text)
                    val = text[m.end():end].strip()
                    if cur is not None:
                        cur[m.group(1)] = val
        flush()

    # ------------------------------------------------------------ playing
    @staticmethod
    def _xf(val, lo, hi, fade_in):
        if lo is None:
            return 1.0
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            t = 1.0 if (val >= hi if fade_in else val <= lo) else 0.0
        else:
            t = np.clip((val - lo) / (hi - lo), 0, 1)
            t = t if fade_in else 1 - t
        return float(t)

    def _select(self, key, vel):
        out = []
        for r in self.regions:
            if not (r["_lo"] <= key <= r["_hi"]):
                continue
            if not (int(r.get("lovel", 0)) <= vel <= int(r.get("hivel", 127))):
                continue
            if r.get("trigger", "attack") not in ("attack", "first"):
                continue
            if "sw_last" in r or "sw_lokey" in r:
                if "sw_default" in r and "sw_last" in r and note_num(r["sw_last"]) != note_num(r["sw_default"]):
                    continue
            out.append(r)
        # round robin
        seqs = [r for r in out if "seq_length" in r]
        if seqs:
            L = int(seqs[0]["seq_length"])
            pos = self.seq[key] % L + 1
            self.seq[key] += 1
            out = [r for r in out if "seq_length" not in r or int(r.get("seq_position", 1)) == pos]
        return out

    def render_note(self, key, vel, dur, cc1=None):
        cc = self.cc1 if cc1 is None else cc1
        voices = []
        for r in self._select(key, vel):
            g = self._xf(cc, r.get("xfin_locc1"), r.get("xfin_hicc1"), True) * \
                self._xf(cc, r.get("xfout_locc1"), r.get("xfout_hicc1"), False) * \
                self._xf(vel, r.get("xfin_lovel"), r.get("xfin_hivel"), True) * \
                self._xf(vel, r.get("xfout_lovel"), r.get("xfout_hivel"), False)
            if g <= 1e-4:
                continue
            if r.get("xf_cccurve", "power") == "power":
                g = np.sqrt(g)
            voices.append(self._voice(r, key, vel, dur, cc, g))
        if not voices:
            return None
        L = max(len(v) for v in voices)
        out = np.zeros((L, 2), np.float32)
        for v in voices:
            out[: len(v)] += v
        return out

    def _voice(self, r, key, vel, dur, cc, xfgain):
        f = lambda k, d=0.0: float(r.get(k, d))
        path = os.path.join(self.root, r["sample"].replace("\\", "/"))
        x, ssr = load_sample(path)
        semis = (key - r["_kc"]) * f("pitch_keytrack", 100) / 100 + f("transpose") + (f("tune") + f("group_tune")) / 100
        ratio = 2 ** (semis / 12) * ssr / SR

        db = f("volume") + f("group_volume") + f("master_volume") + f("gain_cc1") * cc / 127
        vt = f("amp_veltrack", 100) / 100
        amp = (1 - vt + vt * (vel / 127) ** 2) * 10 ** (db / 20) * xfgain

        mode = r.get("loop_mode", "no_loop")
        ls, le = r.get("loop_start"), r.get("loop_end")
        looping = mode in ("loop_continuous", "loop_sustain") and ls is not None and le is not None
        offset = f("offset") + (self.rng.uniform(0, f("offset_random")) if "offset_random" in r else 0)
        rel = max(f("ampeg_release", 0.03), 0.03)
        if mode == "one_shot":
            n_out = int((len(x) - offset) / ratio)
        else:
            n_out = int((dur + rel) * SR)
            if not looping:
                n_out = min(n_out, int((len(x) - offset) / ratio))
        if n_out <= 16:
            return np.zeros((1, 2), np.float32)

        pos = offset + np.arange(n_out) * ratio
        if looping:
            ls, le = float(ls), float(le)
            over = pos > le
            pos[over] = ls + np.mod(pos[over] - ls, le - ls)
        pos = np.clip(pos, 0, len(x) - 2)
        i0 = pos.astype(np.int64)
        fr = (pos - i0)[:, None].astype(np.float32)
        y = x[i0] * (1 - fr) + x[i0 + 1] * fr

        # envelope
        t = np.arange(n_out) / SR
        dl = f("ampeg_delay")
        att = max(f("ampeg_attack") + f("ampeg_vel2attack") * vel / 127, 0.002)
        hold = f("ampeg_hold")
        dec = f("ampeg_decay") + f("ampeg_vel2decay") * vel / 127
        sus = f("ampeg_sustain", 100) / 100
        env = np.ones(n_out, np.float32)
        tt = t - dl
        env[tt < 0] = 0
        a = (tt >= 0) & (tt < att)
        env[a] = tt[a] / att
        if dec > 0 and sus < 1:
            d = tt >= att + hold
            env[d] = sus + (1 - sus) * np.exp(-(tt[d] - att - hold) / (dec / 5))
        if mode != "one_shot":
            off = t >= dur
            if off.any():
                lvl = env[min(int(dur * SR), n_out - 1)]
                env[off] = lvl * np.exp(-(t[off] - dur) / (rel / 4))
        env[-64:] *= np.linspace(1, 0, 64)
        y = y * env[:, None]

        if "amplfo_depth" in r:
            dep = f("amplfo_depth")
            fl = f("amplfo_freq", 5)
            ramp = np.clip((t - f("amplfo_delay")) / max(f("amplfo_fade"), 1e-3), 0, 1)
            y = y * (10 ** (dep * np.sin(2 * np.pi * fl * t) * ramp / 20))[:, None]

        ft = r.get("fil_type")
        if ft and "cutoff" in r:
            cents = f("fil_veltrack") * vel / 127 + f("fil_keytrack") * (key - note_num(r.get("fil_keycenter", "60"))) \
                + f("cutoff_cc1") * cc / 127
            fc = float(np.clip(f("cutoff") * 2 ** (cents / 1200), 30, SR / 2 - 200))
            btype = "high" if ft.startswith("hpf") else "low"
            order = 1 if "1p" in ft else 2
            if not (btype == "low" and fc > 18000):
                y = sosfilt(butter(order, fc, btype=btype, fs=SR, output="sos"), y, axis=0)

        p = f("pan") / 100
        y = y * np.array([min(1, 1 - p), min(1, 1 + p)], np.float32)
        return (y * amp).astype(np.float32)


class Bank:
    """One-shot sample bank: a list of velocity layers, each a list of round-robins."""

    def __init__(self, layers):
        self.layers = [[f for f in l if os.path.exists(f)] for l in layers]
        self.layers = [l for l in self.layers if l]
        self.rr = 0

    @classmethod
    def from_globs(cls, patterns):
        return cls([sorted(glob.glob(p)) for p in patterns])

    def render(self, vel, semis=0.0):
        li = min(int(vel / 128 * len(self.layers)), len(self.layers) - 1)
        files = self.layers[li]
        path = files[self.rr % len(files)]
        self.rr += 1
        x, ssr = load_sample(path)
        ratio = 2 ** (semis / 12) * ssr / SR
        if abs(ratio - 1) < 1e-6:
            y = x.copy()
        else:
            n = int(len(x) / ratio) - 2
            pos = np.arange(n) * ratio
            i0 = pos.astype(np.int64)
            fr = (pos - i0)[:, None].astype(np.float32)
            y = x[i0] * (1 - fr) + x[i0 + 1] * fr
        # within-layer velocity scaling
        return y * (0.55 + 0.45 * vel / 127)
