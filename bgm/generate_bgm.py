#!/usr/bin/env python3
"""
"RAAVAN RAGE" - an original, aggressive mass-hero BGM.

Style reference: Tamil mass anthems (e.g. "Raavana Mavandaa", Jana Nayagan).
Everything here is synthesized from scratch: no samples, no melody taken from
the reference. Only the general flavour is borrowed: thappu/parai-style
war drums, brass stabs, gang "HEY!" chants, a reedy nadaswaram-like lead and a
dark Phrygian-dominant mode. It is pushed harder than the reference with a faster
tempo, distorted sub bass, double-time drums and heavier impacts.

Usage:  python3 generate_bgm.py [out.wav]
Needs:  numpy, scipy
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
BPM = 158
BEAT = 60.0 / BPM
STEP = BEAT / 4          # 16th note
BAR = BEAT * 4
rng = np.random.default_rng(7)

# D Phrygian dominant: D Eb F# G A Bb C
ROOT = 62  # D4 (MIDI)
SCALE = [0, 1, 4, 5, 7, 8, 10]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def t_arr(dur):
    return np.arange(int(dur * SR)) / SR


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype="high", fs=SR, output="sos"), x)


def saw(freq, t, phase=0.0):
    return 2 * ((freq * t + phase) % 1.0) - 1


# ----------------------------------------------------------------- instruments
def kick(vel=1.0):
    t = t_arr(0.45)
    f = 45 + 140 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 6.5)
    click = hp(rng.standard_normal(len(t)), 2000) * np.exp(-t * 300) * 0.4
    return np.tanh(2.6 * (body + click)) * vel


def thappu(vel=1.0, pitch=1.0):
    """Frame-drum slap: tonal skin + band-passed noise crack."""
    t = t_arr(0.28)
    skin = np.sin(2 * np.pi * 210 * pitch * t * (1 + 0.25 * np.exp(-t * 40))) * np.exp(-t * 22)
    crack = bp(rng.standard_normal(len(t)), 350 * pitch, 2600 * pitch) * np.exp(-t * 35)
    return np.tanh(1.8 * (0.8 * skin + 1.1 * crack)) * vel


def snare(vel=1.0):
    t = t_arr(0.35)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    noise = bp(rng.standard_normal(len(t)), 900, 9000) * np.exp(-t * 14)
    return np.tanh(2.0 * (0.6 * tone + noise)) * vel


def hat(vel=1.0, open_=False):
    t = t_arr(0.25 if open_ else 0.06)
    n = hp(rng.standard_normal(len(t)), 7000, 4)
    return n * np.exp(-t * (12 if open_ else 70)) * vel * 0.5


def taiko(vel=1.0):
    t = t_arr(1.2)
    f = 55 + 60 * np.exp(-t * 10)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3)
    slap = lp(rng.standard_normal(len(t)), 900) * np.exp(-t * 20) * 0.7
    return np.tanh(2.2 * (body + slap)) * vel


def bass_note(midi, dur, vel=1.0):
    t = t_arr(dur)
    f = hz(midi)
    x = saw(f, t) + saw(f * 1.006, t, 0.3) + 0.8 * np.sin(2 * np.pi * f / 2 * t)
    env = np.minimum(1, t / 0.005) * np.exp(-t * 2.0)
    env *= np.minimum(1, (dur - t) / 0.01).clip(0, 1)
    x = np.tanh(3.5 * x) * env
    return lp(x, 900) * vel


def brass(midis, dur, vel=1.0):
    t = t_arr(dur)
    x = np.zeros_like(t)
    for m in midis:
        f = hz(m)
        for d in (-0.08, 0.0, 0.08):          # 3 detuned saws per voice
            x += saw(f * 2 ** (d / 12), t, rng.random())
    x /= 3 * len(midis)
    # filter-envelope "blat"
    cutoff_env = 600 + 3800 * np.exp(-t * 7)
    out = np.zeros_like(x)
    chunk = 512
    for i in range(0, len(x), chunk):
        c = float(cutoff_env[i])
        out[i:i + chunk] = lp(x[i:i + chunk + 0], min(c, SR / 2 - 100))
    env = np.minimum(1, t / 0.012) * (0.55 + 0.45 * np.exp(-t * 5))
    env *= np.minimum(1, (dur - t) / 0.03).clip(0, 1)
    return np.tanh(2.0 * out * env) * vel


def reed_lead(notes, vel=1.0):
    """Nadaswaram-ish lead with glides and vibrato. notes: [(start_s, dur_s, midi)]."""
    if not notes:
        return np.zeros(1)
    total = notes[-1][0] + notes[-1][1] + 0.3
    n = int(total * SR)
    pitch = np.full(n, hz(notes[0][2]))
    amp = np.zeros(n)
    for st, du, m in notes:
        a, b = int(st * SR), min(n, int((st + du) * SR))
        target = hz(m)
        prev = pitch[a - 1] if a > 0 else target
        glide = min(int(0.04 * SR), b - a)
        pitch[a:b] = target
        pitch[a:a + glide] = np.linspace(prev, target, glide)
        seg = np.arange(b - a) / SR
        amp[a:b] = np.minimum(1, seg / 0.015) * np.minimum(1, (du - seg) / 0.03).clip(0, 1)
    t = np.arange(n) / SR
    pitch *= 1 + 0.012 * np.sin(2 * np.pi * 6.2 * t) * np.minimum(1, t / 0.3)
    ph = 2 * np.pi * np.cumsum(pitch) / SR
    x = np.sign(np.sin(ph)) * 0.5 + np.sin(2 * ph) * 0.4 + np.sin(3 * ph) * 0.35 + np.sin(5 * ph) * 0.2
    x = bp(x, 400, 5000)
    return np.tanh(1.8 * x * amp) * vel


def hey_chant(vel=1.0):
    """Gang shout 'HEY!' - formant-filtered buzz, several voices."""
    t = t_arr(0.38)
    out = np.zeros_like(t)
    for v in range(6):
        f0 = rng.uniform(150, 230)
        f = f0 * (1 + 0.15 * np.exp(-t * 12)) * (1 - 0.06 * t)
        src = 2 * ((np.cumsum(f) / SR) % 1) - 1
        src += 0.3 * rng.standard_normal(len(t))
        voiced = bp(src, 450, 650) * 1.0 + bp(src, 1700, 2100) * 0.8 + bp(src, 2500, 2900) * 0.4
        d = int(rng.uniform(0, 0.015) * SR)
        out[d:] += voiced[:len(t) - d]
    env = np.minimum(1, t / 0.01) * np.exp(-t * 7)
    asp = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t * 60) * 0.4   # the "H"
    return np.tanh(2.5 * (out / 3 * env + asp)) * vel


def riser(dur):
    t = t_arr(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = int(0.05 * SR)
    for i in range(0, len(t), seg):
        c = 300 + (9000 - 300) * (i / len(t)) ** 2
        out[i:i + seg] = bp(n[i:i + seg], c * 0.7, min(c * 1.3, SR / 2 - 100), 1)
    tone_f = 80 * 2 ** (3 * t / dur)
    tone = saw(1, np.cumsum(tone_f) / SR) * 0.3
    return (out + tone) * (t / dur) ** 2


def impact():
    t = t_arr(2.5)
    boom = np.sin(2 * np.pi * np.cumsum(30 + 90 * np.exp(-t * 6)) / SR) * np.exp(-t * 1.6)
    crash = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 2.2) * 0.35
    return np.tanh(2.5 * boom) + crash


def drone(dur):
    t = t_arr(dur)
    f = hz(ROOT - 24)
    x = saw(f, t) + saw(f * 1.5, t, 0.2) * 0.6 + saw(f * 2.003, t, 0.5) * 0.5
    x = lp(x, 500)
    return x * np.minimum(1, t / 2.0) * 0.5


# ---------------------------------------------------------------------- mixer
class Track:
    def __init__(self, length_s):
        self.buf = np.zeros(int(length_s * SR) + SR * 3)

    def add(self, sig, at_s, gain=1.0):
        a = int(at_s * SR)
        b = min(len(self.buf), a + len(sig))
        self.buf[a:b] += sig[:b - a] * gain


def reverb(x, secs=1.6, wet=0.25):
    t = t_arr(secs)
    ir = rng.standard_normal(len(t)) * np.exp(-t * 4.0)
    ir = lp(ir, 6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    return x + wet * fftconvolve(x, ir)[:len(x)]


# ----------------------------------------------------------------- arrangement
# Sections in bars
INTRO, BUILD, DROP_A, BREAK, DROP_B, OUTRO = 8, 8, 16, 4, 16, 4
TOTAL_BARS = INTRO + BUILD + DROP_A + BREAK + DROP_B + OUTRO
LENGTH = TOTAL_BARS * BAR + 3

drums = Track(LENGTH)
perc = Track(LENGTH)
bass = Track(LENGTH)
horns = Track(LENGTH)
lead = Track(LENGTH)
fx = Track(LENGTH)
vox = Track(LENGTH)

# Thappu pattern (16ths) - a 3+3+2 / 3+3+3+3+... driving feel with accents
THAPPU = [1.0, 0, 0.5, 0.9, 0, 0.5, 0.9, 0, 1.0, 0, 0.5, 0.9, 0, 0.6, 0.9, 0.7]
THAPPU_FILL = [1.0, 0.6, 0.8, 0.6, 1.0, 0.6, 0.8, 0.6, 1.0, 0.7, 0.9, 0.7, 1.0, 0.8, 1.0, 1.0]

# Bass progression (one root per bar, semitones from D): i - bII - i - bVII  (dark, villainous)
PROG = [0, 1, 0, -2, 0, 1, 5, 4]

# Original 2-bar brass riff on 16th grid: (step, len_steps, scale-ish semitone)
RIFF = [(0, 2, 0), (2, 2, 0), (4, 1, 1), (5, 3, 4), (8, 2, 5), (10, 2, 4), (12, 2, 1), (14, 2, 0),
        (16, 3, 0), (19, 1, 7), (20, 2, 8), (22, 2, 7), (24, 2, 4), (26, 2, 5), (28, 4, 1)]

# Lead melody (Drop B) - 4 bars, (step, len, semitone rel. D5)
LEAD = [(0, 6, 7), (6, 2, 8), (8, 4, 7), (12, 2, 5), (14, 2, 4),
        (16, 8, 5), (24, 2, 4), (26, 2, 1), (28, 4, 0),
        (32, 4, 0), (36, 2, 1), (38, 2, 4), (40, 4, 7), (44, 2, 8), (46, 2, 10),
        (48, 6, 12), (54, 2, 10), (56, 4, 8), (60, 2, 7), (62, 2, 4)]


def bar_time(b):
    return b * BAR


def power_chord(semi, octave=0):
    r = ROOT + semi + 12 * octave
    return [r - 12, r - 5, r]


# ---- INTRO: drone, taiko hits, ghost thappu, riser
fx.add(drone(INTRO * BAR + BUILD * BAR), 0, 0.5)
for b in range(INTRO):
    drums.add(taiko(1.0), bar_time(b), 0.9)
    if b % 2 == 1:
        drums.add(taiko(0.7), bar_time(b) + 2.5 * BEAT, 0.8)
    if b >= 4:
        for s, v in enumerate(THAPPU):
            if v and s % 4 == 0:
                perc.add(thappu(v, 1.1), bar_time(b) + s * STEP, 0.4)
    if b in (3, 7):
        vox.add(hey_chant(), bar_time(b) + 3 * BEAT, 0.9)
fx.add(riser(2 * BAR), bar_time(INTRO - 2), 0.35)

# ---- BUILD: full thappu, kicks on 1/3, bass enters, horn swells
start = INTRO
for b in range(BUILD):
    bt = bar_time(start + b)
    pat = THAPPU_FILL if b == BUILD - 1 else THAPPU
    for s, v in enumerate(pat):
        if v:
            perc.add(thappu(v, 1.0 + 0.1 * (s % 3 == 0)), bt + s * STEP, 0.6)
    drums.add(kick(), bt, 1.0)
    drums.add(kick(), bt + 2 * BEAT, 1.0)
    root = PROG[b % len(PROG)]
    for s in range(0, 16, 2):
        bass.add(bass_note(ROOT - 24 + root, STEP * 1.8, 0.9 if s % 4 == 0 else 0.6), bt + s * STEP)
    if b % 2 == 0:
        horns.add(brass(power_chord(root), BAR * 0.95, 0.7), bt, 0.5)
    if b in (3, 7):
        vox.add(hey_chant(), bt + 3 * BEAT, 1.0)
        vox.add(hey_chant(), bt + 3.5 * BEAT, 1.0)
fx.add(riser(2 * BAR), bar_time(start + BUILD - 2), 0.5)


def drop_section(start, bars, intense=False):
    for b in range(bars):
        bt = bar_time(start + b)
        # kick: four-on-floor, plus pushes when intense
        kicks = [0, 4, 8, 12] + ([7, 14] if intense else [10])
        for s in kicks:
            drums.add(kick(), bt + s * STEP, 1.0)
        for s in (4, 12):
            drums.add(snare(), bt + s * STEP, 0.8)
        if intense and b % 4 == 3:          # snare roll into next phrase
            for s in range(12, 16):
                drums.add(snare(0.6 + 0.1 * (s - 12)), bt + s * STEP, 0.6)
        for s in range(16):
            if intense or s % 2 == 0:
                drums.add(hat(0.9 if s % 4 == 2 else 0.5, open_=(s % 8 == 6)), bt + s * STEP)
        pat = THAPPU_FILL if b % 8 == 7 else THAPPU
        for s, v in enumerate(pat):
            if v:
                perc.add(thappu(v, 1.0 + 0.12 * (s % 3 == 0)), bt + s * STEP, 0.65)
        if b % 4 == 0:
            drums.add(taiko(), bt, 0.8)
        # distorted 16th bass, ducked by structure (skip kick steps a little)
        root = PROG[b % len(PROG)]
        for s in range(16):
            v = 1.0 if s % 4 == 0 else (0.75 if s % 2 == 0 else 0.55)
            bass.add(bass_note(ROOT - 24 + root, STEP * 0.9, v), bt + s * STEP)
        # brass riff every 2 bars
        if b % 2 == 0:
            shift = 12 if intense and b % 4 == 2 else 0
            for st, ln, semi in RIFF:
                horns.add(brass(power_chord(semi + shift), ln * STEP * 0.95), bt + st * STEP, 0.55)
        # chants on the backbeat end of each 2-bar phrase
        if b % 2 == 1:
            vox.add(hey_chant(), bt + 2 * BEAT, 0.9)
            vox.add(hey_chant(), bt + 3 * BEAT, 1.0)
            if intense:
                vox.add(hey_chant(), bt + 3.5 * BEAT, 1.0)


# ---- DROP A
fx.add(impact(), bar_time(INTRO + BUILD), 0.8)
drop_section(INTRO + BUILD, DROP_A, intense=False)

# ---- BREAK: half-time stomps, lone lead teaser, big riser
start = INTRO + BUILD + DROP_A
for b in range(BREAK):
    bt = bar_time(start + b)
    drums.add(taiko(), bt, 1.0)
    drums.add(taiko(0.8), bt + 2 * BEAT, 0.9)
    vox.add(hey_chant(), bt + BEAT, 0.8)
    vox.add(hey_chant(), bt + 3 * BEAT, 0.8)
fx.add(drone(BREAK * BAR), bar_time(start), 0.6)
teaser = [(st * STEP, ln * STEP * 0.95, ROOT + 12 + semi) for st, ln, semi in LEAD[:9]]
lead.add(reed_lead(teaser), bar_time(start), 0.45)
fx.add(riser(BREAK * BAR), bar_time(start), 0.6)

# ---- DROP B (more aggressive: double-time hats, extra kicks, lead on top)
start = INTRO + BUILD + DROP_A + BREAK
fx.add(impact(), bar_time(start), 1.0)
drop_section(start, DROP_B, intense=True)
for rep in range(DROP_B // 4):
    bt = bar_time(start + rep * 4)
    up = 12 if rep % 2 == 1 else 0
    notes = [(st * STEP, ln * STEP * 0.95, ROOT + 12 + semi + up) for st, ln, semi in LEAD]
    lead.add(reed_lead(notes), bt, 0.5)

# ---- OUTRO: stop-time hits and a final impact
start = INTRO + BUILD + DROP_A + BREAK + DROP_B
hits = [0, 3, 6, 8, 11, 14]
for b in range(OUTRO - 1):
    bt = bar_time(start + b)
    for s in hits:
        drums.add(kick(), bt + s * STEP, 1.0)
        drums.add(taiko(0.8), bt + s * STEP, 0.6)
        horns.add(brass(power_chord(PROG[b] if s < 8 else 1), STEP * 2), bt + s * STEP, 0.6)
    vox.add(hey_chant(), bt + 3.5 * BEAT, 1.0)
end_t = bar_time(start + OUTRO - 1)
fx.add(impact(), end_t, 1.2)
horns.add(brass(power_chord(0), BAR * 1.5), end_t, 0.7)
bass.add(bass_note(ROOT - 24, BAR * 1.5), end_t, 1.0)
vox.add(hey_chant(), end_t, 1.2)

# ------------------------------------------------------------------ mixdown
# sidechain-style ducking driven by the kick envelope
kick_env = np.abs(drums.buf)
kick_env = lp(kick_env, 8)
duck = 1 - 0.5 * np.clip(kick_env / (kick_env.max() + 1e-9) * 3, 0, 1)

mix = (
    drums.buf * 0.9
    + reverb(perc.buf, 0.8, 0.2) * 0.8
    + bass.buf * duck * 0.55
    + reverb(horns.buf, 1.4, 0.3) * duck * 0.6
    + reverb(lead.buf, 1.8, 0.35) * 0.55
    + reverb(vox.buf, 1.2, 0.35) * 0.55
    + reverb(fx.buf, 2.0, 0.3) * 0.5
)
mix = hp(mix, 28)
mix = np.tanh(1.6 * mix / (np.max(np.abs(mix)) + 1e-9) * 1.4)   # glue + loudness
mix /= np.max(np.abs(mix))
mix *= 0.89                                                       # ~ -1 dBFS
mix = mix[: int(LENGTH * SR)]
fade = int(1.0 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)

# subtle stereo: Haas widen the horns/lead side signal
side = reverb(horns.buf + lead.buf, 0.3, 0.0)[: len(mix)]
side = hp(side, 300) * 0.08 / (np.max(np.abs(side)) + 1e-9)
d = int(0.012 * SR)
L = mix + side
R = mix - np.concatenate([np.zeros(d), side[:-d]])
stereo = np.stack([L, R], axis=1)
stereo /= np.max(np.abs(stereo)) / 0.89

out = sys.argv[1] if len(sys.argv) > 1 else "raavan_rage_bgm.wav"
wavfile.write(out, SR, (stereo * 32767).astype(np.int16))
print(f"wrote {out}: {len(stereo) / SR:.1f}s @ {BPM} BPM")
