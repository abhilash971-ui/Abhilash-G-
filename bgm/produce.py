#!/usr/bin/env python3
"""
RAAVAN RAGE (v2) - original aggressive Tamil-mass style BGM.

Reference vibe: "Raavana Mavandaa" (Jana Nayagan, Anirudh). Nothing is copied
from it; the riff, melody, chords and arrangement are all original.

Pipeline
  1. Compose a full multi-instrument arrangement as separate MIDI stems.
  2. Render every stem with FluidSynth + a sampled General-MIDI soundfont
     (real orchestral strings, brass, choir, shehnai, taiko, timpani,
     distorted guitars, drum kits, hand percussion, sitar, bells ...).
  3. Layer synthesized sound design (sub-kick, impacts, risers, thappu
     cracks) and gang shouts.
  4. Mix each stem (EQ, compression, saturation, panning, reverb, delay,
     sidechain ducking), then master to streaming loudness.

Requirements: fluidsynth, fluid-soundfont-gm, espeak-ng (optional),
              python: numpy scipy mido pedalboard pyloudnorm
Usage:        python3 produce.py [output.wav]
"""
import os
import warnings
warnings.filterwarnings("ignore")
import subprocess
import sys
import tempfile

import mido
import numpy as np
import pedalboard as pb
import pyloudnorm as pyln
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SR = 44100
BPM = 150
BEAT = 60 / BPM
TPB = 480
rng = np.random.default_rng(2026)

D2, D3, D4, D5 = 38, 50, 62, 74

# ---------------------------------------------------------------- form
SECTIONS = {
    "intro": (0, 8), "build": (8, 8), "dropA": (16, 16), "inter": (32, 8),
    "dropB": (40, 16), "bridge": (56, 4), "final": (60, 8), "outro": (68, 4),
}
TOTAL_BARS = 74          # 72 + 2 bars of tail
LEN_S = TOTAL_BARS * 4 * BEAT


def sec(name):
    a, n = SECTIONS[name]
    return range(a, a + n)


def key(bar):            # mass-anthem key lift for the final chorus
    return 1 if bar >= 60 else 0


# 8-bar chord loop: (root semitone from D, quality)
LOOP = [(0, "m"), (0, "m"), (0, "m"), (1, "M"), (0, "m"), (0, "m"), (-4, "M"), (-5, "M")]


def chord_at(bar):
    r, q = LOOP[bar % 8]
    return r + key(bar), q


def triad(root_midi, q):
    return [root_midi, root_midi + (3 if q == "m" else 4), root_midi + 7]


# Main brass hook (16th steps, semitones from D4). 4 bars.
R1 = [(0, 2, 0), (2, 1, 0), (3, 1, 0), (4, 2, 3), (6, 2, 1), (8, 3, 0), (11, 1, -1), (12, 4, 0),
      (16, 2, 7), (18, 1, 7), (19, 1, 8), (20, 2, 7), (22, 2, 3), (24, 2, 5), (26, 2, 3), (28, 2, 1), (30, 2, -1)]
R2 = R1[:8] + [(16, 3, 12), (19, 1, 10), (20, 2, 8), (22, 2, 7), (24, 1, 8), (25, 1, 7), (26, 2, 3), (28, 2, 1), (30, 2, 0)]
RIFF = R1 + [(s + 32, l, n) for s, l, n in R2]

# Lead melody (shehnai), 8 bars, semitones from D5
LEAD = [(0, 12, 7), (12, 2, 8), (14, 2, 7),
        (16, 4, 5), (20, 4, 3), (24, 4, 1), (28, 2, 3), (30, 2, 5),
        (32, 6, 7), (38, 2, 8), (40, 4, 11), (44, 4, 12),
        (48, 12, 12), (60, 2, 13), (62, 2, 12),
        (64, 4, 12), (68, 4, 10), (72, 4, 8), (76, 4, 7),
        (80, 6, 8), (86, 2, 7), (88, 4, 5), (92, 4, 3),
        (96, 6, 1), (102, 2, 3), (104, 4, 5), (108, 4, 7),
        (112, 16, 0)]

# Intro alap (free-ish shehnai phrase), steps from bar 0, semitones from D5
ALAP = [(16, 8, 0), (24, 8, 1), (32, 16, 0), (48, 4, 3), (52, 4, 1), (56, 8, 0),
        (64, 12, 7), (76, 4, 8), (80, 16, 7), (96, 4, 5), (100, 4, 3), (104, 4, 1), (108, 4, 3), (112, 16, 0)]

THAPPU = [1.0, 0, 0.55, 0.9, 0, 0.55, 0.9, 0, 1.0, 0, 0.55, 0.9, 0, 0.6, 0.9, 0.75]
THAPPU_FILL = [1.0, .6, .8, .6, 1.0, .6, .8, .6, 1.0, .7, .9, .7, 1.0, .85, 1.0, 1.0]


# ---------------------------------------------------------------- MIDI parts
class Part:
    def __init__(self, name, program, drum=False, human=0.006):
        self.name, self.program, self.drum, self.human = name, program, drum, human
        self.ev = []            # (beat, order, msg-kwargs)

    def note(self, beat, dur, pitch, vel):
        if not 0 <= pitch <= 127:
            return
        vel = int(np.clip(vel * (1 + rng.normal(0, 0.04)), 1, 127))
        j = rng.normal(0, self.human) if self.human else 0
        b0 = max(0.0, beat + j)
        self.ev.append((b0, 2, dict(type="note_on", note=pitch, velocity=vel)))
        self.ev.append((b0 + max(dur, 0.03), 0, dict(type="note_off", note=pitch, velocity=0)))

    def cc(self, beat, ctrl, val):
        self.ev.append((beat, 1, dict(type="control_change", control=ctrl, value=int(np.clip(val, 0, 127)))))

    def bend(self, beat, val):
        self.ev.append((beat, 1, dict(type="pitchwheel", pitch=int(np.clip(val, -8192, 8191)))))

    def ramp(self, ctrl, b0, b1, v0, v1, n=24):
        for i in range(n + 1):
            self.cc(b0 + (b1 - b0) * i / n, ctrl, v0 + (v1 - v0) * i / n)

    def write(self, path):
        ch = 9 if self.drum else 0
        mf = mido.MidiFile(ticks_per_beat=TPB)
        tr = mido.MidiTrack()
        mf.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
        tr.append(mido.Message("program_change", channel=ch, program=self.program, time=0))
        if not self.drum:
            tr.append(mido.Message("control_change", channel=ch, control=11, value=127, time=0))
            tr.append(mido.Message("control_change", channel=ch, control=7, value=110, time=0))
        evs = sorted(self.ev, key=lambda e: (e[0], e[1]))
        evs.append((TOTAL_BARS * 4, 1, dict(type="control_change", control=1, value=0)))
        last = 0
        for beat, _, kw in evs:
            tick = int(round(beat * TPB))
            tr.append(mido.Message(channel=ch, time=tick - last, **kw))
            last = tick
        mf.save(path)


P = {}


def part(name, program, **kw):
    P[name] = Part(name, program, **kw)
    return P[name]


drums = part("drums", 16, drum=True, human=0.003)      # Power kit
perc = part("perc", 0, drum=True, human=0.004)         # hand percussion (thappu feel)
taiko = part("taiko", 116)
timp = part("timpani", 47)
str_lo = part("strings_lo", 48)
str_hi = part("strings_hi", 49)
trem = part("strings_trem", 44)
brass = part("brass", 61)
tbone = part("trombones", 57)
horns = part("horns", 60)
tuba = part("tuba", 58)
gtrL = part("gtr_L", 30, human=0.004)
gtrR = part("gtr_R", 29, human=0.004)
bass_s = part("bass_synth", 38, human=0.002)
bass_f = part("bass_finger", 33, human=0.003)
choir = part("choir", 52)
oohs = part("oohs", 53)
stab = part("choir_stab", 52, human=0.002)
shehnai = part("shehnai", 111)
saw = part("synth_lead", 81)
ohit = part("orch_hit", 55, human=0.0)
bells = part("bells", 14)
sitar = part("sitar", 104)
pad = part("pad", 89)
revcym = part("rev_cym", 119, human=0.0)

KICKS = []          # beat positions for sub layer + sidechain
IMPACTS = []        # beat positions for impact FX
RISERS = []         # (start_beat, beats)
CHANTS = []         # (beat, word)


def b2(bar, step=0):
    return bar * 4 + step / 4


def kick(beat, vel=120):
    drums.note(beat, 0.2, 36, vel)
    KICKS.append(beat)


def crash(beat, vel=115, alt=False):
    drums.note(beat, 1.0, 57 if alt else 49, vel)


# ---------------- helper writers
def thappu_bar(bar, fill=False, level=1.0):
    pat = THAPPU_FILL if fill else THAPPU
    for s, v in enumerate(pat):
        if not v:
            continue
        b = b2(bar, s)
        if s % 4 == 0:
            perc.note(b, 0.2, 66, 127 * v * level)      # low timbale = big slap
            perc.note(b, 0.2, 64, 100 * v * level)      # low conga
        elif v > 0.8:
            perc.note(b, 0.2, 65, 115 * v * level)      # high timbale crack
        else:
            perc.note(b, 0.2, 62 if s % 2 else 63, 105 * v * level)


def strings_ostinato(bar, vel=100, octave=0):
    r, _ = chord_at(bar)
    flav = 1 if (r - key(bar)) == 0 else 7
    pat = [0, 0, 12, 0, 0, 12, 0, flav, 0, 0, 12, 0, 0, 12, 7, 12]
    for s, o in enumerate(pat):
        acc = 1.0 if s % 4 == 0 else 0.8
        str_lo.note(b2(bar, s), 0.22, D3 + r + o + 12 * octave, vel * acc)


def guitar_chug(bar, pattern=None, vel=110):
    r, _ = chord_at(bar)
    root = D2 + r
    if root < 36:
        root += 12
    pat = pattern or [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    for s, on in enumerate(pat):
        if on:
            for g, off in ((gtrL, 0), (gtrR, 0.006)):
                for n in (root, root + 7, root + 12):
                    g.note(b2(bar, s) + off, 0.2, n, vel * (1 if s % 4 == 0 else 0.85))


def guitar_hold(bar, beats=4, vel=105):
    r, _ = chord_at(bar)
    root = D2 + r
    if root < 36:
        root += 12
    for g in (gtrL, gtrR):
        for n in (root, root + 7, root + 12):
            g.note(b2(bar), beats * 0.97, n, vel)


def bass_bar(bar, mode="8ths", vel=110):
    r, _ = chord_at(bar)
    root = D2 + r - 12
    if root < 26:
        root += 12
    if mode == "whole":
        bass_s.note(b2(bar), 3.9, root, vel)
        bass_f.note(b2(bar), 3.9, root + 12, vel * 0.9)
        return
    pat = [0, 0, 12, 0, 0, 12, 0, 12] if mode == "8ths" else [0] * 16
    stepw = 2 if mode == "8ths" else 1
    for i, o in enumerate(pat):
        b = b2(bar, i * stepw)
        bass_s.note(b, 0.25 * stepw * 0.9, root + o, vel)
        bass_f.note(b, 0.25 * stepw * 0.9, root + 12 + o, vel * 0.85)


def choir_chord(bar, beats=4, vel=95, which=None):
    r, q = chord_at(bar)
    base = D4 + r
    if base > 66:
        base -= 12
    tones = triad(base, q) + [base - 12]
    for c in (which or [choir]):
        for n in tones:
            c.note(b2(bar), beats * 0.98, n, vel)


def strings_hi_chord(bar, beats=4, vel=90, top=True):
    r, q = chord_at(bar)
    base = D4 + r + 12
    if base > 78:
        base -= 12
    for n in triad(base, q) + ([base + 12] if top else []):
        str_hi.note(b2(bar), beats * 0.98, n, vel)


def brass_riff(bar0, vel=118, lead_double=False, shift=0):
    k = key(bar0) + shift
    for s, l, n in RIFF:
        b = b2(bar0, s)
        d = l * 0.25 * 0.92
        p = D4 + n + k
        brass.note(b, d, p, vel)
        brass.note(b, d, p - 12, vel * 0.9)
        tbone.note(b, d, p - 12, vel)
        if lead_double:
            saw.note(b, d, p + 12, vel * 0.8)
    for i in range(4):
        r, _ = chord_at(bar0 + i)
        tuba.note(b2(bar0 + i), 3.8, D2 + r - 12 + 12 * (D2 + r - 12 < 28), 105)


def horn_swell(bar, beats, v0=30, v1=120, chord=True):
    r, q = chord_at(bar)
    base = D3 + r
    tones = triad(base, q) if chord else [base, base + 7]
    horns.ramp(11, b2(bar), b2(bar) + beats, v0, v1)
    for n in tones + [base + 12]:
        horns.note(b2(bar), beats * 0.98, n, 110)


def timp_roll(bar0, beats, pitch=D2, v0=40, v1=125):
    n = int(beats * 6)
    for i in range(n):
        timp.note(b2(bar0) + i / 6, 0.15, pitch, v0 + (v1 - v0) * i / n)


def snare_roll(bar0, beats, v0=50, v1=125):
    n = int(beats * 4)
    for i in range(n):
        frac = i / n
        sub = 2 if frac > 0.5 else 1
        for j in range(sub):
            drums.note(b2(bar0) + i / 4 + j / 8, 0.1, 38, v0 + (v1 - v0) * frac)


def lead_phrase(part_, notes, bar0, base, vel=112, orn=True, kshift=None):
    for s, l, n in notes:
        b = b2(bar0, s)
        k = key(bar0 + s // 16) if kshift is None else kshift
        p = base + n + k
        d = l * 0.25 * 0.96
        if orn and part_ is shehnai:
            if l >= 4:          # gamaka: start a semitone high and slide in
                part_.bend(b - 0.01, 4096)
                for i in range(1, 7):
                    part_.bend(b + i * 0.025, 4096 * (1 - i / 6))
            else:
                part_.bend(b - 0.01, 0)
            if l >= 8:          # delayed vibrato on long notes
                part_.ramp(1, b + d * 0.4, b + d, 0, 70, 8)
                part_.cc(b + d + 0.01, 1, 0)
        part_.note(b, d, p, vel)


def drop_drums(bar, intense=False, phrase_len=8):
    pos = (bar - 16) % phrase_len if bar >= 16 else bar % phrase_len
    kicks = [0, 4, 8, 12] + ([3, 7, 10, 14] if intense else [10])
    for s in kicks:
        kick(b2(bar, s), 127 if s % 4 == 0 else 105)
    for s in (4, 12):
        drums.note(b2(bar, s), 0.2, 38, 127)
        drums.note(b2(bar, s), 0.2, 39, 100)            # clap layer
    for s in range(16):
        if intense:
            drums.note(b2(bar, s), 0.1, 42, 95 if s % 2 == 0 else 65)
        elif s % 2 == 0:
            drums.note(b2(bar, s), 0.1, 42 if s % 4 else 46, 90)
    if bar % 4 == 0:
        crash(b2(bar), 120, alt=bar % 8 == 4)
    if intense:
        drums.note(b2(bar, 8), 0.5, 52, 90)             # china accent
    if pos == phrase_len - 1:                            # tom fill
        for i, (s, t) in enumerate(zip(range(8, 16), [50, 50, 48, 48, 45, 45, 43, 41])):
            drums.note(b2(bar, s), 0.15, t, 100 + i * 3)
        kick(b2(bar, 15), 120)


# ---------------------------------------------------------------- INTRO
for bar in sec("intro"):
    if bar % 2 == 0:
        taiko.note(b2(bar), 1.0, 41, 127)
        taiko.note(b2(bar, 10), 0.5, 41, 110)
        timp.note(b2(bar), 1.0, D2, 110)
    else:
        taiko.note(b2(bar), 0.5, 41, 120)
        taiko.note(b2(bar, 6), 0.5, 45, 100)
        taiko.note(b2(bar, 8), 0.5, 41, 120)
        taiko.note(b2(bar, 14), 0.25, 45, 90)
        taiko.note(b2(bar, 15), 0.25, 45, 105)
    if bar >= 4:
        thappu_bar(bar, level=0.55 + 0.1 * (bar - 4))
    pad.note(b2(bar), 3.95, D3, 90)
    pad.note(b2(bar), 3.95, D3 + 7, 80)
    pad.note(b2(bar), 3.95, D4 + 3, 70)
    trem.note(b2(bar), 3.95, D2, 85 + bar * 4)
    trem.note(b2(bar), 3.95, D3, 80 + bar * 4)
bells.note(b2(0), 4, D5, 110)
bells.note(b2(0), 4, D4, 100)
bells.note(b2(4), 4, D5 + 1, 105)
bells.note(b2(4), 4, D4 + 1, 95)
pad.ramp(11, 0, 8, 20, 110)
trem.ramp(11, 0, 32, 30, 120)
lead_phrase(shehnai, ALAP, 0, D5, vel=105)
for bar in (4, 6):
    horn_swell(bar, 8, 20, 110, chord=False)
choir.ramp(11, b2(4), b2(8), 20, 115)
for bar in range(4, 8):
    choir_chord(bar, 4, 95)
timp_roll(6, 8, D2, 30, 120)
revcym.note(b2(6, 8), 6, 60, 120)
RISERS.append((b2(6), 8))
CHANTS += [(b2(7, 12), "hey"), (b2(7, 14), "hey")]

# ---------------------------------------------------------------- BUILD
IMPACTS.append(b2(8))
crash(b2(8), 120)
for bar in sec("build"):
    i = bar - 8
    strings_ostinato(bar, 85 + i * 4)
    thappu_bar(bar, fill=(bar == 15))
    taiko.note(b2(bar), 0.5, 41, 127)
    taiko.note(b2(bar, 8), 0.5, 41, 115)
    taiko.note(b2(bar, 6), 0.5, 45, 100)
    if bar % 2 == 0:
        timp.note(b2(bar), 1.0, D2 + chord_at(bar)[0], 115)
        horn_swell(bar, 8, 35, 120)
    if i >= 4:
        kick(b2(bar), 127)
        kick(b2(bar, 10), 110)
        drums.note(b2(bar, 8), 0.2, 38, 120)
        guitar_hold(bar, 4, 100)
        bass_bar(bar, "whole")
        choir_chord(bar, 4, 100)
        tbone.note(b2(bar), 0.8, D3 + chord_at(bar)[0] - 12, 120)
        brass.note(b2(bar), 0.8, D3 + chord_at(bar)[0], 120)
    else:
        oohs.note(b2(bar), 3.9, D4, 90)
        oohs.note(b2(bar), 3.9, D4 + 7, 85)
snare_roll(14, 8, 45, 127)
timp_roll(15, 4, D2, 60, 127)
RISERS.append((b2(14), 8))
revcym.note(b2(14, 8), 6, 60, 127)
for bar in (11, 13, 15):
    CHANTS += [(b2(bar, 12), "hey"), (b2(bar, 14), "ha")]


# ---------------------------------------------------------------- DROPS
def drop(bars, intense):
    first = bars[0]
    IMPACTS.append(b2(first))
    for bar in bars:
        drop_drums(bar, intense)
        thappu_bar(bar, fill=(bar % 8 == 7))
        strings_ostinato(bar, 108, octave=1 if intense and bar % 2 else 0)
        guitar_chug(bar, vel=118 if intense else 110)
        bass_bar(bar, "16ths" if intense else "8ths")
        choir_chord(bar, 4, 100 if intense else 92)
        taiko.note(b2(bar), 0.5, 41, 127)
        taiko.note(b2(bar, 8), 0.5, 41, 120)
        if intense:
            for s in (3, 6, 11, 14):
                taiko.note(b2(bar, s), 0.25, 45, 100)
        if bar % 2 == 0:
            timp.note(b2(bar), 1.0, D2 + chord_at(bar)[0], 120)
        if (bar - first) % 4 == 0:
            brass_riff(bar, lead_double=intense)
            ohit.note(b2(bar), 0.5, D4 + key(bar), 120)
            ohit.note(b2(bar), 0.5, D3 + key(bar), 110)
        if bar % 2 == 1:
            for s, w in ((8, "ha"), (12, "hey")) if not intense else ((8, "ha"), (12, "hey"), (14, "hey")):
                CHANTS.append((b2(bar, s), w))
                for n in triad(D4 + chord_at(bar)[0], chord_at(bar)[1]):
                    stab.note(b2(bar, s), 0.35, n, 115)
        if intense:
            strings_hi_chord(bar, 4, 85)


drop(list(sec("dropA")), intense=False)
for bar in (16, 24):
    horn_swell(bar, 32, 70, 115)

# ---------------------------------------------------------------- INTERLUDE (half-time, shehnai theme)
IMPACTS.append(b2(32))
for bar in sec("inter"):
    kick(b2(bar), 120)
    kick(b2(bar, 10), 105)
    drums.note(b2(bar, 8), 0.2, 38, 118)
    drums.note(b2(bar, 8), 0.2, 39, 90)
    for s in range(0, 16, 2):
        drums.note(b2(bar, s), 0.1, 51, 75)          # ride
    if bar % 4 == 0:
        crash(b2(bar), 105)
    taiko.note(b2(bar), 0.5, 41, 120)
    taiko.note(b2(bar, 8), 0.5, 41, 110)
    thappu_bar(bar, level=0.7)
    guitar_hold(bar, 4, 90)
    bass_bar(bar, "whole", 100)
    strings_hi_chord(bar, 4, 80, top=False)
    choir_chord(bar, 4, 85, which=[oohs])
    pad.note(b2(bar), 3.95, D3 + chord_at(bar)[0], 80)
lead_phrase(shehnai, LEAD, 32, D5)
for bar0 in (35, 39):          # sitar fills at phrase ends
    for i, n in enumerate([12, 13, 12, 10, 8, 7, 8, 7, 5, 3, 1, 0]):
        sitar.note(b2(bar0, 4 + i), 0.2, D4 + n, 110)
timp_roll(38, 8, D2, 40, 127)
snare_roll(39, 4, 50, 127)
RISERS.append((b2(38), 8))
revcym.note(b2(38, 8), 6, 60, 127)
CHANTS += [(b2(39, 12), "hey"), (b2(39, 14), "hey")]

# ---------------------------------------------------------------- DROP B (lead on top)
drop(list(sec("dropB")), intense=True)
lead_phrase(shehnai, LEAD, 40, D5, vel=120)
lead_phrase(shehnai, LEAD, 48, D5, vel=124)
for bar in (40, 48):
    horn_swell(bar, 32, 80, 120)

# ---------------------------------------------------------------- BRIDGE (stop-time)
IMPACTS.append(b2(56))
for bar, hits in ((56, [0, 3, 6, 10, 12]), (57, [0, 3, 6, 10, 12]), (58, list(range(0, 16, 2)))):
    for s in hits:
        b = b2(bar, s)
        n = D3 + (1 if (s == 12 and bar < 58) else 0)
        kick(b, 127)
        crash(b, 110, alt=s % 2 == 1)
        taiko.note(b, 0.5, 41, 127)
        ohit.note(b, 0.4, n + 12, 120)
        for g in (gtrL, gtrR):
            for o in (0, 7, 12):
                g.note(b, 0.45, n - 12 + o, 120)
        brass.note(b, 0.4, n + 12, 125)
        brass.note(b, 0.4, n, 125)
        tbone.note(b, 0.4, n - 12, 125)
        bass_s.note(b, 0.45, n - 24, 120)
        for t in triad(n + 12, "m"):
            stab.note(b, 0.35, t, 118)
bells.note(b2(56), 4, D5, 110)
bells.note(b2(57), 4, D5 + 1, 110)
snare_roll(59, 3, 60, 127)
timp_roll(59, 3, D2, 60, 127)
trem.note(b2(59), 3, D3, 110)
trem.note(b2(59), 3, D4 + 1, 110)
RISERS.append((b2(58), 7))
CHANTS += [(b2(57, 14), "hey"), (b2(58, 14), "hey"), (b2(59, 8), "ha"), (b2(59, 10), "ha")]

# ---------------------------------------------------------------- FINAL (key +1)
drop(list(sec("final")), intense=True)
lead_phrase(shehnai, LEAD, 60, D5, vel=127)
lead_phrase(str_hi, LEAD, 60, D5 + 12, vel=110, orn=False)
lead_phrase(saw, LEAD, 60, D5, vel=95, orn=False)
horn_swell(60, 32, 90, 125)
bells.note(b2(60), 4, D5 + 1, 115)

# ---------------------------------------------------------------- OUTRO
IMPACTS.append(b2(68))
for bar in (68, 69):
    for s in [0, 3, 6, 8, 11, 14]:
        b = b2(bar, s)
        n = D3 + 1
        kick(b, 127)
        crash(b, 110, alt=s % 2 == 1)
        taiko.note(b, 0.5, 41, 127)
        ohit.note(b, 0.4, n + 12, 120)
        for g in (gtrL, gtrR):
            for o in (0, 7, 12):
                g.note(b, 0.4, n - 12 + o, 120)
        brass.note(b, 0.4, n + 12, 125)
        tbone.note(b, 0.4, n - 12, 125)
        bass_s.note(b, 0.4, n - 24, 120)
        for t in triad(n + 12, "m"):
            stab.note(b, 0.3, t, 118)
    thappu_bar(bar, fill=True)
    CHANTS += [(b2(bar, 12), "hey"), (b2(bar, 14), "hey")]
snare_roll(70, 4, 50, 127)
timp_roll(70, 4, D2 + 1, 50, 127)
trem.note(b2(70), 4, D3 + 1, 120)
RISERS.append((b2(70), 4))
# the final blow
END = b2(71)
IMPACTS.append(END)
kick(END, 127)
crash(END, 127)
crash(END, 127, alt=True)
taiko.note(END, 2, 41, 127)
timp.note(END, 2, D2 + 1, 127)
ohit.note(END, 1, D4 + 1, 127)
bells.note(END, 6, D5 + 1, 120)
for n in triad(D4 + 1, "m") + [D3 + 1]:
    choir.note(END, 7, n, 115)
    str_hi.note(END, 7, n + 12, 110)
    brass.note(END, 6, n, 120)
for n in (D3 + 1 - 12, D3 + 1 - 5, D3 + 1):
    tbone.note(END, 6, n, 120)
    gtrL.note(END, 6, n, 120)
    gtrR.note(END, 6, n, 120)
bass_s.note(END, 6, D2 + 1 - 12, 120)
tuba.note(END, 6, D2 + 1 - 12, 120)
CHANTS.append((END, "hey"))


# ================================================================ RENDER
def render(p, workdir):
    mid = os.path.join(workdir, p.name + ".mid")
    wav = os.path.join(workdir, p.name + ".wav")
    p.write(mid)
    subprocess.run(["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5", "-r", str(SR),
                    "-T", "wav", "-O", "float", "-F", wav, SF2, mid],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    return fit(x)


N = int(LEN_S * SR)


def fit(x):
    if len(x) >= N:
        return x[:N]
    return np.concatenate([x, np.zeros((N - len(x), 2), np.float32)])


# ---------------------------------------------------------------- synth layers
def sos(kind, f, order=2):
    return butter(order, f, btype=kind, fs=SR, output="sos")


def sub_kick_track():
    t = np.arange(int(0.5 * SR)) / SR
    f = 42 + 110 * np.exp(-t * 30)
    k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    k = np.tanh(2.5 * k)
    out = np.zeros(N, np.float32)
    for b in KICKS:
        a = int(b * BEAT * SR)
        e = min(N, a + len(k))
        out[a:e] += k[:e - a]
    return np.stack([out, out], 1)


def impact_track():
    t = np.arange(int(3.5 * SR)) / SR
    boom = np.tanh(2.8 * np.sin(2 * np.pi * np.cumsum(28 + 80 * np.exp(-t * 5)) / SR) * np.exp(-t * 1.3))
    noiseL = sosfilt(sos("high", 2500), rng.standard_normal(len(t))) * np.exp(-t * 1.8) * 0.3
    noiseR = sosfilt(sos("high", 2500), rng.standard_normal(len(t))) * np.exp(-t * 1.8) * 0.3
    out = np.zeros((N, 2), np.float32)
    for b in IMPACTS:
        a = int(b * BEAT * SR)
        e = min(N, a + len(t))
        out[a:e, 0] += (boom + noiseL)[:e - a]
        out[a:e, 1] += (boom + noiseR)[:e - a]
    return out


def riser_track():
    out = np.zeros((N, 2), np.float32)
    for b, beats in RISERS:
        dur = beats * BEAT
        n = int(dur * SR)
        t = np.arange(n) / SR
        for ch in range(2):
            noise = rng.standard_normal(n)
            y = np.zeros(n)
            seg = int(0.04 * SR)
            for i in range(0, n, seg):
                c = 400 + 9000 * (i / n) ** 2
                y[i:i + seg] = sosfilt(butter(1, [c * 0.6, min(c * 1.5, 20000)], "band", fs=SR, output="sos"), noise[i:i + seg])
            f = 60 * 2 ** (4 * t / dur)
            tone = (2 * ((np.cumsum(f * (1 + 0.004 * ch)) / SR) % 1) - 1) * 0.25
            y = (y + sosfilt(sos("low", 3000), tone)) * (t / dur) ** 2.2
            a = int(b * BEAT * SR)
            e = min(N, a + n)
            out[a:e, ch] += y[:e - a]
    return out


def thappu_crack_track():
    """Synth crack layered on hand percussion for the thappu 'slap'."""
    t = np.arange(int(0.2 * SR)) / SR
    out = np.zeros(N, np.float32)
    for beat, order, kw in P["perc"].ev:
        if kw["type"] != "note_on" or kw["note"] not in (65, 66):
            continue
        crack = sosfilt(butter(2, [700, 4000], "band", fs=SR, output="sos"), rng.standard_normal(len(t))) * np.exp(-t * 45)
        skin = np.sin(2 * np.pi * 230 * t * (1 + 0.3 * np.exp(-t * 50))) * np.exp(-t * 25)
        s = np.tanh(2 * (crack + 0.7 * skin)) * kw["velocity"] / 127
        a = int(beat * BEAT * SR)
        e = min(N, a + len(t))
        out[a:e] += s[:e - a]
    return np.stack([out, out], 1)


def chant_track(workdir):
    voices = ["en+m1", "en+m2", "en+m3", "en+m4", "en+m7", "en-us+m3", "en+m5", "en+m6"]
    words = {"hey": "Hey!", "ha": "Hah!"}
    samples = {}
    for w, text in words.items():
        layers = []
        for i, v in enumerate(voices):
            path = os.path.join(workdir, f"ch_{w}_{i}.wav")
            r = subprocess.run(["espeak-ng", "-v", v, "-p", str(15 + 6 * i), "-s", "260", "-a", "200",
                                "-w", path, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                continue
            sr, x = wavfile.read(path)
            x = x.astype(np.float32) / 32768
            x = resample_poly(x, SR, sr)
            nz = np.where(np.abs(x) > 0.02)[0]
            if len(nz) == 0:
                continue
            x = x[nz[0]:nz[-1]]
            # slight random pitch/time variation per voice
            ratio = rng.uniform(0.92, 1.08)
            x = resample_poly(x, int(1000 * ratio), 1000)
            layers.append(x)
        if not layers:
            return None
        L = max(len(x) for x in layers) + int(0.03 * SR)
        st = np.zeros((L, 2), np.float32)
        for i, x in enumerate(layers):
            d = int(rng.uniform(0, 0.025) * SR)
            pan = (i / (len(layers) - 1)) * 1.6 - 0.8
            st[d:d + len(x), 0] += x * (1 - max(0, pan))
            st[d:d + len(x), 1] += x * (1 + min(0, pan))
        samples[w] = st / np.max(np.abs(st))
    out = np.zeros((N, 2), np.float32)
    for b, w in CHANTS:
        s = samples[w]
        a = int(b * BEAT * SR)
        e = min(N, a + len(s))
        out[a:e] += s[:e - a]
    return out


# ================================================================ MIX
def fx(x, *plugins):
    board = pb.Pedalboard(list(plugins))
    return board(x.T.copy(), SR).T


def pan(x, p):
    g = np.array([min(1, 1 - p), min(1, 1 + p)], np.float32)
    return x * g


def active_rms_db(x):
    m = np.abs(x).mean(1)
    frames = m[: len(m) // 2048 * 2048].reshape(-1, 2048)
    e = np.sqrt((frames ** 2).mean(1) + 1e-12)
    act = e[e > e.max() * 0.03]
    return 20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-12)


def level(x, target_db):
    return x * 10 ** ((target_db - active_rms_db(x)) / 20)


def sidechain_env(depth, release=0.16):
    env = np.zeros(N, np.float32)
    rel = np.exp(-np.arange(int(release * 3 * SR)) / (release * SR))
    for b in KICKS:
        a = int(b * BEAT * SR)
        e = min(N, a + len(rel))
        env[a:e] = np.maximum(env[a:e], rel[: e - a])
    return (1 - depth * env)[:, None]


def room(size, wet, damp=0.5, width=1.0):
    return pb.Reverb(room_size=size, wet_level=wet, dry_level=1.0, damping=damp, width=width)


CHAIN = {
    # name: (target active RMS dB, pan, plugins, sidechain depth)
    "drums": (-15, 0, [pb.HighpassFilter(30), pb.Compressor(-16, 4, 3, 90), pb.Distortion(3),
                       pb.LowShelfFilter(80, 2), pb.HighShelfFilter(9000, 2), room(0.25, 0.1)], 0),
    "perc": (-21, 0.15, [pb.HighpassFilter(120), pb.Compressor(-18, 4, 2, 80), pb.PeakFilter(2500, 3, 1),
                         room(0.35, 0.15)], 0),
    "taiko": (-19, -0.1, [pb.HighpassFilter(35), pb.Compressor(-14, 3, 5, 150), room(0.8, 0.28, 0.4)], 0),
    "timpani": (-23, 0.1, [pb.HighpassFilter(35), room(0.85, 0.3)], 0),
    "strings_lo": (-22, -0.3, [pb.HighpassFilter(90), pb.Compressor(-18, 3, 5, 100), room(0.6, 0.22)], 0.35),
    "strings_hi": (-24, 0.35, [pb.HighpassFilter(200), room(0.8, 0.35)], 0.25),
    "strings_trem": (-24, 0, [pb.HighpassFilter(60), room(0.85, 0.35)], 0),
    "brass": (-19, 0.15, [pb.HighpassFilter(90), pb.Compressor(-18, 3, 8, 120), pb.Distortion(4),
                          pb.PeakFilter(1800, 2.5, 1), room(0.6, 0.25)], 0.2),
    "trombones": (-22, -0.2, [pb.HighpassFilter(60), pb.Compressor(-18, 3, 8, 120), pb.Distortion(5), room(0.6, 0.22)], 0.25),
    "horns": (-24, -0.35, [pb.HighpassFilter(80), room(0.85, 0.35)], 0.25),
    "tuba": (-26, 0, [pb.LowpassFilter(1200), room(0.5, 0.12)], 0.4),
    "gtr_L": (-22, -0.85, [pb.HighpassFilter(95), pb.Distortion(14), pb.LowpassFilter(6500),
                           pb.PeakFilter(400, -3, 1), pb.Compressor(-20, 4, 5, 80)], 0.45),
    "gtr_R": (-22, 0.85, [pb.HighpassFilter(95), pb.Distortion(16), pb.LowpassFilter(6000),
                          pb.PeakFilter(400, -3, 1), pb.Compressor(-20, 4, 5, 80)], 0.45),
    "bass_synth": (-20, 0, [pb.Distortion(8), pb.LowpassFilter(1500), pb.Compressor(-20, 5, 5, 80)], 0.6),
    "bass_finger": (-26, 0, [pb.HighpassFilter(60), pb.LowpassFilter(2500), pb.Compressor(-20, 4, 5, 80)], 0.5),
    "choir": (-23, 0, [pb.HighpassFilter(150), pb.Compressor(-20, 2, 20, 200), room(0.9, 0.4, 0.3)], 0.3),
    "oohs": (-26, 0, [pb.HighpassFilter(150), room(0.9, 0.45, 0.3)], 0.2),
    "choir_stab": (-23, 0, [pb.HighpassFilter(180), pb.Compressor(-18, 3, 2, 80), room(0.7, 0.3)], 0),
    "shehnai": (-18, 0.05, [pb.HighpassFilter(250), pb.Compressor(-20, 3, 5, 120), pb.PeakFilter(3000, 2, 1),
                            pb.Delay(BEAT * 0.75, 0.25, 0.16), room(0.7, 0.28)], 0),
    "synth_lead": (-26, -0.1, [pb.HighpassFilter(200), pb.LowpassFilter(7000), pb.Chorus(), room(0.6, 0.25)], 0.2),
    "orch_hit": (-22, 0, [pb.HighpassFilter(80), room(0.8, 0.35)], 0),
    "bells": (-26, 0.25, [pb.HighpassFilter(200), room(0.9, 0.4)], 0),
    "sitar": (-22, -0.3, [pb.HighpassFilter(200), pb.Delay(BEAT * 0.5, 0.3, 0.2), room(0.6, 0.25)], 0),
    "pad": (-27, 0, [pb.LowpassFilter(5000), room(0.9, 0.5)], 0.35),
    "rev_cym": (-26, 0, [pb.HighpassFilter(300), room(0.7, 0.3)], 0),
    "sub": (-22, 0, [pb.LowpassFilter(110)], 0),
    "impacts": (-21, 0, [room(0.9, 0.3)], 0),
    "risers": (-27, 0, [pb.HighpassFilter(200), room(0.8, 0.35)], 0),
    "thappu_crack": (-25, 0.15, [pb.HighpassFilter(300), room(0.3, 0.15)], 0),
    "chants": (-21, 0, [pb.HighpassFilter(180), pb.Compressor(-22, 4, 2, 80), pb.Distortion(10),
                        pb.LowpassFilter(7000), room(0.75, 0.35, 0.4, 1.0)], 0),
}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "raavan_rage_v2.wav"
    with tempfile.TemporaryDirectory() as wd:
        stems = {}
        for name, p in P.items():
            if p.ev:
                stems[name] = render(p, wd)
                print(f"rendered {name:13s} ({len(p.ev)} events)")
        stems["sub"] = sub_kick_track()
        stems["impacts"] = impact_track()
        stems["risers"] = riser_track()
        stems["thappu_crack"] = thappu_crack_track()
        if CHANTS:
            ch = chant_track(wd)
            if ch is not None:
                stems["chants"] = ch

    mix = np.zeros((N, 2), np.float32)
    for name, x in stems.items():
        if np.max(np.abs(x)) < 1e-6:
            continue
        target, p, plugins, sc = CHAIN[name]
        y = fx(x / np.max(np.abs(x)), *plugins)
        y = pan(y, p)
        y = level(y, target)
        if sc:
            y = y * sidechain_env(sc)
        mix += y

    # master bus
    mix = fx(mix, pb.HighpassFilter(25), pb.Compressor(-14, 2.0, 25, 180),
             pb.LowShelfFilter(70, 1.5), pb.PeakFilter(300, -1.5, 0.8), pb.HighShelfFilter(10000, 2.0))
    meter = pyln.Meter(SR)
    target_lufs = -9.0
    for _ in range(3):
        lim = fx(mix, pb.Limiter(-1.0, 120))
        cur = meter.integrated_loudness(lim)
        mix *= 10 ** ((target_lufs - cur) / 20)
    master = fx(mix, pb.Limiter(-1.0, 120))
    master = np.clip(master, -0.891, 0.891)
    fade = int(2.5 * SR)
    master[-fade:] *= np.linspace(1, 0, fade)[:, None]
    wavfile.write(out_path, SR, (master * 32767).astype(np.int16))
    print(f"wrote {out_path}: {N / SR:.1f}s, {BPM} BPM, "
          f"{meter.integrated_loudness(master):.1f} LUFS, {len(stems)} stems")


if __name__ == "__main__":
    main()
