#!/usr/bin/env python3
"""
RAAVAN RAGE v3: "monstrous" hybrid-orchestral mass/trailer BGM.

Original composition (theme, harmony, arrangement are new; style inspired by
Tamil mass anthems and epic trailer music).

Real recorded instruments, no General-MIDI:
  * Sonatina Symphonic Orchestra (CC Sampling+): string sections
    (violins I/II, violas, celli, basses - staccato, sustain, tremolo),
    brass sections (trumpets, horns, trombones, tuba, bass trombone),
    large choir, timpani.
  * VSCO-2 Community Edition (CC0): giant ethnic war drum, concert bass
    drum, gong, clash cymbals, suspended-cymbal crescendo rolls, marching snare.
  * Virtuosity Drums (CC0): multi-mic acoustic rock kit.
  * Hybrid sound design synthesized here: braams, distorted growl bass,
    sub drops, risers, reverse swells, heartbeat.

Sample libraries are expected at:
  /home/user/peastman/sso, /home/user/sgossner/vsco-2-ce,
  /home/user/sfzinstruments/virtuosity_drums   (override with env vars
  SSO_DIR, VSCO_DIR, VIRT_DIR)
"""
import os
import sys
import time
import warnings

import numpy as np
import pedalboard as pb
import pyloudnorm as pyln
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sampler import SFZ, Bank, load_sample, SR  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1977)

SSO = os.environ.get("SSO_DIR", "/home/user/peastman/sso") + "/Sonatina Symphonic Orchestra/"
VSCO = os.environ.get("VSCO_DIR", "/home/user/sgossner/vsco-2-ce") + "/"
VIRT = os.environ.get("VIRT_DIR", "/home/user/sfzinstruments/virtuosity_drums") + "/Samples/"

BPM = 140
BEAT = 60 / BPM
BAR = 4 * BEAT
TOTAL_BARS = 68
N = int(TOTAL_BARS * BAR * SR)

# ------------------------------------------------------------------ form
A0, B0, C0, D0, E0, F0 = 0, 12, 20, 36, 44, 60     # section start bars
END_HIT = 63


def key_at(bar):
    return 1 if bar >= E0 else 0


# chords: name -> (bass root midi, choir voicing, upper triad pcs)
CH = {
    "Dm": (38, [57, 62, 65, 69], [62, 65, 69]),
    "Bb": (34, [58, 62, 65, 70], [62, 65, 70]),
    "Gm": (31, [58, 62, 67, 70], [62, 67, 70]),
    "A":  (33, [57, 61, 64, 69], [61, 64, 69]),
    "Eb": (39, [58, 63, 67, 70], [63, 67, 70]),
}
CYCLE = ["Dm", "Dm", "Bb", "Bb", "Gm", "Gm", "A", "A"]
MARCH = ["Dm", "Dm", "Eb", "Dm", "Dm", "Dm", "Bb", "A"]


def chord(bar):
    if bar < 4:
        c = "Dm"
    elif bar < B0:
        c = CYCLE[(bar - 4) % 8]
    elif bar < C0:
        c = MARCH[(bar - B0) % 8]
    elif bar < F0:
        c = CYCLE[(bar - (C0 if bar < E0 else E0)) % 8]
    else:
        c = "Dm"
    return c, key_at(bar)


def croot(bar):
    c, k = chord(bar)
    return CH[c][0] + k


# main theme: (beat, beats, semitones from D4) - 8 bars
THEME = [(0, 1, 0), (1, .5, 0), (1.5, .5, 0), (2, 2, 7), (4, 1, 5), (5, 1, 3), (6, 1, 2), (7, 1, 3),
         (8, 2, 3), (10, 1, 0), (11, 1, 3), (12, 3, 8), (15, 1, 7),
         (16, 1.5, 5), (17.5, .5, 7), (18, 1, 8), (19, 1, 10), (20, 3, 12), (23, .5, 10), (23.5, .5, 8),
         (24, 2, 7), (26, 1, 5), (27, 1, 3), (28, 2, 2), (30, 2, -1)]


# ------------------------------------------------------------------ instruments
def sfz(rel, cc1=None):
    return SFZ(SSO + rel, cc1)


def virt(mic, art):
    import glob
    import re
    files = glob.glob(f"{VIRT}{mic}/{art.split('_')[0]}/{mic}_{art}_vl*.flac")
    layers = {}
    for f in files:
        v = int(re.search(r"_vl(\d+)", f).group(1))
        layers.setdefault(v, []).append(f)
    return Bank([sorted(layers[k]) for k in sorted(layers)])


def vsco(pattern_list):
    return Bank.from_globs([VSCO + p for p in pattern_list])


INST = {
    # strings
    "vln1_stc": sfz("Strings - Performance/1st Violins Staccato.sfz"),
    "vln2_stc": sfz("Strings - Performance/2nd Violins Staccato.sfz"),
    "vla_stc": sfz("Strings - Performance/Violas Staccato.sfz"),
    "vc_stc": sfz("Strings - Performance/Celli Staccato.sfz"),
    "cb_stc": sfz("Strings - Performance/Basses Staccato.sfz"),
    "vln1_sus": sfz("Strings - Performance/1st Violins Sustain.sfz"),
    "vln2_sus": sfz("Strings - Performance/2nd Violins Sustain.sfz"),
    "vla_sus": sfz("Strings - Performance/Violas Sustain.sfz"),
    "vc_sus": sfz("Strings - Performance/Celli Sustain.sfz"),
    "cb_sus": sfz("Strings - Performance/Basses Sustain.sfz"),
    "vln_trem": sfz("Strings - Performance/1st Violins Tremolo.sfz"),
    "vc_trem": sfz("Strings - Performance/Celli Tremolo.sfz"),
    "cb_trem": sfz("Strings - Performance/Basses Tremolo.sfz"),
    "vc_marc": sfz("Strings - Performance/Celli Marcato.sfz"),
    # brass
    "tpt_marc": sfz("Brass - Performance/Trumpets Marcato.sfz"),
    "tpt_sus": sfz("Brass - Performance/Trumpets Sustain.sfz"),
    "hn_marc": sfz("Brass - Performance/Horns Marcato.sfz"),
    "hn_sus": sfz("Brass - Performance/Horns Sustain.sfz"),
    "tbn_marc": sfz("Brass - Performance/Trombones Marcato.sfz"),
    "tbn_sus": sfz("Brass - Performance/Trombones Sustain.sfz"),
    "tbn_stc": sfz("Brass - Performance/Trombones Staccato.sfz"),
    "tuba_sus": sfz("Brass - Performance/Tuba Sustain.sfz"),
    "tuba_stc": sfz("Brass - Performance/Tuba Staccato.sfz"),
    "btbn": sfz("Brass - Performance/Bass Trombone Solo Sustain.sfz"),
    # choir + timpani
    "choir": sfz("Chorus - Performance/Large Chorus.sfz"),
    "timp": sfz("Percussion/Timpani.sfz"),
}

BANK = {
    "giant": vsco(["VSCO 1 Percussion/drums/other/ethnic/giant/mallet/EthnicLargeMallet_hit_pp*",
                   "VSCO 1 Percussion/drums/other/ethnic/giant/mallet/EthnicLargeMallet_hit_mf*",
                   "VSCO 1 Percussion/drums/other/ethnic/giant/mallet/EthnicLargeMallet_hit_f_*",
                   "VSCO 1 Percussion/drums/other/ethnic/giant/mallet/EthnicLargeMallet_hit_ff*"]),
    "giant_stk": vsco(["VSCO 1 Percussion/drums/other/ethnic/giant/sticks/EthnicLargeSticks_hit_mp*",
                       "VSCO 1 Percussion/drums/other/ethnic/giant/sticks/EthnicLargeSticks_hit_f_*",
                       "VSCO 1 Percussion/drums/other/ethnic/giant/sticks/EthnicLargeSticks_hit_fff*"]),
    "giant_hand": vsco(["VSCO 1 Percussion/drums/other/ethnic/giant/hand/EthnicLargeHand_hit_mp*",
                        "VSCO 1 Percussion/drums/other/ethnic/giant/hand/EthnicLargeHand_hit_f_*",
                        "VSCO 1 Percussion/drums/other/ethnic/giant/hand/EthnicLargeHand_hit_ff*"]),
    "bdrum": vsco(["VSCO 1 Percussion/drums/bass/bdrum_mp_*", "VSCO 1 Percussion/drums/bass/bdrum_f_*",
                   "VSCO 1 Percussion/drums/bass/bdrum_ff_*", "VSCO 1 Percussion/drums/bass/bdrum_fff_*"]),
    "gong": vsco(["VSCO 1 Percussion/varMetal/Gong/gong_hit_mf.wav", "VSCO 1 Percussion/varMetal/Gong/gong_hit_f.wav",
                  "VSCO 1 Percussion/varMetal/Gong/gong_hit_ff.wav", "VSCO 1 Percussion/varMetal/Gong/gong_hit_fff.wav"]),
    "clash": vsco(["VSCO 1 Percussion/varMetal/Cymbals/clash/crash_hit_mp_loose.wav",
                   "VSCO 1 Percussion/varMetal/Cymbals/clash/crash_hit_ff_loose.wav",
                   "VSCO 1 Percussion/varMetal/Cymbals/clash/crash_hit_fff_loose*.wav"]),
    "snare_m": vsco(["VSCO 1 Percussion/drums/snare/drum3_marching/snare3_p_*",
                     "VSCO 1 Percussion/drums/snare/drum3_marching/snare3_f_*",
                     "VSCO 1 Percussion/drums/snare/drum3_marching/snare3_fff_*"]),
    "kick_oh": virt("oh", "kick_snoff"), "kick_km": virt("kickmic", "kick_snoff"), "kick_rm": virt("room", "kick_snoff"),
    "snare_oh": virt("oh", "snare_center"), "snare_rm": virt("room", "snare_center"),
    "rim_oh": virt("oh", "snare_rimshot"), "rim_rm": virt("room", "snare_rimshot"),
    "htom": virt("oh", "htom_center"), "ltom": virt("oh", "ltom_center"),
    "crash": virt("oh", "crash_crash"), "hh": virt("oh", "hh_closed"),
}
SUSP_CRESC = VSCO + "VSCO 1 Percussion/varMetal/Cymbals/susp/susp_hit_softmall_roll2_cresc.wav"
SUSP_SCRAPE = VSCO + "VSCO 1 Percussion/varMetal/Cymbals/susp/susp_FX_scrape_continuous.wav"

# ------------------------------------------------------------------ score
EV = {k: [] for k in INST}          # (beat, beats, key, vel, cc1)
HITS = {k: [] for k in BANK}        # (beat, vel, semis)
AUDIO = []                          # (stem, beat, array, align) raw audio placements
KICK_TIMES, HEAVY_TIMES = [], []


def b(bar, step=0.0):
    return bar * 4 + step / 4


def n(inst, beat, beats, key, vel=100, cc1=None):
    EV[inst].append((beat, beats, key, vel, cc1))


def hold(inst, beat, beats, key, vel=100, cc0=60, cc1=110, seg=4):
    """Long note on non-looped sustain samples: re-bowed every `seg` beats with rising dynamics."""
    k = 0
    t = 0.0
    nseg = max(1, int(np.ceil(beats / seg)))
    while t < beats - 1e-6:
        d = min(seg, beats - t)
        cc = cc0 + (cc1 - cc0) * (k / max(1, nseg - 1))
        n(inst, beat + t, d + 0.35, key, vel, cc)
        t += seg
        k += 1


def hit(bank, beat, vel=110, semis=0.0):
    HITS[bank].append((beat, vel, semis))


def kit_kick(beat, vel=120):
    for k in ("kick_oh", "kick_km", "kick_rm"):
        hit(k, beat, vel)
    KICK_TIMES.append(beat)


def kit_snare(beat, vel=120, rim=True):
    hit("snare_oh", beat, vel)
    hit("snare_rm", beat, vel)
    if rim:
        hit("rim_oh", beat, vel)
        hit("rim_rm", beat, vel * 0.8)


def giant(beat, vel=120, sticks=False):
    hit("giant_stk" if sticks else "giant", beat, vel)
    HEAVY_TIMES.append(beat)


def susp_into(beat):
    AUDIO.append(("epic", beat, "susp", "end"))


def theme(insts, bar0, octave=0, vel=115, cc1=110, legato=0.97):
    k = key_at(bar0)
    for beat, dur, s in THEME:
        for inst in insts:
            n(inst, b(bar0) + beat, dur * legato, 62 + s + k + 12 * octave, vel, cc1)


def choir_chord(bar, beats, vel=110, cc1=100, low=True):
    c, k = chord(bar)
    for p in CH[c][1]:
        n("choir", b(bar), beats, p + k, vel, cc1)
    if low:
        r = CH[c][0] + k
        while r < 45:
            r += 12
        n("choir", b(bar), beats, r, vel, cc1)


def braam(beat, intensity=1.0, k=0):
    for inst, keys in (("tbn_sus", [38, 45, 50, 53]), ("tuba_sus", [26, 33]), ("hn_sus", [50, 53, 57]),
                       ("btbn", [33])):
        for p in keys:
            n(inst, beat, 7, p + k, 127, 127 * intensity)
    AUDIO.append(("synth", beat, ("braam", intensity, k), "start"))
    HEAVY_TIMES.append(beat)


def ostinato(bar, vel=110, cc1=110, violins=False, violas=True):
    r = croot(bar)
    c, k = chord(bar)
    pat = [0, 0, 12, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 1, 0, -2] if c == "Dm" else \
          [0, 0, 12, 0, 0, 0, 7, 0, 0, 0, 12, 0, 0, 7, 0, -2]
    acc = {0, 3, 6, 8, 11, 14}
    for s, o in enumerate(pat):
        v = vel if s in acc else vel * 0.72
        n("vc_stc", b(bar, s), 0.25, r + 12 + o, v, cc1)
        n("cb_stc", b(bar, s), 0.25, r + o, v, cc1)
        if violas and s % 2 == 0:
            n("vla_stc", b(bar, s), 0.4, r + 24 + (o if o in (0, 12) else 0), v * 0.95, cc1)
    if violins:
        tri = CH[c][2]
        seq = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1]
        for s, i in enumerate(seq):
            v = vel if s in acc else vel * 0.75
            n("vln1_stc", b(bar, s), 0.25, tri[i] + k + 12, v, cc1)
            n("vln2_stc", b(bar, s), 0.25, tri[i] + k, v * 0.9, cc1)


def timp_roll(beat, beats, key=38, v0=50, v1=127):
    m = int(beats * 6)
    for i in range(m):
        n("timp", beat + i / 6, 0.2, key, v0 + (v1 - v0) * (i / m) ** 1.5)


def snare_roll(beat, beats, v0=40, v1=127):
    m = int(beats * 8)
    for i in range(m):
        hit("snare_m", beat + i / 8, v0 + (v1 - v0) * (i / m) ** 1.3)


def tom_fill(bar, start_step=8, vel=115):
    toms = ["htom", "htom", "htom", "htom", "ltom", "ltom", "ltom", "ltom"]
    for i, s in enumerate(range(start_step, 16)):
        hit(toms[i % 8], b(bar, s), vel - 10 + i * 3)
        if s % 2 == 0:
            giant(b(bar, s), 100 + i * 3, sticks=True)


# =============================================== A. AWAKENING (bars 0-11)
braam(b(0), 0.75)
hit("gong", b(0), 100)
AUDIO.append(("synth", b(0), ("drone", 12 * 4), "start"))
for bar in range(A0, B0):
    r = croot(bar)
    hold("cb_sus", b(bar), 4, r - 12 if r - 12 >= 24 else r, 95, 40 + bar * 6, 50 + bar * 6)
    hold("vc_sus", b(bar), 4, r, 95, 40 + bar * 5, 50 + bar * 5)
    if bar % 2 == 0:
        giant(b(bar), 70 + bar * 4)
        n("timp", b(bar), 2, r if r >= 38 else r + 12, 70 + bar * 4)
    if bar >= 8:
        giant(b(bar, 10), 95 + bar * 2)
        hit("bdrum", b(bar), 110)
for bar in range(4, B0):
    choir_chord(bar, 4.2, 100, 30 + (bar - 4) * 10)
AUDIO.append(("epic", b(2), "scrape", "start"))
AUDIO.append(("epic", b(6), "scrape", "start"))
theme(["hn_sus"], 4, octave=-1, vel=110, cc1=70)
theme(["vc_marc"], 4, octave=-1, vel=95, cc1=60)
braam(b(8), 1.0)
hit("gong", b(8), 120)
hit("clash", b(8), 100)
for bar in (10, 11):
    hold("vln_trem", b(bar), 4, 74, 110, 50 + (bar - 10) * 50, 100 + (bar - 10) * 27, seg=2)
    hold("vc_trem", b(bar), 4, 50, 110, 50 + (bar - 10) * 50, 100 + (bar - 10) * 27, seg=2)
timp_roll(b(10), 7, 45, 40, 127)
snare_roll(b(10, 8), 5.5, 30, 127)
susp_into(b(B0))
AUDIO.append(("synth", b(9), ("riser", 3 * 4), "start"))
AUDIO.append(("synth", b(B0), ("reverse", None), "end"))

# =============================================== B. THE MARCH (bars 12-19)
AUDIO.append(("synth", b(B0), ("impact", 0.8), "start"))
hit("gong", b(B0), 110)
hit("clash", b(B0), 120)
for bar in range(B0, C0):
    i = bar - B0
    ostinato(bar, 100 + i * 3, 90 + i * 4, violins=i >= 4, violas=True)
    r = croot(bar)
    for s, v in ((0, 125), (3, 100), (6, 105), (8, 120), (11, 95), (12, 110), (14, 105), (15, 90)):
        giant(b(bar, s), v)
    hit("bdrum", b(bar), 120)
    hit("bdrum", b(bar, 8), 110)
    for s in range(16):
        hit("snare_m", b(bar, s), 70 if s % 4 else 105)
    tk = r if r >= 38 else r + 12
    n("timp", b(bar), 1, tk, 120)
    n("timp", b(bar, 10), 1, tk, 105)
    for s in (0, 6, 12):
        n("tbn_stc", b(bar, s), 0.4, r + 12, 120, 120)
        n("tbn_stc", b(bar, s), 0.4, r + 19, 115, 120)
        n("tuba_stc", b(bar, s), 0.4, r, 120, 120)
    choir_chord(bar, 4.2, 110, 70 + i * 6)
    if i >= 4:
        kit_kick(b(bar), 115)
        kit_kick(b(bar, 10), 105)
        kit_snare(b(bar, 8), 120)
        if bar % 2 == 0:
            for p in CH[chord(bar)[0]][2]:
                n("hn_sus", b(bar), 7.5, p - 12, 115, 90 + i * 5)
tom_fill(C0 - 1)
snare_roll(b(C0 - 1), 4, 60, 127)
susp_into(b(C0))
AUDIO.append(("synth", b(C0 - 2), ("riser", 8), "start"))
AUDIO.append(("synth", b(C0), ("reverse", None), "end"))


# =============================================== C. DROP 1 (bars 20-35) / E. FINAL (44-59)
def drop(bar0, nbars, final=False):
    k = key_at(bar0)
    for bar in range(bar0, bar0 + nbars):
        i = bar - bar0
        drive = final or i >= 8
        ostinato(bar, 120, 125, violins=True)
        r = croot(bar)
        # kit
        kicks = [0, 3, 6, 8, 10, 14] if drive else [0, 6, 10]
        for s in kicks:
            kit_kick(b(bar, s), 127 if s in (0, 8) else 110)
        snares = [4, 12] if drive else [8]
        for s in snares:
            kit_snare(b(bar, s), 127)
        for s in range(0, 16, 2 if not final else 1):
            hit("hh", b(bar, s), 95 if s % 4 == 0 else 70)
        if i % 4 == 0:
            hit("crash", b(bar), 127)
            hit("clash", b(bar), 120)
        if i % 8 == 7:
            tom_fill(bar)
        # epic percussion
        for s, v in ((0, 127), (3, 105), (6, 110), (8, 127), (11, 100), (14, 110)):
            giant(b(bar, s), v)
        if final:
            for s in (2, 5, 13, 15):
                hit("giant_hand", b(bar, s), 110)
        hit("bdrum", b(bar), 127)
        hit("bdrum", b(bar, 8), 120)
        tk = r if r >= 38 else r + 12
        n("timp", b(bar), 1, tk, 127)
        n("timp", b(bar, 8), 1, tk, 120)
        # low brass punches + tuba roots
        for s in (0, 3, 6):
            n("tbn_stc", b(bar, s), 0.4, r + 12, 125, 127)
            n("tuba_stc", b(bar, s), 0.4, r, 125, 127)
        # choir + string pad
        choir_chord(bar, 4.2, 120, 115 if final else 105)
        if final or i >= 8:
            for p in CH[chord(bar)[0]][2]:
                n("vla_sus", b(bar), 4.3, p + k - 12, 110, 110)
        # synth growl (16ths) + sub
        AUDIO.append(("synth", b(bar), ("growl", r), "start"))
        if i % 4 == 0:
            braam(b(bar), 1.0, k)
            if i % 8 == 0:
                hit("gong", b(bar), 127)
    # the theme
    for rep in range(nbars // 8):
        bar = bar0 + rep * 8
        theme(["tpt_marc", "hn_marc"], bar, 0, 120, 120)
        theme(["tbn_marc"], bar, -1, 115, 115)
        if final or rep == 1:
            theme(["vln1_sus"], bar, 1, 110, 115)
            theme(["vc_marc"], bar, -1, 115, 115)
        if final:
            theme(["choir"], bar, 0, 120, 120, legato=1.0)
            theme(["tpt_sus"], bar, 1 if rep == 1 else 0, 110, 110)


AUDIO.append(("synth", b(C0), ("impact", 1.0), "start"))
drop(C0, 16)

# =============================================== D. BREAKDOWN (bars 36-43)
AUDIO.append(("synth", b(D0), ("impact", 1.0), "start"))
hit("gong", b(D0), 127)
hit("clash", b(D0), 127)
kit_kick(b(D0), 127)
giant(b(D0), 127)
n("timp", b(D0), 2, 38, 127)
for p in (38, 45, 50):
    n("tbn_marc", b(D0), 2, p, 127, 127)
AUDIO.append(("synth", b(D0, 4), ("heartbeat", 4 * 4), "start"))
for bar in range(D0, D0 + 8):
    c, k = chord(bar)
    for p in CH[c][2]:
        hold("vla_sus", b(bar), 4, p - 12, 90, 45 + (bar - D0) * 8, 55 + (bar - D0) * 8)
    hold("vc_sus", b(bar), 4, CH[c][0] + 12, 95, 45 + (bar - D0) * 8, 55 + (bar - D0) * 8)
    hold("cb_sus", b(bar), 4, CH[c][0], 95, 45 + (bar - D0) * 8, 55 + (bar - D0) * 8)
    hold("vln_trem", b(bar), 4, CH[c][2][2] + 12, 80, 30 + (bar - D0) * 10, 40 + (bar - D0) * 10, seg=2)
    if bar % 2 == 0:
        giant(b(bar), 80 + (bar - D0) * 5)
        n("timp", b(bar), 2, CH[c][0] if CH[c][0] >= 38 else CH[c][0] + 12, 80 + (bar - D0) * 5)
theme(["choir"], D0, 0, 105, 75, legato=1.0)            # the goosebump moment: choir sings the theme
theme(["vln2_sus"], D0, 0, 90, 60)
for bar in range(D0 + 6, E0):
    ostinato(bar, 95 + (bar - D0 - 6) * 15, 90 + (bar - D0 - 6) * 15, violins=False)
snare_roll(b(D0 + 6), 8, 30, 127)
timp_roll(b(D0 + 6), 7, 45, 40, 127)
hold("vln_trem", b(D0 + 6), 7.5, 81, 120, 60, 127, seg=2)
hold("vc_trem", b(D0 + 6), 7.5, 57, 120, 60, 127, seg=2)
susp_into(b(E0))
AUDIO.append(("synth", b(D0 + 5), ("riser", 3 * 4 - 1), "start"))
AUDIO.append(("synth", b(E0), ("reverse", None), "end"))

# =============================================== E. FINAL (key +1)
AUDIO.append(("synth", b(E0), ("impact", 1.2), "start"))
drop(E0, 16, final=True)

# =============================================== F. STOP-TIME + FINAL BLOW
for bar in (F0, F0 + 1):
    for s in (0, 3, 6, 8, 11, 14):
        bt = b(bar, s)
        k = 1
        kit_kick(bt, 127)
        hit("crash", bt, 120)
        giant(bt, 127, sticks=s % 2 == 1)
        hit("bdrum", bt, 127)
        n("timp", bt, 0.5, 39, 127)
        for inst, keys in (("tbn_stc", [39, 46, 51]), ("tuba_stc", [27, 39]), ("tpt_marc", [63, 66, 70]),
                           ("hn_marc", [51, 58, 63]), ("vc_stc", [39, 51]), ("cb_stc", [27, 39]),
                           ("vla_stc", [58, 63]), ("vln1_stc", [70, 75]), ("vln2_stc", [63, 66])):
            for p in keys:
                n(inst, bt, 0.45, p, 127, 127)
    AUDIO.append(("synth", b(bar), ("growl", 39), "start"))
    choir_chord(bar, 4.2, 120, 120)
bar = F0 + 2
snare_roll(b(bar), 3, 60, 127)
timp_roll(b(bar), 3, 39, 60, 127)
hold("vln_trem", b(bar), 3, 87, 127, 100, 127, seg=3)
hold("vc_trem", b(bar), 3, 51, 127, 100, 127, seg=3)
for p in CH["Eb"][1]:
    n("choir", b(bar), 3, p, 125, 127)
AUDIO.append(("synth", b(bar), ("riser", 3), "start"))
# --- silence on beat 4, then the final blow
END = b(END_HIT)
braam(END, 1.3, 1)
for g in ("gong", "clash", "crash", "bdrum"):
    hit(g, END, 127)
giant(END, 127)
giant(END, 127, sticks=True)
kit_kick(END, 127)
n("timp", END, 3, 39, 127)
AUDIO.append(("synth", END, ("impact", 1.5), "start"))
for p in CH["Eb"][1] + [51, 46]:
    n("choir", END, 10, p, 127, 127)
for inst, keys in (("tpt_sus", [63, 66, 70]), ("vln1_sus", [75, 82]), ("vln2_sus", [70, 75]), ("vla_sus", [63, 66]),
                   ("vc_sus", [39, 51]), ("cb_sus", [27, 39])):
    for p in keys:
        n(inst, END, 4, p, 127, 127)


# =================================================================== RENDER
def place(buf, y, beat, align="start", jitter=0.0):
    t = beat * BEAT + jitter
    a = int(t * SR) if align == "start" else int(t * SR) - len(y)
    if a < 0:
        y, a = y[-a:], 0
    e = min(len(buf), a + len(y))
    if e > a:
        buf[a:e] += y[: e - a]


def render_inst(name):
    buf = np.zeros((N, 2), np.float32)
    inst = INST[name]
    for beat, beats, key, vel, cc1 in sorted(EV[name]):
        lo = min(r["_lo"] for r in inst.regions)
        hi = max(r["_hi"] for r in inst.regions)
        while key < lo:
            key += 12
        while key > hi:
            key -= 12
        y = inst.render_note(int(key), int(np.clip(vel, 1, 127)), beats * BEAT, cc1)
        if y is not None:
            place(buf, y, beat, jitter=rng.normal(0, 0.006))
    return buf


def render_bank(name):
    buf = np.zeros((N, 2), np.float32)
    for beat, vel, semis in sorted(HITS[name]):
        y = BANK[name].render(float(np.clip(vel, 1, 127)), semis)
        place(buf, y * (vel / 127) ** 0.7, beat, jitter=rng.normal(0, 0.003))
    return buf


# ------------------------------------------------------------- synth design
def sos(kind, f, order=2):
    return butter(order, f, btype=kind, fs=SR, output="sos")


def saw(freq, t, ph=0.0):
    return 2 * ((freq * t + ph) % 1) - 1


def mk_braam(intensity, k):
    dur = 6.0
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros((len(t), 2))
    for ch in range(2):
        x = np.zeros(len(t))
        for m, g in ((26, 1.0), (38, 0.9), (45, 0.7), (50, 0.5), (53, 0.35)):
            f = 440 * 2 ** ((m + k - 69) / 12)
            for d in (-0.12, 0.0, 0.12):
                x += g * saw(f * 2 ** ((d + 0.03 * ch) / 12), t, rng.random())
        x = np.tanh(1.5 * x)
        # sweeping filter: opens fast, closes slowly
        y = np.zeros_like(x)
        seg = 1024
        for i in range(0, len(x), seg):
            tt = i / SR
            fc = 180 + 2600 * np.exp(-tt * 1.8) * min(1, tt / 0.12 + 0.2)
            y[i:i + seg] = sosfilt(butter(2, fc, "low", fs=SR, output="sos"), x[i:i + seg])
        out[:, ch] = y
    env = np.minimum(1, t / 0.03) * np.exp(-t * 0.45)
    env[-2000:] *= np.linspace(1, 0, 2000)
    return (out * env[:, None] * intensity).astype(np.float32)


def mk_drone(beats):
    dur = beats * BEAT
    t = np.arange(int(dur * SR)) / SR
    f = 440 * 2 ** ((26 - 69) / 12)
    x = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.003 * t) + 0.3 * saw(f * 1.5, t)
    x = sosfilt(sos("low", 300), x) * np.minimum(1, t / 6) * np.minimum(1, (dur - t) / 0.5)
    return np.stack([x, x], 1).astype(np.float32)


def mk_growl(root):
    # 16th-note chugs locked to the ostinato accents
    dur = BAR + 0.2
    t = np.arange(int(dur * SR)) / SR
    f = 440 * 2 ** ((root - 12 - 69) / 12)
    x = sum(saw(f * 2 ** (d / 12), t, rng.random()) for d in (-0.15, 0, 0.15)) + 0.8 * np.sin(2 * np.pi * f / 2 * t)
    x = np.tanh(4 * x)
    gate = np.zeros_like(t)
    acc = {0, 3, 6, 8, 11, 14}
    for s in range(16):
        a = int(s * BEAT / 4 * SR)
        L = int(BEAT / 4 * 0.85 * SR)
        g = 1.0 if s in acc else 0.55
        env = np.exp(-np.arange(L) / SR * 9) * g
        gate[a:a + L] = env[: max(0, min(L, len(gate) - a))]
    x = sosfilt(sos("low", 900), x * gate)
    return np.stack([x, x], 1).astype(np.float32)


def mk_impact(intensity):
    t = np.arange(int(4 * SR)) / SR
    boom = np.tanh(3 * np.sin(2 * np.pi * np.cumsum(28 + 110 * np.exp(-t * 4)) / SR) * np.exp(-t * 1.1))
    crack = sosfilt(sos("band", [100, 3000]), rng.standard_normal(len(t))) * np.exp(-t * 14) * 0.6
    tail = sosfilt(sos("high", 3000), rng.standard_normal((len(t), 2)), axis=0) * np.exp(-t * 2.5)[:, None] * 0.15
    m = (boom + crack)[:, None] + tail
    return (m * intensity).astype(np.float32)


def mk_riser(beats):
    dur = beats * BEAT
    n_ = int(dur * SR)
    t = np.arange(n_) / SR
    out = np.zeros((n_, 2))
    for ch in range(2):
        noise = rng.standard_normal(n_)
        y = np.zeros(n_)
        seg = int(0.03 * SR)
        for i in range(0, n_, seg):
            c = 300 + 11000 * (i / n_) ** 2.2
            y[i:i + seg] = sosfilt(butter(1, [c * 0.6, min(c * 1.4, 20000)], "band", fs=SR, output="sos"), noise[i:i + seg])
        f = 55 * 2 ** (4.5 * t / dur)
        tone = sosfilt(sos("low", 4000), saw(1, np.cumsum(f * (1 + 0.005 * ch)) / SR)) * 0.35
        out[:, ch] = (y + tone) * (t / dur) ** 2.5
    return out.astype(np.float32)


def mk_heartbeat(beats):
    dur = beats * BEAT
    out = np.zeros(int(dur * SR) + SR)
    tb = np.arange(int(0.35 * SR)) / SR
    thump = np.sin(2 * np.pi * np.cumsum(45 + 40 * np.exp(-tb * 30)) / SR) * np.exp(-tb * 14)
    for i in range(int(beats // 2)):
        a = int(i * 2 * BEAT * SR)
        g = 0.6 + 0.4 * i / (beats / 2)
        out[a:a + len(thump)] += thump * g
        a2 = a + int(0.22 * SR)
        out[a2:a2 + len(thump)] += thump * 0.7 * g
    out = np.tanh(2 * out)
    return np.stack([out, out], 1).astype(np.float32)


def mk_reverse():
    x, sr = load_sample(VSCO + "VSCO 1 Percussion/varMetal/Cymbals/clash/crash_hit_ff_loose.wav")
    y = x[: int(3.5 * SR)][::-1].copy()
    return y * np.linspace(0.2, 1, len(y))[:, None] ** 2


def synth_stem():
    buf = np.zeros((N, 2), np.float32)
    subs = np.zeros((N, 2), np.float32)
    for stem, beat, spec, align in AUDIO:
        if stem != "synth":
            continue
        kind = spec[0]
        if kind == "braam":
            place(buf, mk_braam(spec[1], spec[2]), beat)
        elif kind == "drone":
            place(subs, mk_drone(spec[1]), beat)
        elif kind == "growl":
            place(subs, mk_growl(spec[1]), beat)
        elif kind == "impact":
            place(buf, mk_impact(spec[1]), beat)
        elif kind == "riser":
            place(buf, mk_riser(spec[1]), beat, "start")
        elif kind == "heartbeat":
            place(subs, mk_heartbeat(spec[1]), beat)
        elif kind == "reverse":
            place(buf, mk_reverse(), beat, "end")
    return buf, subs


def epic_audio_stem():
    buf = np.zeros((N, 2), np.float32)
    cres, _ = load_sample(SUSP_CRESC)
    peak = int(np.argmax(np.abs(cres).mean(1)))
    scrape, _ = load_sample(SUSP_SCRAPE)
    for stem, beat, spec, align in AUDIO:
        if stem != "epic":
            continue
        if spec == "susp":
            place(buf, cres[:peak], beat, "end")
        elif spec == "scrape":
            place(buf, scrape[: int(6 * SR)] * 0.7, beat)
    return buf


# ------------------------------------------------------------------- mixing
def fx(x, *plugins):
    return pb.Pedalboard(list(plugins))(x.T.copy(), SR).T


def hall_ir(seconds=3.4, predelay=0.025):
    t = np.arange(int(seconds * SR)) / SR
    ir = np.zeros((len(t) + int(predelay * SR), 2), np.float32)
    for ch in range(2):
        noise = rng.standard_normal(len(t))
        dec = np.exp(-t * 6.9 / seconds)
        x = noise * dec
        x = sosfilt(sos("low", 7000), x) * 0.7 + sosfilt(sos("low", 2500), x) * 0.5
        er = np.zeros(len(t))
        for _ in range(14):     # early reflections
            p = int(rng.uniform(0.005, 0.08) * SR)
            er[p] += rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
        ir[int(predelay * SR):, ch] = x + er
    return ir / np.sqrt((ir ** 2).sum(0)).max()


IR = hall_ir()


def hall(x, wet):
    if wet <= 0:
        return x
    w = np.stack([fftconvolve(x[:, c], IR[:, c])[: len(x)] for c in range(2)], 1)
    w = sosfilt(sos("high", 180), w, axis=0)
    return x + wet * w.astype(np.float32)


def active_rms_db(x):
    m = np.abs(x).mean(1)
    fr = m[: len(m) // 4096 * 4096].reshape(-1, 4096)
    e = np.sqrt((fr ** 2).mean(1) + 1e-12)
    act = e[e > e.max() * 0.05]
    return 20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-12)


def level(x, target):
    if np.abs(x).max() < 1e-7:
        return x
    return x * 10 ** ((target - active_rms_db(x)) / 20)


def duck_env(times, depth, release=0.18):
    env = np.zeros(N, np.float32)
    rel = np.exp(-np.arange(int(release * 4 * SR)) / (release * SR))
    for bt in times:
        a = int(bt * BEAT * SR)
        e = min(N, a + len(rel))
        env[a:e] = np.maximum(env[a:e], rel[: e - a])
    return (1 - depth * env)[:, None]


# stem -> bus, level (dB active RMS inside the bus), pan
STEMS = {
    "vln1_stc": ("str", -22, -0.35), "vln2_stc": ("str", -24, 0.35), "vla_stc": ("str", -24, 0.15),
    "vc_stc": ("str", -19, 0.25), "cb_stc": ("str", -20, 0.35),
    "vln1_sus": ("str", -21, -0.3), "vln2_sus": ("str", -24, 0.3), "vla_sus": ("str", -24, 0.15),
    "vc_sus": ("str", -22, 0.2), "cb_sus": ("str", -22, 0.3),
    "vln_trem": ("str", -24, -0.2), "vc_trem": ("str", -23, 0.2), "cb_trem": ("str", -24, 0.3),
    "vc_marc": ("str", -21, 0.25),
    "tpt_marc": ("brass", -18, 0.2), "tpt_sus": ("brass", -21, 0.25), "hn_marc": ("brass", -19, -0.3),
    "hn_sus": ("brass", -20, -0.35), "tbn_marc": ("brass", -19, 0.1), "tbn_sus": ("brass", -19, 0.1),
    "tbn_stc": ("brass", -20, 0.05), "tuba_sus": ("brass", -21, 0.0), "tuba_stc": ("brass", -21, 0.0),
    "btbn": ("brass", -22, -0.1),
    "choir": ("choir", -18, 0.0),
    "timp": ("epic", -19, 0.1),
    "giant": ("epic", -16, 0.0), "giant_stk": ("epic", -19, -0.15), "giant_hand": ("epic", -22, 0.2),
    "bdrum": ("epic", -19, 0.0), "gong": ("epic", -22, 0.0), "clash": ("epic", -23, 0.0),
    "snare_m": ("epic", -24, -0.2),
    "kick_oh": ("kit", -20, 0), "kick_km": ("kit", -18, 0), "kick_rm": ("kit", -23, 0),
    "snare_oh": ("kit", -20, 0), "snare_rm": ("kit", -22, 0), "rim_oh": ("kit", -24, 0), "rim_rm": ("kit", -26, 0),
    "htom": ("kit", -22, -0.3), "ltom": ("kit", -21, 0.3), "crash": ("kit", -24, 0.4), "hh": ("kit", -28, 0.25),
}

BUS = {
    #       level, reverb, chain
    "str": (-15.5, 0.38, [pb.HighpassFilter(35), pb.Compressor(-20, 2, 15, 150), pb.HighShelfFilter(8000, 1.5),
                          pb.PeakFilter(300, -1.5, 0.8)]),
    "brass": (-15.0, 0.33, [pb.HighpassFilter(40), pb.Compressor(-20, 2.5, 10, 150), pb.Distortion(2.5),
                            pb.PeakFilter(1800, 1.5, 1.0), pb.PeakFilter(350, -2, 0.9)]),
    "choir": (-16.5, 0.5, [pb.HighpassFilter(110), pb.Compressor(-22, 2, 20, 200), pb.PeakFilter(3000, 1.5, 1)]),
    "epic": (-13.5, 0.3, [pb.HighpassFilter(28), pb.Compressor(-16, 3, 8, 160), pb.LowShelfFilter(70, 2.5)]),
    "kit": (-15.0, 0.12, [pb.HighpassFilter(32), pb.Compressor(-18, 4, 4, 100), pb.Distortion(5),
                          pb.LowShelfFilter(65, 3), pb.PeakFilter(400, -3, 1), pb.HighShelfFilter(6000, 2.5)]),
    "fx": (-17.0, 0.2, [pb.HighpassFilter(25)]),
    "sub": (-17.5, 0.0, [pb.LowpassFilter(1000), pb.Compressor(-20, 4, 5, 80)]),
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "raavan_rage_v3.wav"
    t0 = time.time()
    buses = {k: np.zeros((N, 2), np.float32) for k in BUS}
    for name in INST:
        if not EV[name]:
            continue
        x = render_inst(name)
        bus, lvl, p = STEMS[name]
        x = x * np.array([min(1, 1 - p), min(1, 1 + p)], np.float32)
        buses[bus] += level(x, lvl)
        print(f"  {name:10s} {len(EV[name]):5d} notes  ({time.time() - t0:5.1f}s)")
    for name in BANK:
        if not HITS[name]:
            continue
        x = render_bank(name)
        bus, lvl, p = STEMS[name]
        x = x * np.array([min(1, 1 - p), min(1, 1 + p)], np.float32)
        buses[bus] += level(x, lvl)
    buses["epic"] += level(epic_audio_stem(), -22)
    fxs, subs = synth_stem()
    buses["fx"] += fxs
    buses["sub"] += subs

    # macro dynamics: quiet, dark opening -> builds -> slam; breakdown dips -> final slam
    pts = np.array([(0, -11), (3.95, -10), (4, -9), (7.95, -8), (8, -6), (11.9, -1.5), (12, -3.5), (19.9, -0.5),
                    (20, 0), (36, 0), (36.1, -3), (37, -8), (41.5, -6.5), (43.9, -1), (44, 0), (TOTAL_BARS, 0)])
    tb = np.arange(N) / SR / BAR
    auto = (10 ** (np.interp(tb, pts[:, 0], pts[:, 1]) / 20)).astype(np.float32)[:, None]

    duck_k = duck_env(KICK_TIMES + HEAVY_TIMES, 0.55)
    mix = np.zeros((N, 2), np.float32)
    for name, (lvl, wet, chain) in BUS.items():
        x = buses[name]
        if np.abs(x).max() < 1e-7:
            continue
        x = fx(x / np.abs(x).max(), *chain)
        x = hall(x, wet)
        x = level(x, lvl)
        if name != "fx":
            x = x * auto
        if name in ("sub",):
            x = x * duck_k
        if name in ("str", "choir", "brass"):
            x = x * duck_env(KICK_TIMES, 0.12)
        mix += x
        print(f"  bus {name:6s} done")

    mix = fx(mix, pb.HighpassFilter(24), pb.LowShelfFilter(60, 1.0), pb.PeakFilter(250, -1.5, 0.7),
             pb.PeakFilter(3500, 1.0, 1.0), pb.HighShelfFilter(11000, 1.5), pb.Compressor(-12, 1.8, 30, 250))
    meter = pyln.Meter(SR)
    for _ in range(3):
        cur = meter.integrated_loudness(fx(mix, pb.Limiter(-1.0, 150)))
        mix *= 10 ** ((-10.0 - cur) / 20)
    master = np.clip(fx(mix, pb.Limiter(-1.0, 150)), -0.891, 0.891)
    fade = int(3 * SR)
    master[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    wavfile.write(out, SR, (master * 32767).astype(np.int16))
    print(f"wrote {out}: {N / SR:.1f}s, {meter.integrated_loudness(master):.1f} LUFS ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
