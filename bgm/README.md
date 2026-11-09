# Raavan Rage (BGM)

An original hybrid orchestral BGM in the style of a Tamil mass anthem or epic trailer (D minor, rising to E♭ minor; 140 BPM; 1:56).
The style was inspired by "Raavana Mavandaa" (*Jana Nayagan*). The theme, harmony and arrangement here are all original.

**Listen:** `raavan_rage_v3.mp3`

## Structure
| Time | Section |
|---|---|
| 0:00 | Awakening: braam and gong, sub drone, celli and basses, a distant war drum, choir swelling, horns quietly stating the theme |
| 0:14 | Second braam, string tremolo, timpani and snare rolls, cymbal swell |
| 0:21 | The March: driving string ostinato, war-drum ensemble, marching snare, low-brass stabs, choir |
| 0:34 | Drop 1: full orchestra, rock kit and war drums. Trumpets and horns play the theme, with braams, growl bass and gongs |
| 1:02 | Breakdown: heartbeat, with the choir alone singing the theme over strings, then a long build |
| 1:15 | Final: key lifts to E♭ minor. Choir, trumpets, horns and violins all carry the theme |
| 1:43 | Stop-time hits, a moment of silence, then the final blow |

## Rebuilding
```
python3 produce_v3.py raavan_rage_v3.wav
```
Needs `numpy scipy soundfile pedalboard pyloudnorm`. The script also needs these free sample libraries cloned locally (paths can be set with `SSO_DIR`, `VSCO_DIR`, `VIRT_DIR`):

| Library | Used for | License |
|---|---|---|
| [Sonatina Symphonic Orchestra](https://github.com/peastman/sso) | strings, brass, choir, timpani | CC Sampling Plus 1.0 |
| [VSCO-2 Community Edition](https://github.com/sgossner/VSCO-2-CE) | giant war drum, bass drum, gong, cymbals, marching snare | CC0 |
| [Virtuosity Drums](https://github.com/sfzinstruments/virtuosity_drums) | acoustic rock kit | CC0 |

`sampler.py` is a small offline SFZ sampler that plays these libraries. The braams, growl bass, sub drops, risers, reverse swells and heartbeat are synthesized in `produce_v3.py`.
