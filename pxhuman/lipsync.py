"""
Text to viseme timeline.

Produces per-frame weights for the 15-viseme set that MetaPerson, Oculus OVR
LipSync and Audio2Face all share: PP FF TH DD kk CH SS nn RR aa E ih oh ou sil.

The viseme track is the seam in this pipeline. Generated here from text, it can
equally be produced from audio by Audio2Face, which emits weights over the same
names; everything downstream is unchanged.
"""
from __future__ import annotations

import math
import re
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS",
           "nn", "RR", "aa", "E", "ih", "oh", "ou"]
VIS_IDX = {v: i for i, v in enumerate(VISEMES)}

# ARPAbet -> viseme. Standard articulatory grouping: sounds made with the same
# visible mouth shape share a viseme, which is why the set is far smaller than
# the phoneme inventory.
ARPA_VIS: Dict[str, str] = {
    "P": "PP", "B": "PP", "M": "PP",
    "F": "FF", "V": "FF",
    "TH": "TH", "DH": "TH",
    "T": "DD", "D": "DD",
    "K": "kk", "G": "kk", "NG": "kk",
    "CH": "CH", "JH": "CH", "SH": "CH", "ZH": "CH",
    "S": "SS", "Z": "SS",
    "N": "nn", "L": "nn",
    "R": "RR", "ER": "RR",
    "AA": "aa", "AE": "aa", "AH": "aa", "AY": "aa", "AW": "aa", "HH": "aa",
    "EH": "E", "EY": "E",
    "IH": "ih", "IY": "ih", "Y": "ih",
    "AO": "oh", "OW": "oh", "OY": "oh",
    "UH": "ou", "UW": "ou", "W": "ou",
}

# nominal durations in seconds at normal rate
DUR: Dict[str, float] = {}
for p in ("P", "B", "T", "D", "K", "G"):          DUR[p] = 0.070
for p in ("F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"): DUR[p] = 0.095
for p in ("CH", "JH"):                            DUR[p] = 0.100
for p in ("M", "N", "NG"):                        DUR[p] = 0.075
for p in ("L", "R"):                              DUR[p] = 0.065
for p in ("W", "Y"):                              DUR[p] = 0.060
for p in ("IH", "EH", "AE", "AH", "UH"):          DUR[p] = 0.090
for p in ("IY", "EY", "AA", "AO", "OW", "UW", "ER"): DUR[p] = 0.130
for p in ("AY", "AW", "OY"):                      DUR[p] = 0.150

# how wide the mouth opens for each viseme, used to drive jawOpen
JAW: Dict[str, float] = {
    "sil": 0.00, "PP": 0.02, "FF": 0.12, "TH": 0.18, "DD": 0.20,
    "kk": 0.22, "CH": 0.24, "SS": 0.14, "nn": 0.18, "RR": 0.26,
    "aa": 0.62, "E": 0.42, "ih": 0.26, "oh": 0.46, "ou": 0.30,
}
# lip rounding, drives mouthPucker / mouthFunnel
ROUND: Dict[str, float] = {
    "sil": 0.0, "PP": 0.05, "FF": 0.0, "TH": 0.0, "DD": 0.0, "kk": 0.0,
    "CH": 0.35, "SS": 0.05, "nn": 0.0, "RR": 0.30,
    "aa": 0.0, "E": 0.0, "ih": 0.0, "oh": 0.55, "ou": 0.80,
}


@dataclass
class Phone:
    arpa: str
    viseme: str
    start: float
    dur: float


# ---------------------------------------------------------------------------
# text -> phonemes
# ---------------------------------------------------------------------------
def _fallback_phones(word: str) -> List[str]:
    """Crude letter-to-sound for words the dictionary does not carry.

    Only needs to be good enough that the mouth moves plausibly; a wrong
    vowel is far less visible than a mouth that stops moving.
    """
    w = word.lower()
    out, i = [], 0
    digraphs = {"ch": "CH", "sh": "SH", "th": "TH", "ph": "F", "wh": "W",
                "ck": "K", "ng": "NG", "qu": "K", "oo": "UW", "ee": "IY",
                "ea": "IY", "ou": "AW", "ow": "OW", "ai": "EY", "ay": "EY",
                "oa": "OW", "oi": "OY", "oy": "OY", "au": "AO", "aw": "AO"}
    single = {"a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F",
              "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L",
              "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R",
              "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K",
              "y": "IY", "z": "Z"}
    while i < len(w):
        if i + 1 < len(w) and w[i:i + 2] in digraphs:
            out.append(digraphs[w[i:i + 2]]); i += 2
        elif w[i] in single:
            out.append(single[w[i]]); i += 1
        else:
            i += 1
    return out


def text_to_phones(text: str) -> List[List[str]]:
    """Return a list of words, each a list of ARPAbet phones (stress stripped)."""
    try:
        import pronouncing
    except ImportError:
        pronouncing = None
    words = re.findall(r"[A-Za-z']+|[.,!?;:]", text)
    out: List[List[str]] = []
    for w in words:
        if w in ".,!?;:":
            out.append(["<pause>" if w in ",;:" else "<stop>"])
            continue
        phones = None
        if pronouncing is not None:
            p = pronouncing.phones_for_word(w.lower())
            if p:
                phones = [re.sub(r"\d", "", x) for x in p[0].split()]
        if not phones:
            phones = _fallback_phones(w)
        out.append(phones or ["AH"])
    return out


# ---------------------------------------------------------------------------
# phonemes -> timed visemes
# ---------------------------------------------------------------------------
def build_timeline(text: str, rate: float = 1.0,
                   word_gap: float = 0.045) -> Tuple[List[Phone], float]:
    t = 0.0
    seq: List[Phone] = []
    for word in text_to_phones(text):
        if word and word[0] in ("<pause>", "<stop>"):
            pause = 0.22 if word[0] == "<pause>" else 0.42
            seq.append(Phone("SIL", "sil", t, pause / rate))
            t += pause / rate
            continue
        for ph in word:
            d = DUR.get(ph, 0.085) / rate
            seq.append(Phone(ph, ARPA_VIS.get(ph, "aa"), t, d))
            t += d
        seq.append(Phone("SIL", "sil", t, word_gap / rate))
        t += word_gap / rate
    return seq, t


def _envelope(n: int, attack: int, release: int) -> np.ndarray:
    """Raised-cosine attack and release around a flat hold."""
    e = np.ones(n)
    if attack > 0:
        k = min(attack, n)
        e[:k] = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, k))
    if release > 0:
        k = min(release, n)
        e[-k:] = np.minimum(e[-k:], 0.5 + 0.5 * np.cos(np.linspace(0, math.pi, k)))
    return e


def viseme_track(seq: List[Phone], total: float, fps: int = 30,
                 coarticulation: float = 0.55, tail: float = 0.35
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame weights, shape (frames, 15), and the time of each frame.

    Visemes overlap rather than switch: a mouth shape starts forming before
    the sound and relaxes after it. Without that overlap the result reads as
    a flipbook of poses instead of speech.
    """
    n = max(2, int(math.ceil((total + tail) * fps)) + 1)
    times = np.arange(n) / fps
    W = np.zeros((n, len(VISEMES)))

    for ph in seq:
        i0 = int(round(ph.start * fps))
        span = max(1, int(round(ph.dur * fps)))
        blend = max(1, int(round(span * coarticulation)))
        a, b = max(0, i0 - blend), min(n, i0 + span + blend)
        if b <= a:
            continue
        env = _envelope(b - a, blend, blend)
        W[a:b, VIS_IDX[ph.viseme]] = np.maximum(W[a:b, VIS_IDX[ph.viseme]], env)

    # keep the total in range: lips cannot hold two shapes at full strength
    tot = W.sum(axis=1, keepdims=True)
    W = np.where(tot > 1.0, W / np.maximum(tot, 1e-6), W)
    W[:, VIS_IDX["sil"]] = np.clip(1.0 - W[:, 1:].sum(axis=1), 0.0, 1.0)
    return W, times


def jaw_and_round(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Derive jaw opening and lip rounding from the viseme mix."""
    jaw = np.array([JAW[v] for v in VISEMES])
    rnd = np.array([ROUND[v] for v in VISEMES])
    return W @ jaw, W @ rnd


def blink_track(times: np.ndarray, period: float = 4.0, jitter: float = 1.6,
                dur: float = 0.13, seed: int = 7) -> np.ndarray:
    """Blinks on an irregular cadence. A perfectly periodic blink reads as a
    machine; the irregularity is what makes it read as alive."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(times))
    t = rng.uniform(0.6, 1.8)
    end = times[-1] if len(times) else 0.0
    while t < end:
        m = (times > t - dur / 2) & (times < t + dur / 2)
        if m.any():
            u = (times[m] - (t - dur / 2)) / dur
            out[m] = np.maximum(out[m], np.sin(np.pi * u) ** 0.65)
        t += max(0.9, period + rng.uniform(-jitter, jitter))
    return out


def idle_head(times: np.ndarray, amount: float = 1.0, seed: int = 3):
    """Small continuous head motion. A head that is perfectly still during
    speech looks switched off."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 2 * np.pi, 6)
    t = times
    pitch = (0.9 * np.sin(0.37 * t + p[0]) + 0.5 * np.sin(0.91 * t + p[1]))
    yaw = (1.1 * np.sin(0.29 * t + p[2]) + 0.6 * np.sin(0.73 * t + p[3]))
    roll = (0.6 * np.sin(0.23 * t + p[4]) + 0.3 * np.sin(0.61 * t + p[5]))
    s = math.radians(1.0) * amount
    return pitch * s, yaw * s, roll * s
