"""
Arabic text to visemes.

Maps Arabic orthography onto the same 15-viseme set the English path uses, so
everything downstream — timeline, envelopes, jaw, glTF writing — is shared.

Two facts shape this module:

Arabic script is close to phonemic for consonants but writes short vowels only
as optional diacritics, and most real text carries none. Short vowels are
therefore restored when the diacritics are present and inferred when they are
not. For lip-sync that inference is far less damaging than it would be for
speech synthesis: a wrong short vowel moves the jaw slightly too far, while a
missing one stops the mouth entirely.

Emphatic consonants (ص ض ط ظ) and the pharyngeals (ح ع) differ from their plain
counterparts mostly in tongue root and pharynx, which the lips do not show.
They share visemes with their plain pairs rather than being given their own.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# character classes
# ---------------------------------------------------------------------------
FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
FATHATAN, KASRATAN, DAMMATAN = "ً", "ٍ", "ٌ"
SUKUN, SHADDA = "ْ", "ّ"
DAGGER_ALIF = "ٰ"
TATWEEL = "ـ"

HARAKAT = {FATHA, KASRA, DAMMA, FATHATAN, KASRATAN, DAMMATAN, SUKUN, SHADDA,
           DAGGER_ALIF}

# consonant -> viseme. Emphatics share with their plain pair: the difference
# is in the tongue root, which is invisible.
CONS_VIS: Dict[str, str] = {
    "ب": "PP", "م": "PP",
    "ف": "FF",
    "ث": "TH", "ذ": "TH", "ظ": "TH",
    "ت": "DD", "د": "DD", "ض": "DD", "ط": "DD",
    "ك": "kk", "ق": "kk", "خ": "kk", "غ": "kk",
    "ج": "CH", "ش": "CH",
    "س": "SS", "ز": "SS", "ص": "SS",
    "ن": "nn", "ل": "nn",
    "ر": "RR",
    "ح": "aa", "ع": "aa", "ه": "aa", "ء": "aa",
    "أ": "aa", "إ": "aa", "آ": "aa", "ؤ": "aa", "ئ": "aa", "ة": "DD",
    "و": "ou", "ي": "ih", "ى": "aa",
    "ا": "aa",
}

# short vowels
VOWEL_VIS = {FATHA: "aa", KASRA: "ih", DAMMA: "ou",
             FATHATAN: "aa", KASRATAN: "ih", DAMMATAN: "ou"}

SUN_LETTERS = set("تثدذرزسشصضطظلن")

# durations, seconds at normal rate
D_CONS = 0.075
D_EMPH = 0.090           # emphatics and pharyngeals run a little longer
D_SHORT_V = 0.080
D_LONG_V = 0.165
D_GEMINATE = 0.055       # extra hold added by shadda
EMPHATIC = set("صضطظحعق")


# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    text = text.replace(TATWEEL, "")
    return text


def has_diacritics(word: str) -> bool:
    return any(c in HARAKAT for c in word)


def _is_long_vowel(ch: str, prev_haraka: str | None) -> str | None:
    """ا و ي act as long vowels when not carrying their own vowel."""
    if ch == "ا":
        return "aa"
    if ch == "و" and prev_haraka in (DAMMA, None):
        return "ou"
    if ch == "ي" and prev_haraka in (KASRA, None):
        return "ih"
    return None


def word_to_visemes(word: str) -> List[Tuple[str, float]]:
    """Return [(viseme, duration_seconds)] for one Arabic word."""
    word = normalise(word)
    out: List[Tuple[str, float]] = []
    diacritised = has_diacritics(word)

    # definite article: the lam assimilates into a following sun letter, so it
    # is not pronounced and the sun letter is doubled instead
    if word.startswith("ال") and len(word) > 2:
        third = word[2]
        if third in SUN_LETTERS:
            out.append(("aa", D_SHORT_V))
            word = word[2:]
            out.append((CONS_VIS.get(third, "aa"), D_CONS + D_GEMINATE))
            word = word[1:]
        else:
            out.append(("aa", D_SHORT_V))
            out.append(("nn", D_CONS))
            word = word[2:]

    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        if ch in HARAKAT:
            i += 1
            continue
        if ch not in CONS_VIS:
            i += 1
            continue

        # gather the diacritics attached to this letter
        j = i + 1
        marks = []
        while j < n and word[j] in HARAKAT:
            marks.append(word[j]); j += 1

        # Tanween counts as a vowel here: it is a short vowel plus a final /n/.
        # Checking only the three plain harakat drops both the vowel and the
        # nuun from every nunated word.
        prev_haraka = None
        for m in marks:
            if m in (FATHA, KASRA, DAMMA, FATHATAN, KASRATAN, DAMMATAN):
                prev_haraka = m

        longv = _is_long_vowel(ch, None)
        if longv and not any(m in (FATHA, KASRA, DAMMA) for m in marks):
            # Acting as a long vowel: APPEND it. Folding it into the previous
            # entry overwrites that entry's viseme, which silently deletes the
            # consonant before every alif, waw and ya in the text.
            out.append((longv, D_LONG_V))
            # Accusative tanween is very often written over the alif
            # (شكراً). Consuming the alif without reading its marks drops
            # the final nuun of a large share of real text.
            if any(m in (FATHATAN, KASRATAN, DAMMATAN) for m in marks):
                out.append(("nn", D_CONS))
            i = j
            continue

        dur = D_EMPH if ch in EMPHATIC else D_CONS
        if SHADDA in marks:
            dur += D_GEMINATE
        out.append((CONS_VIS[ch], dur))

        if prev_haraka:
            out.append((VOWEL_VIS[prev_haraka], D_SHORT_V))
            if marks and marks[-1] in (FATHATAN, KASRATAN, DAMMATAN):
                out.append(("nn", D_CONS))
        elif SUKUN in marks:
            pass                                   # explicitly vowelless
        elif not diacritised:
            # Undiacritised text: insert a default short vowel so the mouth
            # keeps moving. Fatha is both the most frequent short vowel and
            # the most visually neutral choice.
            nxt = word[j] if j < n else ""
            if nxt not in ("ا", "و", "ي") and i + 1 < n:
                out.append(("aa", D_SHORT_V * 0.8))
        i = j
    return out


# ---------------------------------------------------------------------------
def build_timeline_ar(text: str, rate: float = 1.0, word_gap: float = 0.050):
    """Arabic counterpart of lipsync.build_timeline."""
    from .lipsync import Phone

    tokens = re.findall(r"[؀-ۿݐ-ݿ]+|[.،,!؟?؛;:]", text)
    seq: List[Phone] = []
    t = 0.0
    for tok in tokens:
        if re.match(r"[.،,!؟?؛;:]", tok):
            pause = 0.42 if tok in ".!؟?" else 0.22
            seq.append(Phone("SIL", "sil", t, pause / rate))
            t += pause / rate
            continue
        for vis, dur in word_to_visemes(tok):
            d = dur / rate
            seq.append(Phone(tok, vis, t, d))
            t += d
        seq.append(Phone("SIL", "sil", t, word_gap / rate))
        t += word_gap / rate
    return seq, t
