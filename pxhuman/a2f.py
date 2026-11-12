"""
Audio2Face-3D integration.

A2F turns audio into ARKit blendshape weights. A MetaPerson avatar carries the
full ARKit set alongside its visemes, so A2F's output maps onto it by name with
no viseme conversion in between — a shorter and more faithful path than
routing through the 15 visemes.

Deliberately built around the WEIGHTS, not around a transport. A2F can be run
as an NVIDIA cloud endpoint, a local NIM container, the Maya plugin or the
Unreal plugin, and every one of them can produce a table of blendshape weights
over time. Parsing that table is stable; pinning the pipeline to one API
surface is not.

`load_weights` is tested against the sample client's CSV layout. `A2FClient`
is written against NVIDIA's documented request shape but is UNTESTED: it needs
credentials and a GPU endpoint that are not available here. Treat it as a
starting point and verify it against your own endpoint before relying on it.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ARKit names A2F emits that matter for the mouth. The avatar carries all 52.
MOUTH_KEYS = [
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
]
EYE_KEYS = [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
]

_TIME_COL = re.compile(r"^(time|timecode|timestamp|time_code|frame|t)$", re.I)


def load_weights(path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read A2F blendshape weights.

    Accepts the sample client's CSV (a time column plus one column per
    blendshape) and the JSON shapes the NIM responses use. Returns the time
    vector in seconds and a name -> weights mapping.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")

    if p.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        return _load_json(text)
    return _load_csv(text)


def _load_csv(text: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("empty weights file")
    header = [c.strip() for c in rows[0]]

    numeric_header = all(_is_num(c) for c in header)
    if numeric_header:
        raise ValueError("weights CSV has no header row naming the blendshapes")

    ti = next((i for i, c in enumerate(header) if _TIME_COL.match(c)), None)
    data = np.array([[_f(c) for c in r] for r in rows[1:]], dtype=np.float64)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("no data rows")

    if ti is None:
        # no explicit clock: assume a fixed frame rate and fill it in later
        times = np.arange(len(data), dtype=np.float64)
        cols = range(data.shape[1])
    else:
        times = data[:, ti]
        cols = [i for i in range(data.shape[1]) if i != ti]
        if times.max() > 1e4:          # looks like milliseconds
            times = times / 1000.0

    out = {header[i]: data[:, i].astype(np.float32) for i in cols
           if i < len(header)}
    return times.astype(np.float32), out


def _load_json(text: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    obj = json.loads(text)
    if isinstance(obj, dict):
        for key in ("frames", "blendshapes", "weights", "animation"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list) or not obj:
        raise ValueError("unrecognised JSON weights layout")

    if isinstance(obj[0], dict) and "names" in obj[0]:
        names = obj[0]["names"]
        times = np.array([f.get("time", i / 30.0) for i, f in enumerate(obj)], np.float32)
        vals = np.array([f["values"] for f in obj], np.float32)
        return times, {n: vals[:, i] for i, n in enumerate(names)}

    times = np.array([f.get("time", i / 30.0) for i, f in enumerate(obj)], np.float32)
    keys = [k for k in obj[0] if k != "time"]
    return times, {k: np.array([f.get(k, 0.0) for f in obj], np.float32)
                   for k in keys}


def _is_num(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
def resample(times: np.ndarray, weights: Dict[str, np.ndarray],
             target_times: np.ndarray) -> Dict[str, np.ndarray]:
    """Put A2F's clock onto the animation's clock."""
    out = {}
    for k, v in weights.items():
        out[k] = np.interp(target_times, times, v).astype(np.float32)
    return out


def normalise_names(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """A2F and exporters disagree on case and separators for ARKit names."""
    canon = {re.sub(r"[^a-z]", "", k.lower()): k
             for k in MOUTH_KEYS + EYE_KEYS}
    out = {}
    for k, v in weights.items():
        key = re.sub(r"[^a-z]", "", k.lower())
        out[canon.get(key, k)] = v
    return out


# ---------------------------------------------------------------------------
class A2FClient:
    """Minimal client for an Audio2Face-3D endpoint.

    UNTESTED. Written from NVIDIA's documented request shape; it has never
    been run against a live endpoint here because that needs credentials and a
    GPU. Verify it against your own deployment before trusting its output, and
    prefer `load_weights` on a file the official sample client produced if you
    want a path that is known to work.
    """

    def __init__(self, url: str, api_key: str | None = None,
                 model: str = "claire", timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def weights_for(self, wav_path, emotion: Dict[str, float] | None = None):
        import urllib.request

        audio = Path(wav_path).read_bytes()
        import base64
        payload = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "model": self.model,
            "format": "wav",
        }
        if emotion:
            payload["emotion"] = emotion

        req = urllib.request.Request(
            f"{self.url}/v1/audio2face",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read().decode("utf-8")
        return _load_json(body)
