#!/usr/bin/env python3
"""Download the official llama.cpp tiny-voca GGUF pair into the app assets.

Both files come from ggml-org/tiny-voca-GGUF and are the exact pair converted
from n4ze3m/tiny-voca by the llama.cpp authors. Using the matched pair matters:
the mmproj holds the phoneme vocabulary, the duration/frame transformers, the
source-filter synth and the dizione tables that the text model expects.

The weights stay F32 (~25.8 MB total) on purpose: for a 4.5M-parameter model
the quality difference from quantization is audible on borderline phonemes,
while the speed/memory win is negligible on modern phones. ggml mmaps both
files, so resident memory stays small.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/ggml-org/tiny-voca-GGUF/resolve/main"
OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "models"

FILES = {
    # name: (sha256, size)
    "tiny-voca-F32.gguf": (
        "6cc38fe1a32ac312d906da6549a142de8cb6a2d0d1f10e63966a8f7c49be72cc",
        12169280,
    ),
    "mmproj-tiny-voca-F32.gguf": (
        "c70b6ba351335bb72bdd1ae42412f64490816c3accaec2a2c2990fbcfc33c48c",
        13641888,
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (expected_sha, expected_size) in FILES.items():
        target = OUT / name
        if target.exists() and target.stat().st_size == expected_size and sha256(target) == expected_sha:
            print(f"ok       {name} ({expected_size / 1e6:.1f} MB)")
            continue
        url = f"{BASE}/{name}?download=true"
        tmp = target.with_suffix(target.suffix + ".tmp")
        print(f"download {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        got_sha = sha256(tmp)
        if got_sha != expected_sha or tmp.stat().st_size != expected_size:
            tmp.unlink(missing_ok=True)
            print(f"ERROR: checksum mismatch for {name}: {got_sha}", file=sys.stderr)
            return 1
        tmp.replace(target)
        print(f"ok       {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
