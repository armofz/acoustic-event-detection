"""Save the 2 s waveform segment of every clip, to fine-tune the PANNs models (they compute their own log-Mel).
Saves data/processed/waveform_<version>.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/make_waveforms.py --manifest data/splits/manifest_v2.csv
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
from make_logmel import MANIFEST, OUT_DIR, SAMPLE_RATE, loudest_segment, read_manifest


def parse_args():
    """Read the manifest from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


def main():
    rows, version = read_manifest(parse_args().manifest)
    out_file = OUT_DIR / f"waveform_{version}.npz"

    segments, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz, as PANNs were trained
        segment, offset = loudest_segment(waveform)    # same 2 s as the log-Mel
        segments.append(segment)
        offsets.append(offset)
    X = np.stack(segments)

    # mean 0 and std 1, so train.py leaves the waveforms as they are: PANNs normalize their own log-Mel (bn0)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, X=X, clip_id=np.array([row["clip_id"] for row in rows]), offset_s=np.array(offsets, dtype=np.float32), mean=0.0, std=1.0)
    print(f"{out_file}: X{X.shape}, {X.min():.2f}..{X.max():.2f}")


if __name__ == "__main__":
    main()