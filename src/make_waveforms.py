"""Save the 2 s waveform segment of every clip, to fine-tune the PANNs models (they compute their own log-Mel).
Saves data/processed/waveform_v1.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/make_waveforms.py
"""
import csv
import librosa
import numpy as np
from make_logmel import MANIFEST, OUT_DIR, SAMPLE_RATE, loudest_segment


OUT_FILE = OUT_DIR / "waveform_v1.npz"


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    segments, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz, as PANNs were trained
        segment, offset = loudest_segment(waveform)    # same 2 s as the log-Mel
        segments.append(segment)
        offsets.append(offset)
    X = np.stack(segments)

    # mean 0 and std 1, so train.py leaves the waveforms as they are: PANNs normalize their own log-Mel (bn0)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_FILE, X=X, clip_id=np.array([row["clip_id"] for row in rows]), offset_s=np.array(offsets, dtype=np.float32), mean=0.0, std=1.0)
    print(f"{OUT_FILE}: X{X.shape}, {X.min():.2f}..{X.max():.2f}")


if __name__ == "__main__":
    main()