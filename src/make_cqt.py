"""Compute a constant-Q transform (CQT) for every clip in the manifest, to compare it with the log-Mel.
Saves data/processed/cqt_v1.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/make_cqt.py
"""
import csv
import librosa
import numpy as np
from make_logmel import MANIFEST, MEL, MIN_POWER, OUT_DIR, SAMPLE_RATE, loudest_segment


OUT_FILE = OUT_DIR / "cqt_v1.npz"
CQT = dict(hop_length=MEL["hop_length"], fmin=MEL["fmin"], n_bins=64, bins_per_octave=8)


def cqt(waveform):
    """CQT power in dB: 64 bins x 201 frames for 2 seconds, with the same dB scale as the log-Mel."""
    power = np.abs(librosa.cqt(y=waveform, sr=SAMPLE_RATE, **CQT)) ** 2
    return librosa.power_to_db(power, ref=1.0, amin=MIN_POWER, top_db=None).astype(np.float32)


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    spectrograms, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz
        segment, offset = loudest_segment(waveform)    # same 2 s as the log-Mel
        spectrograms.append(cqt(segment))
        offsets.append(offset)
    X = np.stack(spectrograms)

    is_train = np.array([row["split"] == "train" for row in rows])
    mean, std = float(X[is_train].mean()), float(X[is_train].std())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_FILE, X=X, clip_id=np.array([row["clip_id"] for row in rows]), 
             offset_s=np.array(offsets, dtype=np.float32), mean=mean, std=std)
    print(f"{OUT_FILE}: X{X.shape}, {X.min():.1f}..{X.max():.1f} dB, train mean={mean:.2f} std={std:.2f}")


if __name__ == "__main__":
    main()