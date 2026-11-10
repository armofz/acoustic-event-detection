"""Compute a constant-Q transform (CQT) for every clip in the manifest, to compare it with the log-Mel.
Saves data/processed/cqt_<version>.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/make_cqt.py --manifest data/splits/manifest_v2.csv
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
from make_logmel import MANIFEST, MEL, MIN_POWER, OUT_DIR, SAMPLE_RATE, loudest_segment, read_manifest


CQT = dict(hop_length=MEL["hop_length"], fmin=MEL["fmin"], n_bins=64, bins_per_octave=8)


def cqt(waveform):
    """CQT power in dB: 64 bins x 201 frames for 2 seconds, with the same dB scale as the log-Mel."""
    power = np.abs(librosa.cqt(y=waveform, sr=SAMPLE_RATE, **CQT)) ** 2
    return librosa.power_to_db(power, ref=1.0, amin=MIN_POWER, top_db=None).astype(np.float32)


def parse_args():
    """Read the manifest from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


def main():
    rows, version = read_manifest(parse_args().manifest)
    out_file = OUT_DIR / f"cqt_{version}.npz"

    spectrograms, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz
        segment, offset = loudest_segment(waveform)    # same 2 s as the log-Mel
        spectrograms.append(cqt(segment))
        offsets.append(offset)
    X = np.stack(spectrograms)

    is_train = np.array([row["split"] == "train" for row in rows])
    mean, std = float(X[is_train].mean()), float(X[is_train].std())

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, X=X, clip_id=np.array([row["clip_id"] for row in rows]), offset_s=np.array(offsets, dtype=np.float32), mean=mean, std=std)
    print(f"{out_file}: X{X.shape}, {X.min():.1f}..{X.max():.1f} dB, train mean={mean:.2f} std={std:.2f}")


if __name__ == "__main__":
    main()