"""Compute a log-Mel spectrogram for every clip in the manifest.
Saves data/processed/logmel_win<n_fft>_<version>.npz with: X, clip_id, offset_s, mean, std, n_fft.
Run from the repo root: python src/make_logmel.py --n-fft 1024 --manifest data/splits/manifest_v2.csv
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np


MANIFEST = Path("data/splits/manifest_v1.csv")
OUT_DIR = Path("data/processed")

SAMPLE_RATE = 32000    # 32 kHz
SEGMENT_LENGTH = 2 * SAMPLE_RATE
SEARCH_STEP = SAMPLE_RATE // 20    # check a new start every 50 ms
# Same settings as the PANNs Cnn14 model, except the window length (n_fft), which is set per run
MEL = dict(hop_length=320, window="hann", n_mels=64, fmin=50, fmax=14000, center=True, pad_mode="reflect", power=2.0)
MIN_POWER = 1e-8    # -80 dB


def parse_args():
    """Read the window length (n_fft) and the manifest from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=1024, help="window length in samples: 512, 1024 (PANNs) or 2048")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    return parser.parse_args()


def read_manifest(manifest):
    """Rows of the manifest and its version ('manifest_v2.csv' -> 'v2'), which goes in the name of the output file."""
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, manifest.stem.removeprefix("manifest_")


def loudest_segment(waveform):
    if len(waveform) <= SEGMENT_LENGTH:    # add zeros on both sides if too short
        padding = SEGMENT_LENGTH - len(waveform)
        return np.pad(waveform, (padding // 2, padding - padding // 2)), 0.0
    starts = range(0, len(waveform) - SEGMENT_LENGTH + 1, SEARCH_STEP)
    energies = [np.sum(waveform[s:s + SEGMENT_LENGTH].astype(np.float64) ** 2) for s in starts]
    start = starts[np.argmax(energies)]
    return waveform[start:start + SEGMENT_LENGTH], start / SAMPLE_RATE


def logmel(waveform, n_fft):
    """Log-Mel spectrogram in dB: 64 bands x 201 frames for 2 seconds."""
    power = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_fft=n_fft, win_length=n_fft, **MEL)
    return librosa.power_to_db(power, ref=1.0, amin=MIN_POWER, top_db=None).astype(np.float32)


def main():
    args = parse_args()
    n_fft = args.n_fft
    rows, version = read_manifest(args.manifest)
    out_file = OUT_DIR / f"logmel_win{n_fft}_{version}.npz"

    spectrograms, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz
        segment, offset = loudest_segment(waveform)
        spectrograms.append(logmel(segment, n_fft))
        offsets.append(offset)
    X = np.stack(spectrograms)

    # Mean and std come from train only, so val/test do not leak into the normalization
    is_train = np.array([row["split"] == "train" for row in rows])
    mean, std = float(X[is_train].mean()), float(X[is_train].std())

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, X=X, clip_id=np.array([row["clip_id"] for row in rows]),
             offset_s=np.array(offsets, dtype=np.float32), mean=mean, std=std, n_fft=n_fft)
    print(f"{out_file}: X{X.shape}, {X.min():.1f}..{X.max():.1f} dB, train mean={mean:.2f} std={std:.2f}")


if __name__ == "__main__":
    main()