"""Compute a log-Mel spectrogram for every clip in the manifest.
Saves data/processed/logmel_v1.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/make_logmel.py
"""
import csv
import librosa
import numpy as np

from pathlib import Path


MANIFEST = Path("data/splits/manifest_v1.csv")
OUT_FILE = Path("data/processed/logmel_v1.npz")

SAMPLE_RATE = 32000    # 32 kHz
SEGMENT_LENGTH = 2 * SAMPLE_RATE
SEARCH_STEP = SAMPLE_RATE // 20    # check a new start every 50 ms
# Same settings as the PANNs Cnn14 model
MEL = dict(n_fft=1024, win_length=1024, hop_length=320, window="hann", n_mels=64, fmin=50, 
           fmax=14000, center=True, pad_mode="reflect", power=2.0)
MIN_POWER = 1e-8    # -80 dB


def loudest_segment(waveform):
    if len(waveform) <= SEGMENT_LENGTH:    # add zeros on both sides if too short
        padding = SEGMENT_LENGTH - len(waveform)
        return np.pad(waveform, (padding // 2, padding - padding // 2)), 0.0
    starts = range(0, len(waveform) - SEGMENT_LENGTH + 1, SEARCH_STEP)
    energies = [np.sum(waveform[s:s + SEGMENT_LENGTH].astype(np.float64) ** 2) for s in starts]
    start = starts[np.argmax(energies)]
    return waveform[start:start + SEGMENT_LENGTH], start / SAMPLE_RATE


def logmel(waveform):
    """Log-Mel spectrogram in dB: 64 bands x 201 frames for 2 seconds."""
    power = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, **MEL)
    return librosa.power_to_db(power, ref=1.0, amin=MIN_POWER, top_db=None).astype(np.float32)


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    spectrograms, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE) # mono, 32 kHz
        segment, offset = loudest_segment(waveform)
        spectrograms.append(logmel(segment))
        offsets.append(offset)
    X = np.stack(spectrograms)

    # Mean and std come from train only, so val/test do not leak into the normalization
    is_train = np.array([row["split"] == "train" for row in rows])
    mean, std = float(X[is_train].mean()), float(X[is_train].std())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_FILE, X=X, clip_id=np.array([row["clip_id"] for row in rows]), offset_s=np.array(offsets, dtype=np.float32), mean=mean, std=std)
    print(f"{OUT_FILE}: X{X.shape}, {X.min():.1f}..{X.max():.1f} dB, train mean={mean:.2f} std={std:.2f}")


if __name__ == "__main__":
    main()