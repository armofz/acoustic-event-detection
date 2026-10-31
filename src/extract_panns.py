"""Compute a PANNs Cnn14 embedding for every clip in the manifest.
Cnn14 is used frozen, as a feature extractor: each 2 s segment becomes a vector of 2048 values.
Saves data/processed/panns_cnn14_v1.npz with: X, clip_id, offset_s, mean, std.
Run from the repo root: python src/extract_panns.py
"""
import csv
from pathlib import Path

import librosa
import numpy as np
from panns_inference import AudioTagging

from make_logmel import MANIFEST, SAMPLE_RATE, loudest_segment


OUT_FILE = Path("data/processed/panns_cnn14_v1.npz")


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Uses the GPU if there is one, otherwise the CPU. Downloads the checkpoint (~300 MB) to ~/panns_data the first time
    cnn14 = AudioTagging(device="cuda")

    embeddings, offsets = [], []
    for row in rows:
        waveform, _ = librosa.load(row["path"], sr=SAMPLE_RATE)    # mono, 32 kHz, as Cnn14 was trained
        segment, offset = loudest_segment(waveform)    # same 2 s as the log-Mel, so the comparison is fair
        _, embedding = cnn14.inference(segment[None])    # add batch dimension; the AudioSet scores are not used
        embeddings.append(embedding[0])
        offsets.append(offset)
    X = np.stack(embeddings)

    # Mean and std come from train only, so val/test do not leak into the normalization
    is_train = np.array([row["split"] == "train" for row in rows])
    mean, std = float(X[is_train].mean()), float(X[is_train].std())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_FILE, X=X, clip_id=np.array([row["clip_id"] for row in rows]),
             offset_s=np.array(offsets, dtype=np.float32), mean=mean, std=std)
    print(f"{OUT_FILE}: X{X.shape}, {X.min():.2f}..{X.max():.2f}, train mean={mean:.3f} std={std:.3f}")


if __name__ == "__main__":
    main()