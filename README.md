# Design and implementation of a spectrogram-based automatic acoustic event detection system for assistive applications

Final degree project (UOC, Applied Data Science). The system detects emergency and everyday sounds from log-Mel spectrograms.

**Classes:** alarm, phone, siren, glass breaking, door (knock and doorbell).

**Status:** v1

## Setup

Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

This installs the CPU version of PyTorch, which works on any computer.
On Linux, the microphone needs the PortAudio library (`sudo apt install libportaudio2`).

## Data

The audio is not in the repo. Download it into `data/raw/`:

- ESC-50: https://github.com/karolpiczak/ESC-50 -> `data/raw/ESC-50-master/`
- FSD50K: https://zenodo.org/records/4060432 -> `data/raw/FSD50K/`
  (with `FSD50K.ground_truth`, `FSD50K.metadata`, `FSD50K.dev_audio`, `FSD50K.eval_audio`)

Each dataset has its own license.

## Run

From the repo root:

```bash
# Writes data/splits/manifest_v1.csv
python src/build_dataset_v1.py

# Writes data/processed/logmel_win<n_fft>_v1.npz
# --n-fft: window length in samples: 512, 1024 or 2048 (default: 1024)
python src/make_logmel.py --n-fft 1024

# Writes data/processed/cqt_v1.npz
# Computes a constant-Q transform (CQT) for every clip, to compare it with the log-Mel
python src/make_cqt.py

# Writes data/processed/panns_cnn14_v1.npz
# Computes a PANNs Cnn14 embedding (2048 values) for every clip. Cnn14 is used frozen
python src/extract_panns.py

# Writes data/processed/waveform_v1.npz
# Saves the 2 s waveform of every clip, the input for fine-tuning PANNs
python src/make_waveforms.py

# Trains a classifier and logs the run in MLflow
# --model: CNN from scratch on log-Mel, linear layer on the PANNs embeddings, or fine-tuning cnn14 or mobilenetv1 of PANNs on the waveforms (default: cnn)
# --manifest: list of clips and splits (default: data/splits/manifest_v1.csv)
# --features: spectrograms, embeddings or waveforms file (default: data/processed/logmel_win1024_v1.npz)
# --seed: random seed (default: 666)
# --learning-rate: learning rate (default: 1e-3, use 1e-4 for fine-tuning)
# --batch-size: clips per batch (default: 32, lower it if the GPU runs out of memory)
python src/train.py --model cnn --manifest data/splits/manifest_v1.csv --features data/processed/logmel_win1024_v1.npz --seed 666
python src/train.py --model linear --manifest data/splits/manifest_v1.csv --features data/processed/panns_cnn14_v1.npz --seed 666
python src/train.py --model mobilenetv1 --manifest data/splits/manifest_v1.csv --features data/processed/waveform_v1.npz --seed 666 --learning-rate 1e-4 --batch-size 16
python src/train.py --model cnn14 --manifest data/splits/manifest_v1.csv --features data/processed/waveform_v1.npz --seed 666 --learning-rate 1e-4 --batch-size 16

# Starts the real-time detector and the web page http://localhost:8000 on the computer
# On a phone in the same Wi-Fi, open http://<computer's IP>:8000/?view=band to use it as the wristband
python src/app.py
```
The server listens on all network interfaces and has no authentication, use it only on a trusted network.

To see the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and then open http://127.0.0.1:5000 in the browser.

`data/processed/`, `models/` and `mlflow.db` are not versioned: they can be rebuilt with the commands above.

## Docs

Design decisions are in `docs/decisions.md`.

`src/panns_models.py` is adapted from [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (MIT). See `THIRD_PARTY_LICENSES.md`.