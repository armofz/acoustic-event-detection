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

# Trains the CNN and logs the run in MLflow
# --manifest: list of clips and splits (default: data/splits/manifest_v1.csv)
# --features: spectrograms file (default: data/processed/logmel_win1024_v1.npz)
# --seed: random seed (default: 666)
python src/train_cnn.py --manifest data/splits/manifest_v1.csv --features data/processed/logmel_win1024_v1.npz --seed 666
```

To see the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and then open http://127.0.0.1:5000 in the browser.

`data/processed/`, `models/` and `mlflow.db` are not versioned: they can be rebuilt with the commands above.

## Docs

Design decisions are in `docs/decisions.md`.