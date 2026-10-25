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
python src/build_dataset_v1.py   # writes data/splits/manifest_v1.csv
python src/make_logmel.py        # writes data/processed/logmel_v1.npz
```

`data/processed/` is not versioned: it can be rebuilt with the commands above.

## Docs

Design decisions are in `docs/decisions.md`.