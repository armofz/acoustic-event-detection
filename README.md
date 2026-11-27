# Design and implementation of a spectrogram-based automatic acoustic event detection system for assistive applications

Final degree project (UOC, Applied Data Science). The system detects emergency and everyday sounds from log-Mel spectrograms.

**Classes:** 4 alert classes: alarm, siren, glass breaking, door (knock and doorbell). 3 classes that can be activated: door slam, crying, scream or shout. 10 classes of home sounds that never alert (speech, music, pets...)

**Status:** v2

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

The audio is not in the repo. Download it into *data/raw/*:

- ESC-50: https://github.com/karolpiczak/ESC-50 -> *data/raw/ESC-50-master/*
- FSD50K: https://zenodo.org/records/4060432 -> *data/raw/FSD50K/*
  (with `FSD50K.ground_truth`, `FSD50K.metadata`, `FSD50K.dev_audio`, `FSD50K.eval_audio`)

Each dataset has its own license.

## Run

From the repo root.

Build the dataset: select the clips from ESC-50 and FSD50K and split them into train, validation and test. `manifest_v1.csv` has 5 classes and is used for the representation comparison; `manifest_v2.csv` has 17 classes and is used for the final model.

```bash
python src/build_dataset_v1.py
python src/build_dataset_v2.py
```

Compute the features of every clip, from the same 2 s segment: the log-Mel spectrogram (`logmel_win<n_fft>_<version>.npz`), the constant-Q transform (`cqt_<version>.npz`), the PANNs Cnn14 embedding with Cnn14 frozen (`panns_cnn14_<version>.npz`) and the waveform for fine-tuning PANNs (`waveform_<version>.npz`). They are written to *data/processed/*.

- `--manifest`: list of clips and splits (default: *data/splits/manifest_v1.csv*)
- `--n-fft`: only in make_logmel.py, window length in samples: 512, 1024 or 2048 (default: 1024)

```bash
python src/make_logmel.py --n-fft 1024 --manifest data/splits/manifest_v2.csv
python src/make_cqt.py --manifest data/splits/manifest_v2.csv
python src/extract_panns.py --manifest data/splits/manifest_v2.csv
python src/make_waveforms.py --manifest data/splits/manifest_v2.csv
```

Train a classifier, keep the epoch with the best validation F1 and log the run in MLflow: macro F1, and the F1 and false alarm rate of the alert classes (alone and with the classes that can be activated), as the app sees them. The model is saved in `models/`. To train all the models with 3 seeds on a GPU, see the Colab notebook below.

- `--model`: **CNN** from scratch on log-Mel, **linear** on the PANNs embeddings, or **cnn14 / mobilenetv1** fine-tuning PANNs on the waveforms (default: cnn)
- `--manifest`: list of clips and splits (default: *data/splits/manifest_v1.csv*)
- `--features`: spectrograms, embeddings or waveforms file (default: *data/processed/logmel_win1024_v1.npz*)
- `--seed`: random seed (default: 666)
- `--learning-rate`: learning rate (default: 1e-3, use 1e-4 for fine-tuning)
- `--batch-size`: clips per batch (default: 32, lower it if the GPU runs out of memory)

```bash
python src/train.py --model cnn --manifest data/splits/manifest_v2.csv --features data/processed/logmel_win1024_v2.npz --seed 666
python src/train.py --model linear --manifest data/splits/manifest_v2.csv --features data/processed/panns_cnn14_v2.npz --seed 666
python src/train.py --model mobilenetv1 --manifest data/splits/manifest_v2.csv --features data/processed/waveform_v2.npz --seed 666 --learning-rate 1e-4 --batch-size 16
python src/train.py --model cnn14 --manifest data/splits/manifest_v2.csv --features data/processed/waveform_v2.npz --seed 666 --learning-rate 1e-4 --batch-size 16
```

Evaluate the chosen model on the test split and log the result in MLflow. Run it only once, for the final model.

- `--model`: the same --model used in train.py
- `--model-file`: the .pt file saved by train.py
- `--manifest`, `--features`: the same files used in train.py

```bash
python src/evaluate.py --model cnn14 --model-file models/cnn14_waveform_v2_seed666.pt --manifest data/splits/manifest_v2.csv --features data/processed/waveform_v2.npz
```

Start the real-time detector and the web page http://localhost:8000. On a phone in the same Wi-Fi, open `http://<computer's IP>:8000/?view=band` to use it as the wristband.

```bash
python src/app.py
```
The server listens on all network interfaces and has no authentication, use it only on a trusted network.

## Train on Google Colab (GPU)

`notebooks/train_colab.ipynb` trains all the models of the comparison with dataset v2 and 3 seeds (123, 333, 666) in one run, on a free T4 GPU:

1. Build the features on your computer (commands above) and upload `logmel_win1024_v2.npz`, `panns_cnn14_v2.npz` and `waveform_v2.npz` to *MyDrive/tfg/processed/* in Google Drive.
2. Open the notebook in Colab (*File → Open notebook → GitHub*), choose *Runtime → Change runtime type → T4 GPU* and *Runtime → Run all*.
3. The results (`mlflow.db`, `mlruns/` and `models/`) are saved in *MyDrive/tfg/results_v2/*. If Colab disconnects, run it again: finished runs are skipped.

To see them on your computer, download the folder, for example to `results_v2/` in the repo and run inside it:

```bash
python ../src/relocate_mlflow.py   # MLflow saved the Colab paths; this points them to this folder
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## See the runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

and then open http://127.0.0.1:5000 in the browser.

`data/processed/`, `models/`, `mlflow.db`, `mlruns/` and `results_v2/` are not versioned: they can be rebuilt with the commands above.

## Docs

Design decisions are in `docs/decisions.md`.

`src/panns_models.py` is adapted from [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (MIT). See `THIRD_PARTY_LICENSES.md`.

## License

Code under the MIT License (see `LICENSE`). Datasets and third-party code keep their own licenses (see `THIRD_PARTY_LICENSES.md`).