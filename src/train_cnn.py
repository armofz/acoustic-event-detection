"""Train the CNN from scratch on the log-Mel spectrograms and log the run in MLflow.
The test split is not used here: it is kept for the final evaluation.
Run from the repo root: python src/train_cnn.py --seed 1 (see --help for more options)
To see the runs: mlflow ui --backend-store-uri sqlite:///mlflow.db   (then open http://127.0.0.1:5000)
"""
import argparse
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Constants for training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
PATIENCE = 10


def parse_args():
    """Read the manifest, the features file and the seed from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest_v1.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/logmel_win1024_v1.npz"))
    parser.add_argument("--seed", type=int, default=666)
    return parser.parse_args()


def load_data(manifest, features):
    """Load and normalize spectrograms, integer labels, class names and split of every clip."""
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = {row["clip_id"]: row for row in csv.DictReader(f)}
    data = np.load(features)
    rows = [rows[clip_id] for clip_id in data["clip_id"]]  # same order as the spectrograms

    classes = sorted({row["class"] for row in rows})
    X = (data["X"] - data["mean"]) / data["std"]
    y = np.array([classes.index(row["class"]) for row in rows])
    splits = np.array([row["split"] for row in rows])
    return X.astype(np.float32), y, classes, splits


def make_cnn(n_classes):
    """Make a CNN with three conv blocks (3x3 filters) and a linear classifier."""
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2), # halves frequency and time
        )
    return nn.Sequential(
        block(1, 32), block(32, 64), block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3), # randomly switch off 30% of the values while training
        nn.Linear(128, n_classes),
    )


def predict(model, X):
    """Return the model scores for every clip, in evaluation mode (no dropout)."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).unsqueeze(1))


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch.unsqueeze(1)), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def confusion_matrix_figure(y_true, y_pred, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes, xticks_rotation=45, colorbar=False, ax=ax)
    ax.set_title("Validation")
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    manifest, features, seed = args.manifest, args.features, args.seed
    # One model file per features file and seed, so runs do not overwrite each other
    model_file = Path("models") / f"cnn_{features.stem}_seed{seed}.pt"
    torch.manual_seed(seed)

    # Load data and split into train and validation
    X, y, classes, splits = load_data(manifest, features)
    X_train, y_train = X[splits == "train"], y[splits == "train"]
    X_val, y_val = X[splits == "val"], y[splits == "val"]

    # Create the model, optimizer, loss function and data loader for training
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE,
                              shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = make_cnn(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Start an MLflow run (all runs go to a local file)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cnn_baseline")
    with mlflow.start_run(run_name=f"cnn_{features.stem}_seed{seed}"):
        mlflow.log_params({"features": features.name, "manifest": manifest.name, "seed": seed, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                           "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "n_train": len(y_train), "n_val": len(y_val),
                           "n_params": sum(p.numel() for p in model.parameters())})
        train_dataset = mlflow.data.from_numpy(X_train, targets=y_train, name=features.stem)
        val_dataset = mlflow.data.from_numpy(X_val, targets=y_val, name=features.stem)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        # Train the model and keep the best one according to validation F1 score
        best_f1, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_logits = predict(model, X_val)
            val_loss = loss_fn(val_logits, torch.from_numpy(y_val)).item()
            val_pred = val_logits.argmax(dim=1).numpy()

            # Macro: average of the F1 of each class
            val_f1 = f1_score(y_val, val_pred, average="macro")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": accuracy_score(y_val, val_pred),
                                "val_f1_macro": val_f1}, step=epoch)
            print(f"epoch {epoch:2d}  train_loss {train_loss:.3f}  val_loss {val_loss:.3f}  val_f1 {val_f1:.3f}")

            # Keep the best model, not the last one
            if val_f1 > best_f1:
                best_f1, best_epoch, best_state = val_f1, epoch, copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= PATIENCE:
                print(f"Early stopping: no improvement since epoch {best_epoch}")
                break

        # Restore the best model and evaluate it on the validation set
        model.load_state_dict(best_state)
        val_pred = predict(model, X_val).argmax(dim=1).numpy()
        report = classification_report(y_val, val_pred, target_names=classes, digits=3)
        print(f"\nBest epoch {best_epoch}\n{report}")

        # Save the model to a file and log the final results in MLflow
        model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "classes": classes}, model_file)
        fig = confusion_matrix_figure(y_val, val_pred, classes)
        mlflow.log_figure(fig, "confusion_val.png")
        plt.close(fig)
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_f1_macro": best_f1, "best_val_accuracy": accuracy_score(y_val, val_pred)})
        mlflow.log_text(report, "classification_report_val.txt")
        mlflow.log_artifact(str(model_file))


if __name__ == "__main__":
    main()