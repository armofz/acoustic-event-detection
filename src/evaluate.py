"""Evaluate the chosen model on the test split, once, and log the result in MLflow.
Run from the repo root: python src/evaluate.py --model --model-file  --manifest --features
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from train import DEVICE, EXPERIMENTS, app_metrics, confusion_matrix_figure, load_data, make_model, predict


def parse_args():
    """Read the model type, the saved model and the data files from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=EXPERIMENTS, required=True, help="same --model used in train.py")
    parser.add_argument("--model-file", type=Path, required=True, help=".pt file saved by train.py")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="same features file used in train.py")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, classes, splits = load_data(args.manifest, args.features)
    X_test, y_test = X[splits == "test"], y[splits == "test"]

    saved = torch.load(args.model_file, map_location="cpu")
    if saved["classes"] != classes:
        raise ValueError(f"Classes in {args.model_file} do not match {args.manifest}")
    model = make_model(args.model, X.shape[1], len(classes)).to(DEVICE)
    model.load_state_dict(saved["state_dict"])

    test_pred = predict(model, X_test).argmax(dim=1).numpy()
    report = classification_report(y_test, test_pred, target_names=classes, digits=3)
    metrics = {"test_accuracy": accuracy_score(y_test, test_pred), "test_f1_macro": f1_score(y_test, test_pred, average="macro"),
               **{f"test_{k}": v for k, v in app_metrics(y_test, test_pred, classes).items()}}
    print(report)
    print("  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("final_evaluation")
    with mlflow.start_run(run_name=f"test_{args.model_file.stem}"):
        mlflow.log_params({"model": args.model, "model_file": args.model_file.name, "features": args.features.name,
                           "manifest": args.manifest.name, "n_test": len(y_test), "device": DEVICE})
        mlflow.log_metrics(metrics)
        mlflow.log_text(report, "classification_report_test.txt")
        fig = confusion_matrix_figure(y_test, test_pred, classes)
        fig.axes[0].set_title("Test")
        mlflow.log_figure(fig, "confusion_test.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
