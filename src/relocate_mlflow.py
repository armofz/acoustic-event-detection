"""Point an MLflow database to its mlruns/ folder after copying both to another computer.
Run from the folder with mlflow.db and mlruns/: python <repo>/src/relocate_mlflow.py
"""
import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("mlflow.db"))
    args = parser.parse_args()
    new_root = str((args.db.parent / "mlruns").resolve())

    with sqlite3.connect(args.db) as db:
        locations = [row[0] for row in db.execute("SELECT artifact_location FROM experiments")]
        old_roots = {loc.split("/mlruns")[0] + "/mlruns" for loc in locations if "/mlruns" in loc}
        for old_root in old_roots - {new_root}:
            db.execute("UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, ?)",
                       (old_root, new_root))
            db.execute("UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)", (old_root, new_root))
            print(f"{old_root} -> {new_root}")
    if old_roots <= {new_root}:
        print("Already pointing to", new_root)


if __name__ == "__main__":
    main()
