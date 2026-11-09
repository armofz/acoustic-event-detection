"""Build dataset v1 (5 classes) from ESC-50 and FSD50K.
Writes data/splits/manifest_v1.csv: one row per selected clip with its class and split.
Run from the repo root: python src/build_dataset.py
"""
import csv
import json
import random
import wave
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

RAW_DIR = Path("data/raw")
ESC_DIR = RAW_DIR / "ESC-50-master"
FSD_GT = RAW_DIR / "FSD50K" / "FSD50K.ground_truth"
FSD_META = RAW_DIR / "FSD50K" / "FSD50K.metadata"
OUT_FILE = Path("data/splits/manifest_v1.csv")

SEED = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SILENCE_THR = 0.01  # silent clip
NO_RATING = -2.0

CLIPS_PER_CLASS = {("alarm", None): 160, ("phone", None): 160, ("siren", None): 160, ("glass_breaking", None): 160, 
                   ("door_knock_bell", "knock"): 80, ("door_knock_bell", "doorbell"): 80,}

ESC_MAP = {"clock_alarm": ("alarm", None), "door_wood_knock": ("door_knock_bell", "knock"), 
           "glass_breaking": ("glass_breaking", None), "siren": ("siren", None),}

FSD_RULES = [("Telephone", "phone", None), ("Doorbell", "door_knock_bell", "doorbell"), ("Knock", "door_knock_bell", "knock"),
             ("Siren", "siren", None), ("Alarm", "alarm", None), ("Shatter", "glass_breaking", None),]


@dataclass
class Clip:
    source: str
    clip_id: str
    path: Path
    cls: str
    subclass: str | None
    group: str
    tie: float
    rating: float | None = None
    label_count: int = 1
    duration: float = 0.0
    peak: float = 0.0
    split: str = ""



def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_uploaders():
    """Load uploader information for every FSD50K clip"""
    info = {**read_json(FSD_META / "dev_clips_info_FSD50K.json"), **read_json(FSD_META / "eval_clips_info_FSD50K.json")}
    return {freesound_id: d["uploader"] for freesound_id, d in info.items()}


def audio_info(path):
    """Duration (s) and peak (0-1) of a 16-bit WAV, or None if it cannot be read"""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            n_frames, sr = w.getnframes(), w.getframerate()
            raw = w.readframes(n_frames)
    except (wave.Error, EOFError, OSError):
        return None
    samples = np.frombuffer(raw, dtype="<i2").astype(np.int32)    # WAV is little-endian
    peak = int(np.abs(samples).max()) / 32768 if samples.size else 0.0
    return n_frames / sr, peak


def load_esc50(uploader, rng, esc_map):
    """Load ESC-50 clips of the categories in esc_map, and the Freesound IDs of their recordings"""
    clips, sources = [], set()
    for r in read_rows(ESC_DIR / "meta" / "esc50.csv"):
        if r["category"] not in esc_map:
            continue
        cls, subclass = esc_map[r["category"]]
        src = r["src_file"]     # Freesound ID of the original recording
        sources.add(src)
        group = f"u:{uploader[src]}" if src in uploader else f"e:{src}"
        clips.append(Clip(source="ESC-50", clip_id=r["filename"],
                          path=ESC_DIR / "audio" / r["filename"], cls=cls, subclass=subclass,
                          group=group, tie=rng.random()))
    return clips, sources


def find_rule(labels, rules, excluded_labels, alert_classes):
    """Finds the first rule whose label the clip has, or None. Clips with an excluded label only go to alert classes"""
    excluded = not excluded_labels.isdisjoint(labels)
    for rule in rules:
        if rule[0] in labels and (rule[1] in alert_classes or not excluded):
            return rule
    return None


def load_fsd50k(uploader, skip_ids, rng, rules, excluded_labels=frozenset(), alert_classes=frozenset()):
    """Load FSD50K clips of the labels in rules, and how many were skipped as duplicates"""
    with open(FSD_GT / "vocabulary.csv", newline="", encoding="utf-8") as f:
        label_to_mid = {label: mid for _, label, mid in csv.reader(f)}
    ratings = read_json(FSD_META / "pp_pnp_ratings_FSD50K.json")

    clips, duplicates = [], 0
    for part in ("dev", "eval"):    # the official split is ignored
        audio_dir = RAW_DIR / "FSD50K" / f"FSD50K.{part}_audio"
        for r in read_rows(FSD_GT / f"{part}.csv"):
            labels = r["labels"].split(",")
            rule = find_rule(labels, rules, excluded_labels, alert_classes)
            if rule is None:
                continue
            freesound_id = r["fname"]
            if freesound_id in skip_ids:    # same recording is already in ESC-50
                duplicates += 1
                continue
            label, cls, subclass = rule
            # Rating votes: PP=1, PNP=0.5, U=0, NP=-1
            scores = ratings.get(freesound_id, {}).get(label_to_mid[label], [])
            clips.append(Clip(source="FSD50K", clip_id=freesound_id,
                              path=audio_dir / f"{freesound_id}.wav", cls=cls, subclass=subclass,
                              group=f"u:{uploader[freesound_id]}", tie=rng.random(),
                              rating=sum(scores) / len(scores) if scores else NO_RATING,
                              label_count=len(labels)))
    return clips, duplicates


def ranked(candidates):
    """ESC-50 clips first (random order), then FSD50K by rating and fewest labels"""
    esc = sorted((c for c in candidates if c.source == "ESC-50"), key=lambda c: c.tie)
    fsd = sorted((c for c in candidates if c.source == "FSD50K"),
                 key=lambda c: (-c.rating, c.label_count, c.tie))
    return esc + fsd


def select(candidates, clips_per_class):
    """Select the best-ranked clips of each (class, subclass) skipping unreadable or silent ones"""
    candidates_by_key = defaultdict(list)
    for c in candidates:
        candidates_by_key[c.cls, c.subclass].append(c)

    selected = []
    dropped = {"silent": Counter(), "unreadable": Counter()}
    for (cls, subclass), quota in clips_per_class.items():
        picked = []
        for c in ranked(candidates_by_key[cls, subclass]):
            if len(picked) == quota:
                break
            info = audio_info(c.path)
            if info is None:
                dropped["unreadable"][cls] += 1
                continue
            duration, peak = info
            if peak < SILENCE_THR:
                dropped["silent"][cls] += 1
                continue
            c.duration, c.peak = round(duration, 2), round(peak, 3)
            picked.append(c)
        if len(picked) < quota:
            print(f"WARNING: {cls}/{subclass} has only {len(picked)} of {quota} clips")
        selected += picked
    return selected, dropped


def most_needed_split(count, total):
    """Find the split that is furthest below its target size"""
    return max(SPLITS, key=lambda s: SPLITS[s] * total - count[s])


def assign_splits(clips, rng):
    """For each class, give whole uploader groups (largest first) to the split that needs
    clips most. A group keeps its split in every class, so no uploader is in two splits"""
    by_class = defaultdict(lambda: defaultdict(list))
    for c in clips:
        by_class[c.cls][c.group].append(c)

    group_split = {}
    for groups in by_class.values():
        total = sum(len(group_clips) for group_clips in groups.values())
        count = Counter()
        group_order = list(groups.items())
        rng.shuffle(group_order)    # random order among groups of equal size
        group_order.sort(key=lambda item: -len(item[1]))    # largest groups first
        for group, group_clips in group_order:
            if group not in group_split:
                group_split[group] = most_needed_split(count, total)
            split = group_split[group]
            count[split] += len(group_clips)
            for c in group_clips:
                c.split = split


def write_manifest(clips, out_file):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "source", "path", "class", "subclass", "group_id", "split", "duration_s", "peak", "rating"])
        for c in sorted(clips, key=lambda c: (c.cls, c.split, c.clip_id)):
            w.writerow([c.clip_id, c.source, c.path.as_posix(), c.cls, c.subclass or "", c.group, c.split, c.duration, c.peak, 
                        "" if c.rating is None else c.rating])


def print_summary(clips, duplicates, dropped, out_file):
    table = Counter((c.cls, c.split) for c in clips)
    print(f"\n{'class':20}" + "".join(f"{s:>7}" for s in SPLITS) + f"{'total':>7}")
    for cls in sorted({c.cls for c in clips}):
        n = [table[cls, s] for s in SPLITS]
        print(f"{cls:20}" + "".join(f"{x:>7}" for x in n) + f"{sum(n):>7}")
    
    print("\nBy source:", dict(Counter(c.source for c in clips)))
    print("Distinct uploader groups:", len({c.group for c in clips}))
    print("FSD50K duplicates dropped (already in ESC-50):", duplicates)
    print("Dropped as silent:", dict(dropped["silent"]))
    print("Unreadable (not 16-bit WAV):", dict(dropped["unreadable"]))
    print("Written:", out_file)


def main():
    rng = random.Random(SEED)
    uploader = load_uploaders()
    esc, skip_ids = load_esc50(uploader, rng, ESC_MAP)
    fsd, duplicates = load_fsd50k(uploader, skip_ids, rng, FSD_RULES)
    selected, dropped = select(esc + fsd, CLIPS_PER_CLASS)
    assign_splits(selected, rng)
    write_manifest(selected, OUT_FILE)
    print_summary(selected, duplicates, dropped, OUT_FILE)


if __name__ == "__main__":
    main()