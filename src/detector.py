"""Real-time detector of the MVP that listens to the microphone (or plays a test clip), classifies the last 2 s
every 0.5 s and decides when to raise an alert. Every result is passed to on_message as a dict.
The audio stays in memory and is never saved to disk.
"""
import csv
import queue
import random
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import torch

from make_cqt import cqt
from make_logmel import MANIFEST, OUT_DIR, SAMPLE_RATE, SEGMENT_LENGTH, logmel
from panns_models import load_panns
from train import make_model


MODELS_DIR = Path("models")
HOP_LENGTH = SAMPLE_RATE // 2    # a new 2 s window every 0.5 s
STARTUP_S = 1.0
MIN_LEVEL_DB = -50
COOLDOWN_S = 5
ALL_THREADS = torch.get_num_threads()
PATTERNS = {"alarm": [400, 200, 400, 200, 400], "door_knock_bell": [150, 100, 150], "glass_breaking": [100, 50, 100, 50], 
            "phone": [300, 150, 300], "siren": [1000, 400, 1000, 400, 1000, 400],}


def list_models():
    """Model files that can be loaded"""
    return sorted(f.name for f in MODELS_DIR.glob("*.pt") if (OUT_DIR / f"{features_name(f)}.npz").exists())


def features_name(model_file):
    """'linear_panns_cnn14_v1_seed1.pt' -> 'panns_cnn14_v1'."""
    return Path(model_file).stem.split("_", 1)[-1].rsplit("_seed", 1)[0]


def load_model(model_file):
    """Load a model trained by train.py, and returns its classes and a function from a 2 s waveform to class probabilities."""
    model_type, features = Path(model_file).name.split("_", 1)[0], features_name(model_file)
    checkpoint = torch.load(MODELS_DIR / model_file, map_location="cpu")
    classes = checkpoint["classes"]
    data = np.load(OUT_DIR / f"{features}.npz")
    mean, std = float(data["mean"]), float(data["std"])
    # Same input as in training: log-Mel or CQT for the CNN, Cnn14 embedding for linear, waveform for fine-tuned PANNs
    if features.startswith("logmel"):
        n_fft = int(data["n_fft"])
        to_input = lambda window: logmel(window, n_fft)
    elif features.startswith("cqt"):
        to_input = cqt
    elif features.startswith("panns_cnn14"):
        cnn14 = load_panns("cnn14").eval()
        to_input = lambda window: cnn14(torch.from_numpy(window)[None])["embedding"][0].numpy()
    else:
        to_input = lambda window: window

    model = make_model(model_type, 2048, len(classes))    # Size of the Cnn14 embedding
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(1 if model_type == "cnn" else ALL_THREADS)

    def predict(window):
        with torch.no_grad():
            x = (to_input(window) - mean) / std
            scores = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None])
            return torch.softmax(scores, dim=1)[0].numpy()
    return classes, predict


def test_clips():
    """Clips of the test split by class for the simulations"""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    return {cls: [row for row in rows if row["class"] == cls] for cls in {row["class"] for row in rows}}


class Detector:
    def __init__(self, on_message):
        self.on_message = on_message
        self.blocks = queue.Queue()    # 0.5 s blocks of audio from the microphone or from a simulated clip
        self.stream = None    # microphone stream while listening
        self.simulating = False
        self.clips = test_clips()
        self.model_name, self.classes, self.predict = None, [], None
        self.events = set(PATTERNS)    # events the user wants to be alerted of
        self.threshold = 0.6    # minimum probability to alert
        self.reset()
        threading.Thread(target=self.run, daemon=True).start()

    def reset(self):
        """Forget the audio of the last window and the last detections."""
        self.window = np.zeros(SEGMENT_LENGTH, dtype=np.float32)
        self.n_blocks, self.candidate, self.last_alert = 0, None, {}

    def status(self):
        return {"type": "status", "listening": self.stream is not None, "simulating": self.simulating,
                "models": list_models(), "model": self.model_name, "classes": self.classes,
                "events": sorted(self.events), "threshold": self.threshold, "patterns": PATTERNS}

    def handle(self, command):
        """Listen, model, settings or simulate."""
        try:
            if command["type"] == "listen":
                self.start_mic() if command["on"] else self.stop_mic()
            elif command["type"] == "model":
                # The name comes from the page: only models of the models folder are accepted
                if command["name"] not in list_models():
                    raise ValueError(f"Unknown model: {command['name']}")
                # Load first, so if it fails the previous model stays selected
                self.classes, self.predict = load_model(command["name"])
                self.model_name = command["name"]
                self.reset()
            elif command["type"] == "settings":
                self.events, self.threshold = set(command["events"]), float(command["threshold"])
            elif command["type"] == "simulate" and not self.simulating:
                if command["event"] not in self.clips:
                    raise ValueError(f"Unknown event: {command['event']}")
                clip = random.choice(self.clips[command["event"]])
                threading.Thread(target=self.play_clip, args=(clip,), daemon=True).start()
                self.on_message({"type": "simulation", "event": command["event"], "clip_id": clip["clip_id"]})
        except Exception as error:
            self.on_message({"type": "error", "message": str(error)})
        self.on_message(self.status())

    def start_mic(self):
        if self.stream is None:
            self.reset()
            self.n_blocks = -int(STARTUP_S * SAMPLE_RATE / HOP_LENGTH)
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=HOP_LENGTH,
                                         callback=lambda data, frames, time_info, status: self.blocks.put(data[:, 0].copy()))
            self.stream.start()

    def stop_mic(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def play_clip(self, clip):
        was_listening = self.stream is not None
        self.stop_mic()
        self.simulating = True
        self.reset()
        try:
            waveform, _ = librosa.load(clip["path"], sr=SAMPLE_RATE)
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            audio = np.concatenate([silence, waveform, silence])
            for start in range(0, len(audio) - HOP_LENGTH + 1, HOP_LENGTH):
                self.blocks.put(audio[start:start + HOP_LENGTH])
                time.sleep(HOP_LENGTH / SAMPLE_RATE)
        finally:
            self.simulating = False
            if was_listening:
                self.start_mic()
            self.on_message(self.status())

    def run(self):
        """Process every 0.5 s block: slide the 2 s window, classify it and send the result to the pages."""
        while True:
            block = self.blocks.get()
            self.n_blocks += 1
            if self.n_blocks <= 0:    # startup click of the microphone
                continue
            self.window = np.concatenate([self.window[len(block):], block])
            level_db = float(10 * np.log10(np.mean(block ** 2) + 1e-10))
            frame = {"type": "frame", "level_db": round(level_db, 1),
                     "spectrogram": np.round(logmel(block, 1024)).astype(int).T.tolist()}    # 64 bands x 51 frames
            if self.predict is not None:
                probabilities = self.predict(self.window)
                frame["probabilities"] = {cls: round(float(p), 3) for cls, p in zip(self.classes, probabilities)}
                self.decide(probabilities, float(10 * np.log10(np.mean(self.window ** 2) + 1e-10)))
            self.on_message(frame)

    def decide(self, probabilities, level_db):
        """Alert when the same selected event is the most likely one, above the threshold, in 2 windows in a row."""
        event = self.classes[int(np.argmax(probabilities))]
        confidence = float(np.max(probabilities))
        is_candidate = event in self.events and confidence >= self.threshold and level_db >= MIN_LEVEL_DB
        if is_candidate and self.candidate == event and time.time() - self.last_alert.get(event, 0) > COOLDOWN_S:
            self.last_alert[event] = time.time()
            self.on_message({"type": "alert", "event": event, "confidence": round(confidence, 3),
                             "pattern": PATTERNS[event], "time": time.strftime("%H:%M:%S")})
        self.candidate = event if is_candidate else None