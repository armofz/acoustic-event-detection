"""Build dataset v2 (17 classes) from ESC-50 and FSD50K: 4 alert classes (v1 without 'phone'), 3 extra classes that can be
activated and 10 classes of normal home sounds that never alert.
Writes data/splits/manifest_v2.csv: one row per selected clip with its class, subclass (original sound) and split.
It uses the same steps as v1 (build_dataset_v1.py), only the classes and rules change.
Run from the repo root: python src/build_dataset_v2.py
"""
import random
from pathlib import Path

from build_dataset_v1 import SEED, assign_splits, load_esc50, load_fsd50k, load_uploaders, print_summary, select, write_manifest


OUT_FILE = Path("data/splits/manifest_v2.csv")

CLIPS_PER_CLASS = {
    # Alert classes, as in v1 without 'phone'
    ("alarm", None): 160, ("siren", None): 160, ("glass_breaking", None): 160,
    ("door_knock_bell", "knock"): 80, ("door_knock_bell", "doorbell"): 80,
    # Can be activated in the app
    ("door_slam", "slam"): 160, ("crying", "crying"): 160, ("scream_shout", "scream"): 80, ("scream_shout", "shout"): 80,
    # Never alert
    ("speech", "speech"): 40, ("speech", "laughter"): 40,
    ("cough_sneeze", "cough"): 40, ("cough_sneeze", "sneeze"): 40,
    ("music", "music"): 40, ("music", "singing"): 40,
    ("clinks", "dishes"): 27, ("clinks", "cutlery"): 27, ("clinks", "coin"): 26,
    ("water_cooking", "tap_sink"): 20, ("water_cooking", "drip"): 20, ("water_cooking", "toilet_flush"): 20,
    ("water_cooking", "frying_boiling"): 20,
    ("appliances", "vacuum_cleaner"): 16, ("appliances", "washing_machine"): 16, ("appliances", "fan"): 16,
    ("appliances", "printer"): 16, ("appliances", "microwave"): 16,
    ("typing_paper", "typing"): 20, ("typing_paper", "writing"): 20, ("typing_paper", "paper"): 20, ("typing_paper", "zipper"): 20,
    ("pets", "dog"): 40, ("pets", "cat"): 40,
    ("footsteps_furniture", "footsteps"): 27, ("footsteps_furniture", "drawer_cupboard"): 27,
    ("footsteps_furniture", "door_creak"): 26,
    ("outdoors", "rain"): 27, ("outdoors", "wind"): 27, ("outdoors", "traffic"): 26,
}

ALERT_CLASSES = {"alarm", "siren", "glass_breaking", "door_knock_bell"}
# 'phone' is removed: FSD50K 'Telephone' mixes rings, dial tones and keypad beeps, and a phone already notifies a deaf
# user. Its clips also have the 'Alarm' label, so they are dropped instead of ending up in 'alarm'
DROPPED_LABELS = {"Telephone", "Ringtone"}

ESC_MAP = {"clock_alarm": ("alarm", None), "door_wood_knock": ("door_knock_bell", "knock"),
           "glass_breaking": ("glass_breaking", None), "siren": ("siren", None),
           "crying_baby": ("crying", "crying"), "laughing": ("speech", "laughter"),
           "coughing": ("cough_sneeze", "cough"), "sneezing": ("cough_sneeze", "sneeze"),
           "pouring_water": ("water_cooking", "tap_sink"), "water_drops": ("water_cooking", "drip"),
           "toilet_flush": ("water_cooking", "toilet_flush"), "vacuum_cleaner": ("appliances", "vacuum_cleaner"),
           "washing_machine": ("appliances", "washing_machine"), "keyboard_typing": ("typing_paper", "typing"),
           "mouse_click": ("typing_paper", "typing"), "dog": ("pets", "dog"), "cat": ("pets", "cat"),
           "footsteps": ("footsteps_furniture", "footsteps"), "door_wood_creaks": ("footsteps_furniture", "door_creak"),
           "rain": ("outdoors", "rain"), "wind": ("outdoors", "wind"),}

FSD_RULES = [("Doorbell", "door_knock_bell", "doorbell"), ("Knock", "door_knock_bell", "knock"),
             ("Siren", "siren", None), ("Alarm", "alarm", None), ("Shatter", "glass_breaking", None),
             ("Slam", "door_slam", "slam"), ("Crying_and_sobbing", "crying", "crying"),
             ("Screaming", "scream_shout", "scream"), ("Shout", "scream_shout", "shout"), ("Yell", "scream_shout", "shout"),
             ("Cough", "cough_sneeze", "cough"), ("Sneeze", "cough_sneeze", "sneeze"),
             ("Laughter", "speech", "laughter"), ("Singing", "music", "singing"),
             ("Dishes_and_pots_and_pans", "clinks", "dishes"), ("Cutlery_and_silverware", "clinks", "cutlery"),
             ("Coin_(dropping)", "clinks", "coin"),
             ("Water_tap_and_faucet", "water_cooking", "tap_sink"), ("Sink_(filling_or_washing)", "water_cooking", "tap_sink"),
             ("Drip", "water_cooking", "drip"), ("Toilet_flush", "water_cooking", "toilet_flush"),
             ("Frying_(food)", "water_cooking", "frying_boiling"), ("Boiling", "water_cooking", "frying_boiling"),
             ("Mechanical_fan", "appliances", "fan"), ("Printer", "appliances", "printer"),
             ("Microwave_oven", "appliances", "microwave"),
             ("Typing", "typing_paper", "typing"), ("Writing", "typing_paper", "writing"),
             ("Crumpling_and_crinkling", "typing_paper", "paper"), ("Tearing", "typing_paper", "paper"),
             ("Zipper_(clothing)", "typing_paper", "zipper"),
             ("Bark", "pets", "dog"), ("Meow", "pets", "cat"),
             ("Walk_and_footsteps", "footsteps_furniture", "footsteps"),
             ("Drawer_open_or_close", "footsteps_furniture", "drawer_cupboard"),
             ("Cupboard_open_or_close", "footsteps_furniture", "drawer_cupboard"),
             ("Sliding_door", "footsteps_furniture", "drawer_cupboard"),
             ("Rain", "outdoors", "rain"), ("Wind", "outdoors", "wind"),
             ("Traffic_noise_and_roadway_noise", "outdoors", "traffic"), ("Car_passing_by", "outdoors", "traffic"),
             ("Speech", "speech", "speech"), ("Music", "music", "music"),]

# Sounds that alert or are too close to an alert class, a clip with any of them only goes to an alert class
EXCLUDED_LABELS = {"Alarm", "Siren", "Telephone", "Ringtone", "Doorbell", "Knock", "Tap", "Shatter", "Glass", "Crack",
                   "Bell", "Bicycle_bell", "Church_bell", "Chime", "Wind_chime", "Cowbell", "Gong",
                   "Vehicle_horn_and_car_horn_and_honking", "Explosion", "Fireworks", "Gunshot_and_gunfire",
                   "Thunder", "Thunderstorm", "Crash_cymbal", "Clock", "Tick", "Tick-tock"}


def main():
    rng = random.Random(SEED)
    uploader = load_uploaders()
    esc, skip_ids = load_esc50(uploader, rng, ESC_MAP)
    fsd, duplicates = load_fsd50k(uploader, skip_ids, rng, FSD_RULES, EXCLUDED_LABELS, ALERT_CLASSES, DROPPED_LABELS)
    selected, dropped = select(esc + fsd, CLIPS_PER_CLASS)
    assign_splits(selected, rng)
    write_manifest(selected, OUT_FILE)
    print_summary(selected, duplicates, dropped, OUT_FILE)


if __name__ == "__main__":
    main()