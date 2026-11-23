# Classes (13.10.26)

Five classes will be used, taken from ESC-50 and FSD50K:
- **'alarm'**: generic alarm
- **'phone'**: phone ringtone or phone alarm
- **'door_knock_bell'**: door knock or doorbell
- **'glass_breaking'**: glass breaking
- **'siren'**: emergency services siren

In dataset v2, 'phone' is removed and new classes are added (see Dataset v2)

## Problems and solutions

| Reason | Description | Solution |
| --- | --- | --- |
| Vocal sounds | Crying, coughing and laughing are voice sounds, and the project only detects non-voice events | Exclude them |
| Parent labels in FSD50K | Audio clips have the labels of its parent classes, e.g 'Doorbell' and 'Siren' have the label 'Alarm' | Check first the specific labels, and use 'Alarm' only when none of the rest are present |
| No emergency alarm label | Neither dataset has a label for smoke or fire alarms | Define 'alarm' as a generic alarm and leave emergency alarms for a later version |
| Noisy 'Alarm' | Listening tests showed that most clips were not clean alarms | Rank the clips by the quality ratings of FSD50K and remove silent clips |
| Shared recordings | Some clips in ESC-50 and FSD50K come from the same Freesound recording | Keep only one copy of each recording |
| Official split not usable | The official FSD50K validation set is too small for the specified classes | Use 70/15/15 split for train, validation and test, balanced by class |
| Data leakage | In FSD50K, no uploader appears in both the dev and eval sets | Keep all clips from the same uploader in the same split |
| Unbalanced classes | Different classes have different ammount of clips | Limit every class to 160 clips |


# Dataset v1 (22.10.26)

The dataset has 800 clips, 160 per class ('door_knock_bell' has 80 'knock' and 80 'doorbell'). 

The split is 70/15/15 for train, validation and test (112/24/24 clips per class). The list of clips is in 'data/splits/manifest_v1.csv', built by 'src/build_dataset_v1.py'.

| Reason | Description | Solution |
| --- | --- | --- |
| Duplicate recordings | 94 FSD50K clips come from the same recording as an ESC-50 clip. | Keep the ESC-50 copy and drop the FSD50K clip |
| Silent clips | 4 clips were silent (3 'alarm', 1 'siren'). | Drop clips with a peak below 1 % of full scale and take the next best clip |
| Mixed 'alarm' class | In a listening test of 8 'alarm' clips, only 5 were typical alarms. Many 'Alarm' clips in FSD50K seem to be sound effects, not real recordings | Keep the class in version 1, and change in future ones |

## Spectrograms (25.10.26)

Each clip becomes one log-Mel spectrogram of the 2 loudest seconds (64 bands x 201 frames), with the same settings as PANNs Cnn14: 32 kHz, window of 1024 samples, hop of 320, 64 bands, 50 Hz to 14 kHz.


### Window length comparison (26.10.26)

Three log-Mel variants are generated with a window of 512, 1024 (PANNs setting) and 2048 samples (16, 32 and 64 ms at 32 kHz), built by 'src/make_logmel.py --n-fft'. The hop (320 samples), the 64 bands and the 50 Hz to 14 kHz range stay fixed, so any difference comes from the window length alone. All variants use the same 2 s segment of each clip.


### Representation and model comparison (31.10.26)

A constant-Q transform (CQT) is built by 'src/make_cqt.py' to compare it with the log-Mel: 64 bins, 8 per octave from 50 Hz, with the same hop (320) and 2 s segment as the log-Mel, so every representation has the same size (64 x 201). It reaches 11.7 kHz and not 14 kHz, with the usual 12 bins per octave it would only reach 1.9 kHz.

Every model is trained with seeds 1, 2 and 3 and compared by the mean macro F1 on validation (120 clips). The CNN with seed 666 got 0.751, but seeds 1 to 3 got 0.664 to 0.728, so a single seed is never enough. The test split is kept for the final evaluation.

| Model | Input | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | --- | --- | --- | --- | --- |
| CNN from scratch | log-Mel, window 512 | 0.713 | 0.702 | 0.731 | 0.715 |
| CNN from scratch | log-Mel, window 1024 | 0.704 | 0.664 | 0.728 | 0.699 |
| CNN from scratch | log-Mel, window 2048 | 0.700 | 0.706 | 0.692 | 0.699 |
| CNN from scratch | CQT | 0.707 | 0.620 | 0.701 | 0.676 |
| Frozen Cnn14 + linear layer | PANNs embeddings | 0.866 | 0.858 | 0.867 | 0.864 |

The difference between the representations (means from 0.676 to 0.715) is smaller than the variation between seeds of the same representation, so there is no clear winner. The log-Mel with window 1024 is kept as the input because it is the PANNs setting. Regarding the model, the frozen Cnn14 improves the CNN by 0.165 and varies much less between seeds.


# Dataset v2 (10.11.26)

The models of v1 always choose one of their classes, even with silence or normal home sounds (with silence they say 'phone'), so in a real home they would raise false alarms. Version 2 adds the sounds that should not alert.

The dataset has 1920 clips in 17 classes (619 from ESC-50 and 1301 from FSD50K). It uses the same steps and split as v1 (70/15/15, uploaders never in two splits), only the classes and rules change. The 'class' column is the class the model learns, and 'subclass' keeps the original sound, so the classes can be regrouped later without rebuilding the dataset. The detector decides which classes alert, and the user can change it in the app.

Alert classes (they alert by default, 160 clips each):
- **'alarm'**: generic alarm
- **'siren'**: emergency services siren
- **'glass_breaking'**: glass breaking
- **'door_knock_bell'**: door knock or doorbell

Classes that can be activated (they do not alert by default, 160 clips each):
- **'door_slam'**: door slam
- **'crying'**: baby or adult crying
- **'scream_shout'**: scream, shout or yell

Home sounds (they never alert, 80 clips each):
- **'speech'**: speech or laughter
- **'cough_sneeze'**: cough or sneeze
- **'music'**: music or singing
- **'clinks'**: dishes, cutlery or coins
- **'water_cooking'**: tap or sink, dripping, toilet flush, frying or boiling
- **'appliances'**: vacuum cleaner, washing machine, fan, printer or microwave
- **'typing_paper'**: typing, writing, paper or zipper
- **'pets'**: dog or cat
- **'footsteps_furniture'**: footsteps, drawers and cupboards, or door creak
- **'outdoors'**: rain, wind or traffic

| Reason | Description | Solution |
| --- | --- | --- |
| Home sounds | ESC-50 and FSD50K have many more labels than the ones needed | Cover the frequent sounds of a home that should not alert. Body sounds, construction and outdoor engines are left out |
| Sounds that may alert in the future | Training door slams, crying or screams as 'other' would teach the model to ignore them | Train them as their own classes that do not alert by default, so they can be activated later |
| Clips close to an alert class | Clinking glasses, ringtones, bells, beeps, explosions or ticking clocks would teach the model to ignore sounds that are almost alerts | FSD50K clips with any of these labels only go to the alert classes |
| Clips with several labels | A clip can have 'Cough' and 'Speech', or 'Singing' and 'Music' | The most specific label wins |
| 'phone' removed | FSD50K 'Telephone' mixes rings, dial tones and keypad beeps, ESC-50 has no phone class, and in a first version of v2 'phone' sent multitude of false positives. Besides, a mobile phone already notifies an user when it rings | Remove the class |
