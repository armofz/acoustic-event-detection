# Classes (13.10.26)

Five classes will be used, taken from ESC-50 and FSD50K:
- **'alarm'**: generic alarm
- **'phone'**: phone ringtone or phone alarm
- **'door_knock_bell'**: door knock or doorbell
- **'glass_breaking'**: glass breaking
- **'siren'**: emergency services siren

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