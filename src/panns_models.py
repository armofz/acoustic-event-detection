"""PANNs models (Cnn14 and MobileNetV1) with their AudioSet weights, and a new linear layer for the classes.
Cnn14 comes from panns_inference. MobileNetV1 is not in that package: its definition is copied from
https://github.com/qiuqiangkong/audioset_tagging_cnn (pytorch/models.py, MIT license, (c) 2018-2020 Qiuqiang Kong).
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from panns_inference.models import Cnn14
from torch import nn
from torch.hub import download_url_to_file
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


# Input settings of both models, the same as our log-Mel. 527 is the number of AudioSet classes
PANNS = dict(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)


class MobileNetV1(nn.Module):
    """Copied from audioset_tagging_cnn, without the weight initialization and without the mixup"""
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        # Hides random bands of time and frequency while training, so the model does not memorize the clips
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.AvgPool2d(stride), 
                                 nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.AvgPool2d(stride),
                                 nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            conv_bn(1, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1),
            conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2),
            conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1))
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

    def forward(self, input, mixup_lambda=None):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        x = self.features(x)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return {"clipwise_output": clipwise_output, "embedding": embedding}


def load_panns(name):
    """Make Cnn14 or MobileNetV1 and load its AudioSet weights"""
    if name == "cnn14":
        model, file_name = Cnn14(**PANNS), "Cnn14_mAP=0.431.pth"
    else:
        model, file_name = MobileNetV1(**PANNS), "MobileNetV1_mAP=0.389.pth"

    # Same folder as panns_inference, so Cnn14 is not downloaded twice
    checkpoint_file = Path.home() / "panns_data" / file_name
    if not checkpoint_file.exists():
        checkpoint_file.parent.mkdir(exist_ok=True)
        # The "=" in the file name has to be written as %3D in a web address
        url = "https://zenodo.org/record/3987831/files/" + file_name.replace("=", "%3D") + "?download=1"
        download_url_to_file(url, checkpoint_file)
    model.load_state_dict(torch.load(checkpoint_file, map_location="cpu")["model"])
    return model


class PannsClassifier(nn.Module):
    """PANNs model with a new linear layer on its embedding"""
    def __init__(self, name, n_classes):
        super().__init__()
        self.base = load_panns(name)
        self.head = nn.Linear(self.base.fc1.out_features, n_classes)    # 2048 for Cnn14, 1024 for MobileNetV1

    def forward(self, x):
        # flatten(1) removes the channel dimension that train.py adds for the CNN: (batch, 1, samples) -> (batch, samples)
        return self.head(self.base(x.flatten(1))["embedding"])