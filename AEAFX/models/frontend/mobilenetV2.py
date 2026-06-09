import torch
from transformers import MobileNetV2Config, MobileNetV2Model
from .main import Frontend
import nnAudio.features

conf = MobileNetV2Config(num_channels=1, image_size=109)


class MobileNetV2CQT(Frontend):
    def __init__(
        self,
        samplerate,
        n_bins: int = 113,
    ):
        super().__init__()

        self.cqt = nnAudio.features.CQT1992v2(n_bins=n_bins, sr=samplerate)

        conf = MobileNetV2Config(num_channels=1, image_size=n_bins)

        self.mobileNet = MobileNetV2Model(conf)

    
    def forward(self, x):

        x = self.cqt(self)
        x = self.mobileNet(self)

        return x