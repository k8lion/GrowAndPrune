import sys
import os
sys.path.append(os.path.expanduser("~/repos/NeurOps/pytorch"))
from neurops import *



class ModMLP(ModSequential):
    def __init__(self, inputs, hidden, outputs, track_activations=False):
        super().__init__(
            ModLinear(inputs, hidden),
            ModLinear(hidden, outputs, nonlinearity=''),
            track_activations=track_activations,
        )

    def forward(self, x):
        return super().forward(x).flatten()


class ModVGG11(ModSequential):
    #https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
    #https://download.pytorch.org/models/vgg11-8a719046.pth
    def __init__(self, num_classes: int = 10):
        super().__init__(
            ModConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, postpool=nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, postpool=nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ModConv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, postpool=nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, postpool=nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            ModConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, postpool=nn.MaxPool2d(2,2)),
            ModLinear(512, 4096, preflatten=True),
            ModLinear(4096, 4096),
            ModLinear(4096, num_classes, nonlinearity=''),
            track_activations=True,
        )
