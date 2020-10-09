from efficientnet_pytorch import EfficientNet
from fastai.vision.models.xresnet import xse_resnet50, xresnet18
from fastai.vision.models import densenet121
import torch.nn as nn

efficient_b6 = EfficientNet.from_name('efficientnet-b6')
efficient_b6._fc = nn.Linear(2304, 2)
efficient_b6.name = 'efficientnet-b6'

efficient_b7 = EfficientNet.from_name('efficientnet-b7')
efficient_b7._fc = nn.Linear(2560, 2)
efficient_b7.name = 'efficientnet-b7'

xse_resnet50 = xse_resnet50(n_out=2)
xse_resnet50.name = 'xse_resnet50'

densenet121 = densenet121()
densenet121.name = 'densenet121'

tst_model = xresnet18(n_out=2)
tst_model.name = 'tst'