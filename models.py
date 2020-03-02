import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

class UnetTransfer(nn.Module):
  def __init__(self):
    super(UnetTransfer, self).__init__()
    unet = smp.Unet('resnet18', encoder_weights='imagenet')
    for param in unet.parameters():
      param.requires_grad = False
    self.encoder = unet.encoder
    self.avgpool = nn.AdaptiveAvgPool2d((2,2))
    self.linear_layers = nn.Sequential(
    nn.Linear(in_features=512*2*2, out_features=1024),
    nn.ELU(),
    nn.Linear(in_features=1024, out_features=1024),
    nn.Linear(in_features=1024, out_features=1)
)
  def forward(self, input):
    input = input.view(input.size(0), 3, 70, 320)
    output = self.encoder(input)[-1]
    output = self.avgpool(output)
    output = output.view(output.size(0), -1)
    output = self.linear_layers(output)
    return output


class NetworkDense(nn.Module):
  # DeepPicar by NVIDIA

    def __init__(self):
        super(NetworkDense, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output