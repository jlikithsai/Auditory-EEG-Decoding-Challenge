import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm 

class ResNet34EEG(nn.Module):
    def __init__(self):
        super(ResNet34EEG, self).__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)

class ResNet34Mel(nn.Module):
    def __init__(self):
        super(ResNet34Mel, self).__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)


device = 'cuda'

resnet_eeg = ResNet34EEG().to(device)
resnet_mel = ResNet34Mel().to(device)
