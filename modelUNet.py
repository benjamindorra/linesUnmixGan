import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, pool=True):
        super().__init__()
        self.pool = pool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                padding=padding)
        self.normLayer = nn.BatchNorm2d(out_channels)
        self.poolLayer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.normLayer(x)
        skip = x
        if self.pool:
            x = self.poolLayer(x)
        return x, skip

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, dropout=False):
        super().__init__()
        self.upConv =  nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                padding=padding)
        self.normLayer = nn.BatchNorm2d(out_channels)
        self.dropout = dropout
        self.dropoutLayer = nn.Dropout2d(p=0.5)

    def forward(self, x, skip):
        x = self.upConv(x)
        x = torch.cat((x, skip), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.dropout:
            x = self.normLayer(x)
            x = self.dropoutLayer(x)
        return x

class Gen(nn.Module):
    def __init__(self, in_channels=3, exit_channels=3, stages=4):
        super().__init__()
        self.stages=stages
        self.downConvs = nn.ModuleList()
        self.upConvs = nn.ModuleList()
        out_channels = 64
        self.outConv = nn.Conv2d(in_channels=out_channels,
                out_channels=exit_channels,
                kernel_size=1)
        for i in range(stages):
            self.downConvs.append(DownConv(in_channels, out_channels))
            if i==(stages-1):
                dropout = True
            else:
                dropout = False
            self.upConvs.append(UpConv(2*out_channels, out_channels,
                dropout=dropout))
            in_channels = out_channels
            out_channels = out_channels*2
        self.lastDownConv = DownConv(in_channels, out_channels, pool=False)

    def forward(self, x):
        skips = [None]*self.stages
        for (i, downConv) in enumerate(self.downConvs):
            x, skips[i]  = downConv(x)
        x, _ = self.lastDownConv(x)
        for (upConv, skip) in zip(reversed(self.upConvs), reversed(skips)):
            x = upConv(x,skip)
        x = self.outConv(x)
        return x

class Discrim(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=2,
                stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2,
                stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2,
                stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2)
        self.normLayer = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
        #Convolutional classifier inspired by SqueezeNet
        #self.classifier = nn.Sequential(
        #        nn.Dropout(p=0.5),
        #        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
        #        nn.ReLU(inplace=True),
        #        nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #)

    def forward(self, input, x):
        x = torch.cat((input, x), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.normLayer(x)
        x = self.conv5(x)
        #x = self.classifier(x)
        #return x.view(x.size(0), -1)
        return x


if __name__=="__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    gen = Gen().to(device)
    discrim = Discrim().to(device)
    image = cv2.imread("database/trainingSetLines/input/00000.png")
    #image = cv2.resize(image, (512,512), interpolation=2)
    cv2.imshow("gen",image)
    cv2.waitKey(10000)
    transform = torchvision.transforms.ToTensor()
    ten = transform(image).unsqueeze(0)
    ten = ten.to(device)
    print("Input shape:", ten.shape)
    x = gen(ten)
    print("Generator output shape:", x.shape)
    if ten.shape==x.shape:
        print("Generator OK")
    show = x.detach().to('cpu')
    show = torch.flatten(show, start_dim=0, end_dim=1)
    show = show.permute(1,2,0).numpy()
    cv2.imshow("gen",show)
    cv2.waitKey(10000)
    y = discrim(ten,x)
    print("output shape:", y.shape)
    print("Output vector:", y)
    print("Discriminator OK")


