import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class PyramidLayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False)

    def forward(self, x, prev=None):
        x = self.conv(x)
        x += self.upsample(prev)
        return x

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class convUpsamples(nn.Module):
    def __init__(self, in_channels, out_channels, nUps):
        super().__init__()
        convs = [ConvBnRelu(in_channels, out_channels, kernel_size=3,padding=1)]
        if nUps>=1:
            convs.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=False))
        for i in range(1,nUps):
            convs.append(ConvBnRelu(out_channels, out_channels,
                                    kernel_size=3, padding=1))
            convs.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False))
        self.convs = nn.Sequential(*convs)
    def forward(self, x):
        x = self.convs(x)
        return x

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        decoderLayers = 128
        segLayers = 64
        resnet18 = torchvision.models.resnet18(pretrained=False, progress=False)
        featureExtractor = nn.Sequential(*list(resnet18.children())[:-2])
        self.layer1 = featureExtractor[:4]
        self.layer2 = featureExtractor[5]
        self.layer3 = featureExtractor[6]
        self.layer4 = featureExtractor[7]

        self.pBlock4 = nn.Conv2d(512, decoderLayers, kernel_size=1)
        self.convUp4 = convUpsamples(decoderLayers,segLayers, nUps=3)
        self.pBlock3 = PyramidLayerBlock(256, decoderLayers)
        self.convUp3 = convUpsamples(decoderLayers,segLayers, nUps=2)
        self.pBlock2 = PyramidLayerBlock(128, decoderLayers)
        self.convUp2 = convUpsamples(decoderLayers, segLayers, nUps=1)
        self.pBlock1 = PyramidLayerBlock(64, decoderLayers)
        self.convUp1 = convUpsamples(decoderLayers, segLayers, nUps=0)
        self.dropout = nn.Dropout2d(p=0.5)
        self.finalUpsample = nn.Upsample(scale_factor=4, mode ='bilinear',
                                         align_corners=False)
        self.normLayer = nn.BatchNorm2d(4*segLayers)
        self.channelReduction = ConvBnRelu(4*segLayers, 3, kernel_size=1)

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        block4 = self.pBlock4(l4)
        block3 = self.pBlock3(l3, block4)
        block2 = self.pBlock2(l2, block3)
        block1 = self.pBlock1(l1, block2)

        block4 = self.convUp4(block4)
        block3 = self.convUp3(block3)
        block2 = self.convUp2(block2)
        block1 = self.convUp1(block1)

        x = torch.cat((block4, block3, block2, block1), dim=1)
        x = self.normLayer(x)
        x = self.dropout(x)
        x = self.finalUpsample(x)
        x = self.channelReduction(x)
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
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=2,
                stride=2)

        self.normLayer = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, input, x):
        x = torch.cat((input, x), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.normLayer(x)
        x = self.conv5(x)
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
    #cv2.imshow("gen",image)
    #cv2.waitKey(10000)
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
    #cv2.imshow("gen",show)
    #cv2.waitKey(10000)
    y = discrim(ten,x)
    print("output shape:", y.shape)
    print("Output vector:", y)
    print("Discriminator OK")


