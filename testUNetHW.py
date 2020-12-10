"""Test the FPN generator for handwritten test unmixing"""

from modelUNet import Gen
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os

baseDir = os.path.dirname(__file__)
saveFileGen = "netWeights/genUNetHW.pth" #Relative path to weights
saveFileGen = os.path.join(baseDir, saveFileGen)
#Relative path to input and output directories
dirin = "database/testSetHWLines/input/"
dirout = "database/testSetHWLines/result/"
dirin = os.path.join(baseDir, dirin)
dirout = os.path.join(baseDir, dirout)
if not os.path.exists(dirout):
    os.mkdir(dirout)

#Normalize input
mean, std = [0.9700, 0.9700, 0.9700], [0.1505, 0.1505, 0.1505]
toimage = transforms.ToPILImage()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

#un-normalize output
mean, std = [0.0154, 0.0147, 0.0300], [0.1092, 0.1062, 0.1505]
unNormalize = transforms.Normalize(mean=[-m/d for m,d in
    zip(mean,std)], std=[1.0/d for d in std])

#assign correct device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('cuda:0')
else:
    device = torch.device('cpu')
    print('cpu')

#Networks
#Max downsampling of the model
netDown=32
gen = Gen().to(device)
try:
    if torch.cuda.is_available():
      gen.load_state_dict(torch.load(saveFileGen))
    else:
      gen.load_state_dict(torch.load(saveFileGen, map_location=torch.device('cpu') ))
except:
    print("No weights found, generator initialized")


imgs_names = os.listdir(dirin)
for img in imgs_names:
    imginit = Image.open(os.path.join(dirin,img)).convert("RGB")
    size = imginit.size
    #Resize so it is compatible
    imginit=imginit.resize((size[0]+netDown-size[0]%netDown,size[1]+netDown-size[1]%netDown),resample=Image.BILINEAR)
    #Light blur leads to a smoother result
    imginit = imginit.filter(ImageFilter.GaussianBlur(radius=2))
    imgin = preprocess(imginit).unsqueeze(0).to(device)
    gen.eval()
    with torch.no_grad():
        imgout = gen(imgin)
    imgout = unNormalize(imgout[0,:,:,:].detach().cpu())
    imgout = toimage(torch.flatten(imgout, start_dim=0, end_dim=1))
    #Back to original size
    imgout = imgout.resize((size[0], 3*size[1]), resample=Image.BILINEAR)
    imgout.save(os.path.join(dirout,img))

