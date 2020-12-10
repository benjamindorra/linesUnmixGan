from model import Gen
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

#Normalize input
mean, std=[0.9880, 0.9880, 0.9880], [0.0860, 0.0860, 0.0860]
toimage = transforms.ToPILImage()
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

#un-normalize output
mean, std=[0.0063, 0.0063, 0.9791],[0.0613, 0.0612, 0.1262]
unNormalize = transforms.Normalize(mean=[-m/d for m,d in
    zip(mean,std)], std=[1.0/d for d in std])

#assign correct device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#Networks
saveFileGen = "gen.pth"
saveFileDiscrim = "discrim.pth"
gen = Gen().to(device)
try:
    gen.load_state_dict(torch.load(saveFileGen))
except:
    print("No weights found, generator initialized")
#discrim = Discrim().to(device)
#try:
#    discrim.load_state_dict(saveFileDiscrim)
#except:
#    print("No weights found, discriminator initialized")

dirin = "database/validSetLines/input/"
dirout = "database/validSetLines/result/"
imMax = 100
for i in range(imMax):
    img = f'{i:05}'+'.png'

    imginit = Image.open(dirin+img)
    imgin = preprocess(imginit).unsqueeze(0).to(device)
    gen.eval()
    with torch.no_grad():
        imgout = gen(imgin)
    imgout = unNormalize(imgout[0,:,:,:].detach().cpu())
    imgout = toimage(torch.flatten(imgout, start_dim=0, end_dim=1))
    imgout.save(dirout+img)

