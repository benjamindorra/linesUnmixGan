"""Train a UNet based GAN"""

from modelUNet import Gen, Discrim
from customDataset import customDataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os.path

def loadDatasets():
    """Load datasets"""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    baseDir = os.path.dirname(__file__)

    augment = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1,0.1),
            scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    #Mean and std must be computed from the training dataset with
    #customDataset.py
    normInput = transforms.Normalize(mean=[0.9700, 0.9700, 0.9700],
                                     std=[0.1505, 0.1505, 0.1505])
    normTarget = transforms.Normalize(mean=[0.0154, 0.0154, 0.0300], std=
                                      [0.1092, 0.1062, 0.1505])
    basedir = os.path.dirname(__file__)
    trainset = customDataset(root_dir=os.path.join(
            basedir,'database/trainSetHWLines'), input_dir='input',
            target_dir='target', transform=augment, normInput=normInput,
            normTarget=normTarget)
    valset = customDataset(root_dir=os.path.join(
            basedir, 'database/validSetHWLines'), input_dir='input',
            target_dir='target', transform=preprocess, normInput=normInput,
            normTarget=normTarget)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
            shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2,
            shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, valloader

## Train
class trainGan():
    def __init__(self, nb_epochs=60, lr=3e-4, savefileGen="netWeights/genUNetHW.pth",
            savefileDiscrim="netWeights/discrimUNetHW.pth", discrimTrainPeriod=2,
            displayPeriod=100):
         #Input/output
         basedir = os.path.dirname(__file__)
         self.savefileGen = os.path.join(basedir, savefileGen)
         self.savefileDiscrim = os.path.join(basedir, savefileDiscrim)
         self.displayPeriod = displayPeriod
         self.graphFile = os.path.join(basedir, 'loss_curve.png')
         self.generatedFile = os.path.join(basedir, 'generated_image.png')
         self.targetFile = os.path.join(basedir, 'target_image.png')
         self.transformToImage = transforms.ToPILImage()
         mean, std=[0.0154, 0.0154, 0.0300],[0.1092, 0.1062, 0.1505]
         self.unNormalize = transforms.Normalize(mean=[-m/d for m,d in
             zip(mean,std)], std=[1.0/d for d in std])
         #Networks
         self.gen = Gen().to(device)
         try:
            self.gen.load_state_dict(torch.load(self.savefileGen))
         except:
             print("No weights found, generator initialized")
         self.discrim = Discrim().to(device)
         try:
            self.discrim.load_state_dict(torch.load(self.savefileDiscrim))
         except:
             print("No weights found, discriminator initialized")
         #Datasets
         self.trainloader, self.valloader = loadDatasets()
         #Training parameters
         self.criterionGan = torch.nn.MSELoss()
         self.criterionL1 = torch.nn.L1Loss()
         self.optimizerGen = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5,
             0.999))
         self.optimizerDiscrim = torch.optim.Adam(self.discrim.parameters(), lr=lr,
                 betas=(0.5,0.999))
         self.nb_epochs = nb_epochs #number of epochs
         self.discrimTrainPeriod = discrimTrainPeriod
         self.lambda_L1 = 0.01
         #Statistics
         self.train_loss = 0.
         self.gan_loss = 0.
         self.loss_discrim = 0.
         self.gan_loss_list = []
         self.train_loss_list = []
         self.val_loss_list = []

    def trainDiscrim(self, inputs, real):
        self.optimizerDiscrim.zero_grad()
        self.optimizerGen.zero_grad()
        self.discrim.train()
        self.gen.eval()
        with torch.no_grad():
            fake = self.gen(inputs)
        estimFake = self.discrim(inputs, fake.detach())
        fakeTarget = torch.zeros_like(estimFake)
        lossDiscrimFake = self.criterionGan(estimFake, fakeTarget)
        estimReal = self.discrim(inputs, real)
        realTarget = torch.ones_like(estimReal)
        lossDiscrimReal = self.criterionGan(estimReal, realTarget)
        lossDiscrim = (lossDiscrimFake+lossDiscrimReal) * 0.5
        lossDiscrim.backward()
        self.optimizerDiscrim.step()
        #statistics
        self.loss_discrim += lossDiscrim.item()


    def trainGen(self, inputs, real):
        self.optimizerGen.zero_grad()
        self.optimizerDiscrim.zero_grad()
        self.gen.train()
        self.discrim.train()
        fake = self.gen(inputs)
        #with torch.no_grad():
        estimFake = self.discrim(inputs, fake)
        #Try to fool the discriminator to see the gen output as real
        realTarget = torch.ones_like(estimFake)
        lossGenGan = self.criterionGan(estimFake, realTarget)
        #Make the output look like the training set
        lossGenL1 = self.criterionL1(fake, real)
        lossGen = self.lambda_L1 * lossGenL1 + lossGenGan
        lossGen.backward()
        self.optimizerGen.step()
        #statistics
        self.train_loss += lossGenL1.item()
        self.gan_loss += lossGenGan.item()

    def __call__(self):
        """Train the generator and discriminator"""
        #Losses and optimizers
        for epoch in range(self.nb_epochs):  # loop over the dataset multiple times
            self.train_loss = 0.0
            self.gan_loss = 0.0
            self.loss_discrim = 0.0
            val_loss = 0.0
            nb_data = 0.
            nb_data_val = 0.
            for i, data in enumerate(self.trainloader, 0):
                # get the batch; data is a list of [inputs, labels]
                inputs, real = data
                inputs, real = inputs.to(device), real.to(device)
                if i%self.discrimTrainPeriod==0:
                    self.trainDiscrim(inputs, real)
                else:
                    self.trainGen(inputs, real)
                    nb_data += 1.
                #occasionnally save an example target/generated
                if i%self.displayPeriod==0:
                    self.gen.eval()
                    real = self.unNormalize(real[0,:,:,:].detach().cpu())
                    self.transformToImage(real).save(self.targetFile)
                    fake = self.gen(inputs)
                    fake = self.unNormalize(fake[0,:,:,:].detach().cpu())
                    self.transformToImage(fake).save(self.generatedFile)

            self.gen.eval()
            for i, data in enumerate(self.valloader, 0):
                with torch.no_grad():
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, real = data
                    inputs, real = inputs.to(device), real.to(device)
                    #compute L1 loss
                    fake = self.gen(inputs)
                    lossGenL1 = self.criterionL1(fake, real)
                    #statistics
                    val_loss += lossGenL1.item()
                    nb_data_val += 1.
            self.gan_loss = self.gan_loss / nb_data
            self.train_loss = self.train_loss / nb_data
            self.loss_discrim = self.loss_discrim / nb_data
            val_loss = val_loss / nb_data_val
            self.gan_loss_list.append(self.gan_loss)
            self.train_loss_list.append(self.train_loss)
            self.val_loss_list.append(val_loss)
            print("Epoch ", epoch, "; train loss = ", self.train_loss,
                    "; val loss = ", val_loss, "; gan loss = ", self.gan_loss,
                    "; loss discrim = ", self.loss_discrim)

            plt.plot(range(len(self.train_loss_list)), self.train_loss_list,
                    self.val_loss_list, self.gan_loss_list)
            plt.xlabel("Epochs")
            plt.ylabel("Generator Loss")
            plt.savefig(self.graphFile)
            #save the weights
            torch.save(self.gen.state_dict(), self.savefileGen)
            torch.save(self.discrim.state_dict(), self.savefileDiscrim)

if __name__=="__main__":
    #assign correct device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)
    train = trainGan()
    train()

