import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os.path
import glob
import matplotlib.pyplot as plt
import random

#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
class customDataset(torch.utils.data.Dataset):

    def __init__(self,root_dir, input_dir, target_dir, transform,
            normInput=None, normTarget=None):
        self.input_name_frame = glob.glob(os.path.join(root_dir, input_dir, "*"))
        self.target_name_frame = glob.glob(os.path.join(root_dir, target_dir,
        "*"))
        self.transform = transform
        self.normInput = normInput
        self.normTarget = normTarget

    def __len__(self):
        return len(self.input_name_frame)

    def __getitem__(self, idx):
        input_name = self.input_name_frame[idx]
        input_image = Image.open(input_name)
        #use the same random transformations to input and target
        #https://github.com/pytorch/vision/issues/9
        seed = random.randint(0,2**32) #create random seed
        random.seed(seed) #use random seed
        input_image = self.transform(input_image)
        target_name = self.target_name_frame[idx]
        #Converting is necessary because some images seem to be open as
        #grayscale, for an unknown reason
        target_image =  Image.open(target_name).convert("RGB")
        random.seed(seed) #reuse the same seed (same operations)
        target_image = self.transform(target_image)
        if self.normInput:
            input_image=self.normInput(input_image)
        if self.normTarget:
            target_image=self.normTarget(target_image)
        sample = (input_image,  target_image)

        return sample

#Test the dataset code
if __name__=="__main__":
    root_dir="./database/validSetHWLines"
    input_dir="input"
    target_dir="target"
    transform=transforms.ToTensor()
    testDs = customDataset(root_dir=root_dir, input_dir=input_dir,
            target_dir=target_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(testDs, batch_size=4,
            shuffle=True, num_workers=4)
    print("Data loading OK")

    # Helper function to show a batch
    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, targets_batch = sample_batched
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = \
        torchvision.utils.make_grid(torch.cat((images_batch,targets_batch),3))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    print("Shape of the 3 first batches:")
    for i_batch, sample_batched in enumerate(dataloader):
        images, targets = sample_batched
        print(i_batch, images.size(), targets.size())
        print(i_batch, type(images), type(targets))
        # observe 4th batch and stop.
        if i_batch == 3:
            print("You should now see a patch. Please check if the result is correct.")
            plt.figure()
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
    #Compute the mean and std for the dataset (WARNING: heavy
    # on memory. Big datasets should be divided.)
    dataloader2 = torch.utils.data.DataLoader(testDs, batch_size=1,
            shuffle=True, num_workers=4)
    allImages = torch.empty([len(testDs),images.shape[1],
        images.shape[2],images.shape[3]])
    allTargets = torch.empty([len(testDs),targets.shape[1],
        targets.shape[2],targets.shape[3]])
    for i,data in enumerate(dataloader2):
        allImages[i], _ = data
    image_means = torch.mean(allImages, (0,2,3))
    print("Mean training set values:", image_means)
    image_std = torch.std(allImages, (0,2,3))
    print("Standard deviation on training set:",  image_std)
    for i,data in enumerate(dataloader2):
        _, allTargets[i] = data
    target_means = torch.mean(allTargets, (0,2,3))
    print("Mean target set values:", target_means)
    target_std = torch.std(allTargets, (0,2,3))
    print("Standard deviation on training set targets:",  target_std)
