# linesUnmixGan

## Info

linesUnmixGan leverages the power of GANs to separate mixed lines of text.

There are two different generative architectures, U-Net and FPN.
The discriminator is based on PatchGan. 

## Website

For more information, visit [the github.io page](https://benjamindorra.github.io/linesUnmixGan/).

## Usage

### Test

To test the performance, download the pretrained weights and add them in the netWeights/ directory using git LFS.

Then launch either testUNetHW.py or testFPNHW.py.

By default, they read database/testSetHWLines/input which contains extracts from the ICDAR 2013 line segmentation dataset, and output the result to database/testSetHWLines/result.

### Training

Read the database README to generate the datasets.

Use customDataset.py to test your dataset and compute the mean and std over the entire dataset.

In trainUNet.py or trainFPN.py modify the saveFileGen, saveFileDiscrim, trainset and valset to the desired paths. For better results modify normInput, normTarget, mean and std to the values given by customDataset.py.

Then launch trainUNet.py or trainFPN.py and regularly check the progress with the curve (loss_curve.png), as well as the generated and target image (generated_image.png and target_image.png) until they are satisfactory.

## Requirements

python 3
torch
torchvision
matplotlib
PIL
opencv-python
glob

optional:
LaTeX
ImageMagick
(for printed text only)

Tested on linux with python 3.6.9 and torch 1.6.0
