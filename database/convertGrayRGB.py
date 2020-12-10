from PIL import Image
#import cv2
from glob import glob
import os.path
import numpy as np
#import matplotlib.pyplot as plt

in_dir = "validSetLines/inputGray"
out_dir = "validSetLines/input"

imgs_names = glob(os.path.join(in_dir, "*"))

for img_name in imgs_names:
    img = np.asarray(Image.open(img_name))
    new_img = np.stack((img, img, img), axis=2)
    new_img = Image.fromarray(np.uint8(new_img))
    new_img.save(os.path.join(out_dir, os.path.split(img_name)[-1]))
