"""
Data preparation script (not for use on Spark)
converts images to float32 numpy arrays
"""
import numpy as np
from scipy.misc import imread
import os
from tqdm import tqdm
from shutil import rmtree


# constants
IMAGES_DIR = 'D:\\Education\\Yandex\\Big Data\\data\\4th PIV-Challenge Images Case A'
OUT_DIR = os.path.join(IMAGES_DIR, 'out')
IMAGE_EXTENSION = '.tif'


# get images
img_names = [name for name in os.listdir(IMAGES_DIR) if name[-4:] == IMAGE_EXTENSION]

# reset output dir
rmtree(os.path.join(IMAGES_DIR, 'out'), ignore_errors=True)
os.mkdir(os.path.join(IMAGES_DIR, 'out'))

# process images
for img_name in tqdm(img_names):
    image = imread(os.path.join(IMAGES_DIR, img_name)).astype(np.float32)
    new_name = os.path.join(OUT_DIR, img_name[:-4])
    np.save(new_name, image)
