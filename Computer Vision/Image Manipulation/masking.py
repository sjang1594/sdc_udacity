import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def create_mask(path, color_threshold):
    img = np.array(Image.open(path).convert('RGB'))
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    rt, gt, bt = color_threshold
    mask = (r > rt) & (g > gt) & (b > bt)
    return img, mask

def mask_and_display(img, mask):
    masked_image = img * np.stack([mask]*3, axis=2)
    f, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[2].imshow(masked_image)
    plt.show()

if __name__ == '__main__':
    path = './Image Manipulation/images/segment-12212767626682531382_2100_150_2120_150_with_camera_labels_20.png'
    # Color Threshold - Middle Value
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)