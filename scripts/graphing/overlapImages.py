import os
import matplotlib.pyplot as plt
import numpy as np

def overlapImages():
    """
    Overlaps multiple images from the 'images' directory and displays the result.
    Each image is converted to grayscale and then overlapped using alpha blending.
    The resulting overlapped image is displayed using matplotlib.
    """
    image_dir = "./images"
    images = os.listdir(image_dir)
    alpha = 1 / len(images)

    fig, ax = plt.subplots()

    total_img = np.zeros((480, 640))

    for image_file in images:
        image_path = os.path.join(image_dir, image_file)
        rgb_img = np.array(plt.imread(image_path))
        bw_img = 0.2126 * rgb_img[:, :, 0] + 0.7152 * rgb_img[:, :, 1] + 0.0722 * rgb_img[:, :, 2]
        bw_img = 1 - bw_img
        total_img += bw_img * alpha

    ax.imshow(total_img, cmap='gray')

    plt.show()

overlapImages()