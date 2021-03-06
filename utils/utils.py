import os

import pydicom
from PIL import Image
import numpy as np


def load_img(image_path):
    """loads image into numpy array.

    Args:
        image_path (String): a string path to the image.
    """
    ext = os.path.basename(image_path).split(".")[-1]
    if ext == "dcm":
        image = pydicom.read_file(image_path)
        image = image.pixel_array

    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(image_path)
        image = np.array(image)

    else:
        raise ValueError(f"the image has unsupprted extension: {ext}")

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3,
                          axis=2)  # make it 3 channel by repeating

    image = image.transpose(2, 0, 1)  # [C, H, W]
    return image
