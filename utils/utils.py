import os
import io

import pydicom
from PIL import Image
import numpy as np


def fetch_file_as_bytesIO(path, sftp):
    """
    Using the sftp client it retrieves the file on the given path by using pre fetching.
    :param sftp: the sftp client
    :param path: path of the file to retrieve
    :return: bytesIO with the file content
    """
    with sftp.file(path, mode='rb') as file:
        file_size = file.stat().st_size
        file.prefetch(file_size)
        file.set_pipelined()
        return io.BytesIO(file.read(file_size))


def load_img(image_path, sftp_client):
    """loads image into numpy array.

    Args:
        image_path (String): a string path to the image.
    """
    ext = os.path.basename(image_path).split(".")[-1].lower()
    img_file = fetch_file_as_bytesIO(image_path, sftp_client)
    if ext == "dcm":
        image = pydicom.read_file(img_file)
        image = image.pixel_array

    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(img_file)
        image = np.array(image)

    else:
        raise ValueError(f"the image has unsupprted extension: {ext}")

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3,
                          axis=2)  # make it 3 channel by repeating

    if image.shape[2] == 4:
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = np.array(image)  # make 4 channel to 3 channel

    image = image.transpose(2, 0, 1)  # [C, H, W]

    img_file.close()
    return image
