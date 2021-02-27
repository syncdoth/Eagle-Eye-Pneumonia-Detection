import os

import numpy as np
import torch
import pandas as pd
from PIL import Image
import pydicom

from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms


class ClassificationDataset(torch.utils.data.Dataset):
    """Usage:

    img, labels = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image?
    labels: [1,] torch.Tensor. each image's label.
        Pneumonia = 1, Covid = 2, normal = 0
    """
    def __init__(self, root, mode="train", trans=None):
        self.root = root
        self.mode = mode
        self.trans = trans

        self.data_path = pd.read_csv(os.path.join(root, f"data_{mode}.csv"))
        self.imgs = list(dict.fromkeys(self.data_path["filename"].to_list()))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, self.mode, self.imgs[idx])
        img = load_img(img_path)

        # normalize to [0, 1]
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        img = torch.Tensor(img)

        # load bbox and label
        objs = self.data_path[self.data_path['filename'] == self.imgs[idx]]
        labels = objs["label"].to_numpy()

        # data augmentation
        target_transform = transforms.Compose([
            # TODO: Add resize, change it to actual resnet norm, and add more augmentations
            transforms.Resize([224, 224]),
            # use the first channel as last channel again
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if self.trans is not None:
            train_transform = transforms.Compose([
                self.trans,
                target_transform,
            ])
        else:
            train_transform = target_transform

        img = train_transform(img) if self.mode == "train" else target_transform(img)

        return img, labels


# TODO: move to utils.
def load_img(image_path):
    """loads image into numpy array.

    Args:
        image_path (String): a string path to the image.
    """
    ext = os.path.basename(image_path).split(".")[1]
    if ext == "dcm":
        image = pydicom.read_file(image_path)
        image = image.pixel_array
    elif ext in ["png", "jpg"]:
        image = Image.open(image_path)
        image = np.array(image)
    else:
        raise ValueError(f"the image has unsupprted extension: {ext}")
    return image
