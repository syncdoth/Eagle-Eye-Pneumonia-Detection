import os

import numpy as np
import torch
import pandas as pd
from PIL import Image
import pydicom

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms


class ClassificationDataset(torch.utils.data.Dataset):
    """Usage:

    img, labels = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image?
    labels: [1,] torch.Tensor. each image's label.
        Pneumonia = 1, Covid = 2, normal = 0
    """
    def __init__(self, root="data_server", split=[0.8, 0.1, 0.1], trans=None):
        self.root = root
        self.split = split
        self.trans = trans
        self.class2id = {
            'Normal': 0,
            'Not Normal': 0,
            'COVID19-PN': 1,
            'TheOther-PN': 2,
            'Viral-PN': 3
        }

        self.data_path = pd.read_csv(os.path.join(root, "class/meta/metadata.csv"))
        self.data_path = self.data_path[self.data_path["CLASS"] != "Unclassified-PN"]

        self.data_path["IMG_PATH"] = self.data_path["FILE_PATH"].map(
            lambda x: x[2:]) + self.data_path["FILE NAME"]
        self.imgs = self.data_path["IMG_PATH"].to_numpy()
        self.labels = self.data_path["CLASS"].map(lambda x: self.class2id[x]).to_numpy()

        self.train_idx, self.val_idx, self.test_idx = self.dataset_split()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # load image and label
        img_path = os.path.join(self.root, self.imgs[idx])
        img = load_img(img_path)
        label = self.labels[idx]

        # normalize to [0, 1]
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[0])).reshape(img.shape)
        img = torch.Tensor(img)

        # data augmentation
        target_transform = transforms.Compose([
            # TODO: Add resize, change it to actual resnet norm, and add more augmentations
            transforms.Resize([224, 224]),
            # use actual spec
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if self.trans is not None:
            train_transform = transforms.Compose([
                self.trans,
                target_transform,
            ])
        else:
            train_transform = target_transform

        img = train_transform(img) if idx in self.train_idx else target_transform(img)

        return img, label

    def dataset_split(self):
        labels = self.labels
        train_idx, val_idx = train_test_split(np.arange(len(labels)),
                                              test_size=self.split[1] + self.split[2],
                                              shuffle=True,
                                              stratify=labels)
        val_idx, test_idx = train_test_split(np.arange(len(val_idx)),
                                             test_size=self.split[2] /
                                             (self.split[1] + self.split[2]),
                                             shuffle=True,
                                             stratify=labels[val_idx])

        return train_idx, val_idx, test_idx


# TODO: move to utils.
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
