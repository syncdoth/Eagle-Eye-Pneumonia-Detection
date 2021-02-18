import os

import numpy as np
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms


# TODO: Differentiate between Pretrain Dataset and SSL dataset
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
        img = np.load(img_path)

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

        train_transform = transforms.Compose([
            self.trans,
            target_transform,
        ])

        img = train_transform(img) if self.mode == "train" else target_transform(img)

        return img, labels