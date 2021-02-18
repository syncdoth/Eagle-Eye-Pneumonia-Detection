import os

import numpy as np
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import torchvision.transforms.functional as F


class DetectionDataset():
    """Usage:

    img, target = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image
    target: dict with 2 keys:
    {
        "boxes": [K, 4] torch.Tensor. ground truth bbox for each roi. rchw format.
        "labels": [K,] torch.Tensor. each image's label.
            Pneumonia = 1, Covid = 2, normal = 0, unlabeled = -1
    }
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
        boxes = objs[["x", "y", "w", "h"]].to_numpy()

        # data augmentation
        W = img.shape[2]
        if self.mode == "train" and torch.rand(1) < 0.5:
            # random horizontal flip with prob 0.5
            img = F.hflip(img)
            boxes[:, 0] = W - boxes[:, 0] - boxes[:, 2]

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
        # target dict of tensors
        target = {}
        target["boxes"] = torch.Tensor(boxes)
        target["labels"] = torch.Tensor(labels).long()

        return img, target
