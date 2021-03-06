import os

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.transforms.functional as F

from utils.utils import load_img


class DetectionDataset(torch.utils.data.Dataset):
    """Usage:

    img, target = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image
    target: dict with 2 keys:
    {
        "boxes": [K, 4] torch.Tensor. ground truth bbox for each roi. rchw format.
        "labels": [K,] torch.Tensor. each image's label.
            Covid = 1, Other PN = 2, Viral PN = 3, normal = 0
    }
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
        self.data_path["IMG_PATH"] = self.data_path["FILE_PATH"].map(
            lambda x: x[2:]) + self.data_path["FILE NAME"]
        self.imgs = list(dict.fromkeys(self.data_path["IMG_PATH"].to_list()))

        self.train_idx, self.val_idx, self.test_idx = self.dataset_split()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, self.imgs[idx])
        img = load_img(img_path)

        # normalize to [0, 1]
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[0])).reshape(img.shape)
        img = torch.Tensor(img)

        # load bbox and label
        objs = self.data_path[self.data_path['IMG_PATH'] == self.imgs[idx]]

        # TODO: change here
        labels = objs["CLASS"].map(lambda x: self.class2id[x]).to_numpy()
        boxes = objs[["x", "y", "w", "h"]].to_numpy()

        # data augmentation
        W = img.shape[2]
        if idx in self.train_idx and torch.rand(1) < 0.5:
            # random horizontal flip with prob 0.5
            img = F.hflip(img)
            boxes[:, 0] = W - boxes[:, 0] - boxes[:, 2]

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
        # target dict of tensors
        target = {
            "boxes": torch.Tensor(boxes),
            "labels": torch.Tensor(labels).long(),
        }

        return img, target

    def dataset_split(self):
        images = self.imgs
        train_idx, val_idx = train_test_split(np.arange(len(images)),
                                              test_size=self.split[1] + self.split[2],
                                              shuffle=True)
        val_idx, test_idx = train_test_split(np.arange(len(val_idx)),
                                             test_size=self.split[2] /
                                             (self.split[1] + self.split[2]),
                                             shuffle=True)
        return train_idx, val_idx, test_idx
