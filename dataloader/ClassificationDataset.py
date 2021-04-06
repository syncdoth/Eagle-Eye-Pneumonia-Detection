import os

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms

from utils.utils import load_img


class AddGaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ClassificationDataset(torch.utils.data.Dataset):
    """Usage:

    img, labels = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image
    labels: [1,] torch.Tensor. each image's label.
        Covid = 1, Other PN = 2, Viral PN = 3, normal = 0
    """

    def __init__(self,
                 root="data_server",
                 split=[0.8, 0.1, 0.1],
                 trans=None,
                 neg_prop=0.5):
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

        self.data_path = pd.read_csv(os.path.join(root, "class/meta/final1.csv"))
        self.data_path = self.data_path[self.data_path["CLASS"] != "Unclassified-PN"]
        self.data_path = self.data_path[self.data_path["CLASS"] != "COVID19-PN"]

        self.data_path["IMG_PATH"] = self.data_path["FILE_PATH"].map(
            lambda x: x[2:]) + self.data_path["FILE NAME"]
        self.imgs = self.data_path["IMG_PATH"].to_numpy()
        self.labels = self.data_path["CLASS"].map(lambda x: self.class2id[x]).to_numpy()

        pos_idx, neg_idx = self.negative_sampling(neg_prop)
        data_idx = np.concatenate([pos_idx, neg_idx])
        self.imgs, self.labels = self.imgs[data_idx], self.labels[data_idx]
        self.train_idx, self.val_idx, self.test_idx = self.dataset_split()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # load image and label
        img_path = os.path.join(self.root, self.imgs[idx])
        img = load_img(img_path)
        label = self.labels[idx]
        label = torch.Tensor([label]).long()

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

        return img, label - 2

    def dataset_split(self):
        labels = self.labels
        train_idx, val_idx = train_test_split(np.arange(len(labels)),
                                              test_size=self.split[1] + self.split[2],
                                              shuffle=True,
                                              stratify=labels)
        val_idx, test_idx = train_test_split(val_idx,
                                             test_size=self.split[2] /
                                             (self.split[1] + self.split[2]),
                                             shuffle=True,
                                             stratify=labels[val_idx])

        return train_idx, val_idx, test_idx

    def negative_sampling(self, neg_prop=0.5):
        pos_idx = np.where(self.labels > 0)[0]
        num_neg = int(pos_idx.shape[0] * neg_prop)

        neg_idx = np.where(self.labels == 0)[0]
        sampled_neg_idx = np.random.choice(neg_idx, num_neg, replace=False)

        return pos_idx, sampled_neg_idx

    def get_class_weights(self):
        return compute_class_weight('balanced',
                                    classes=np.unique(self.labels[self.train_idx]),
                                    y=self.labels[self.train_idx])


class InferenceDataset(torch.utils.data.Dataset):
    """Usage:

    img, labels = dataset[idx]

    img: [3, 224, 224] torch.Tensor for image
    labels: [1,] torch.Tensor. each image's label.
        Covid = 1, Other PN = 2, Viral PN = 3, normal = 0
    """

    def __init__(self, root="data_server"):
        self.root = root

        self.data_file = pd.read_csv(os.path.join(root, "class/meta/final1.csv"))
        self.data_file["IMG_PATH"] = self.data_file["FILE_PATH"].map(
            lambda x: x[2:]) + self.data_file["FILE NAME"]

        self.candidate = self.data_file[self.data_file["CLASS"] == "Unclassified-PN"]
        # choose the images with only one bbox
        self.candidate = self.candidate[self.candidate["IMG_PATH"].map(
            self.candidate["IMG_PATH"].value_counts()) == 1]
        self.imgs = self.candidate["IMG_PATH"].to_numpy()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # load image and label
        img_path = os.path.join(self.root, self.imgs[idx])
        img = load_img(img_path)

        # normalize to [0, 1]
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img.reshape(-1, img.shape[0])).reshape(img.shape)
        img = torch.Tensor(img)

        # data augmentation
        target_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = target_transform(img)

        return img
