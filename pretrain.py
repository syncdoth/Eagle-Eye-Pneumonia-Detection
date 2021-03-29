import torch
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler

from models.type_classifier import TypeClassifier
import models.train
from models.train import train, make_batch
import dataloader.ClassificationDataset
from dataloader.ClassificationDataset import ClassificationDataset, AddGaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_aug = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=[0.1, 0.5], contrast=0, saturation=0, hue=0),
    transforms.RandomAffine(degrees=(0, 15),
                            translate=None,
                            scale=None,
                            shear=None,
                            fillcolor=None,
                            resample=None),
    AddGaussianNoise(0., 1.)
])

dataset = ClassificationDataset(root="/home/server/duhyeuk", neg_prop=0.5, trans=None)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=64,
                                           collate_fn=make_batch,
                                           sampler=SubsetRandomSampler(dataset.train_idx))
val_loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=64,
                                         collate_fn=make_batch,
                                         sampler=SubsetRandomSampler(dataset.val_idx))
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=64,
                                          collate_fn=make_batch,
                                          sampler=SubsetRandomSampler(dataset.test_idx))

version = "b1"
pretrain_model = TypeClassifier(version=version, num_classes=4).to(device)

optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-5)
loss = torch.nn.CrossEntropyLoss(
    weight=torch.FloatTensor(dataset.get_class_weights()).to(device))
epochs = 10

history = train(pretrain_model,
                optimizer,
                train_loader,
                device,
                loss,
                val_dataset=val_loader,
                epochs=epochs,
                model_dir=f"/content/drive/MyDrive/eagle_eye_models/{version}-new")

# evaluate
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure()
# plt.title("Train vs Valid Accuracy")
# plt.plot(history["train_acc"], label="train_acc")
# plt.plot(np.repeat(np.array(history["val_acc"]), len(train_loader)), label="val_acc")
# plt.legend()

# plt.figure()
# plt.title("Train vs Valid Loss")
# plt.plot(history["train_loss"], label="train_loss")
# plt.plot(np.repeat(np.array(history["val_loss"]), len(train_loader)), label="val_loss")
# plt.legend()

# plt.figure()
# plt.title("Valid Accuracy vs F1")
# plt.plot(np.array(history["val_acc"]), label="val_acc")
# plt.plot(np.array(history["val_f1"]), label="val_f1")
# plt.legend()
