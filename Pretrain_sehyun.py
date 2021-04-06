#!/usr/bin/env python
# coding: utf-8

# ## Initial Setup

# In[12]:


import os
# os.ENVIRON["CUDA_VISIBLE_DEVICES"] = "0"


# ## Train

# In[13]:


import torch
from importlib import reload
from torch.utils.data import SubsetRandomSampler
from models.type_classifier import TypeClassifier
import models.train
reload(models.train)
from models.train import train, make_batch
import dataloader.ClassificationDataset
reload(dataloader.ClassificationDataset)
from dataloader.ClassificationDataset import ClassificationDataset, AddGaussianNoise
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[14]:


feature_aug = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=[0.8, 1.0], contrast=0, saturation=0, hue=0)], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(degrees=(0,15), fill=0),
                                        transforms.RandomHorizontalFlip(p=0.5)], p=0.2),
                transforms.RandomApply([AddGaussianNoise(0., 0.01)], p=0.5)])


# In[15]:


dataset = ClassificationDataset(root="/home/janghoon/pneumonia_data", neg_prop=0, trans=feature_aug)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=make_batch, sampler=SubsetRandomSampler(dataset.train_idx))
val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=make_batch, sampler=SubsetRandomSampler(dataset.val_idx))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=make_batch, sampler=SubsetRandomSampler(dataset.test_idx))


# In[19]:


version = "b1"
pretrain_model = TypeClassifier(version=version, num_classes=2).to(device)

optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-5)
# loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(dataset.get_class_weights()).to(device))
loss = torch.nn.CrossEntropyLoss()
epochs = 10

history = train(pretrain_model, optimizer, train_loader, device, loss, 
                val_dataset=val_loader, epochs=epochs,
                model_dir=f"eagle_eye_models/sehyun/{version}-final1-neg0-cov0-aug")


# ## Visualize Results

# In[ ]:


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
# plt.legend();


# In[ ]:





# In[ ]:




