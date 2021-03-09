from EfficientNet import EfficientNet as EN
import torch
import torch.nn
from math import ceil


class TypeClassifier(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError
            

    def forward(self,X):
        logit = self.EN(X)
        
        return logit

