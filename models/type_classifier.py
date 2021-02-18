import torch


class TypeClassifier(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError