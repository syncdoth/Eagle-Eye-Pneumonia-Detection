import torch


class TypeClassifier(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, X):
        """
        X.shape = [batch_size, 3, 224, 224]

        logit = [batch_size, num_classes]
        """
        raise NotImplementedError