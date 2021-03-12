"""This is a class used for type classification task.

Currently, we are using EfficientNet implementations.
"""
from models.efficientnet import EfficientNet
from efficientnet_pytorch import EfficientNet as EN_torch
import torch
import torch.nn


class TypeClassifier(torch.nn.Module):
    def __init__(self, version, num_classes, implementation="EN-torch", pretrained=True):
        super().__init__()
        if implementation == "ours":
            self.model = EfficientNet(version, num_classes)
        elif implementation == "EN-torch":
            if pretrained:
                self.model = EN_torch.from_pretrained(f"efficientnet-{version}",
                                                      num_classes=num_classes)
            else:
                self.model = EN_torch.from_name(f"efficientnet-{version}",
                                                num_classes=num_classes)
        else:
            raise NotImplementedError(
                "The implementation must be either [ours, EN-torch]."
                f"\nyour input: {implementation}")

    def forward(self, X):
        logit = self.model(X)

        return logit
