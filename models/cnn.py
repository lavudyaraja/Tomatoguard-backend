import torch
import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    """
    CNN: Convolutional Neural Network (using ResNet50 as base)
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(CNNModel, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = CNNModel(pretrained=False)
    print(model)
