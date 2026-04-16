import torch
import torch.nn as nn
import timm

class MobileViTModel(nn.Module):
    """
    MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileViTModel, self).__init__()
        # Using timm's implementation of MobileViT
        self.model = timm.create_model('mobilevitv2_100', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = MobileViTModel(pretrained=False)
    print(model)
