import torch
import torch.nn as nn
import torchvision.models as models

class ViTModel(nn.Module):
    """
    ViT: Vision Transformer
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(ViTModel, self).__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        self.model.heads = nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ViTModel(pretrained=False)
    print(model)
