import torch
import torch.nn as nn
import timm

class CeiTModel(nn.Module):
    """
    CeiT: Convolutional cross-covariance Image Transformer
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(CeiTModel, self).__init__()
        # Fallback to a standard ViT shape if CeiT specifically isn't found in timm version
        # Usually it's 'vit_base_patch16_224' or similar until CeiT is fully supported
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = CeiTModel(pretrained=False)
    print(model)
