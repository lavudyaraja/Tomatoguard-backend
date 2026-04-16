import torch
import torch.nn as nn
import timm

class CvTModel(nn.Module):
    """
    CvT: Convolutional Vision Transformer
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(CvTModel, self).__init__()
        self.model = timm.create_model('cvt_13', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = CvTModel(pretrained=False)
    print(model)
