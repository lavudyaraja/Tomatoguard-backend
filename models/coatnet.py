import torch
import torch.nn as nn
import timm

class CoAtNetModel(nn.Module):
    """
    CoAtNet: Marrying Convolution and Attention for All Data Sizes
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(CoAtNetModel, self).__init__()
        self.model = timm.create_model('coatnet_0_rw_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = CoAtNetModel(pretrained=False)
    print(model)
