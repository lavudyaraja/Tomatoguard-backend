import torch
import torch.nn as nn

class HybridCNNTransformer(nn.Module):
    """
    Hybrid model combining CNN and Transformer architectures
    """
    def __init__(self, num_classes=10):
        super(HybridCNNTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=2
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.cnn(x)     # [B, 128, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1) # [seq_len, B, 128]
        x = self.transformer(x)
        x = x.mean(dim=0)   # Global average pooling
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = HybridCNNTransformer()
    dummy = torch.randn(1, 3, 224, 224)
    print(model(dummy).shape)
