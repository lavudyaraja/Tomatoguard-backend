import torch
import torch.nn as nn

class iMHSA_ViT_DNN_Model(nn.Module):
    """
    iMHSA-ViT + DNN: improved Multi-Head Self-Attention Vision Transformer with Deep Neural Network
    """
    def __init__(self, num_classes=10):
        super(iMHSA_ViT_DNN_Model, self).__init__()
        # Feature extraction using simulated ViT structure
        self.vit_features = nn.Linear(224*224*3, 512) 
        
        # improved Multi-Head Self Attention representation
        self.imhsa = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        
        # Deep Neural Network for classification
        self.dnn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        features = self.vit_features(x).unsqueeze(0) # [1, B, 512]
        
        # Self-attention 
        attn_out, _ = self.imhsa(features, features, features)
        attn_out = attn_out.squeeze(0) # [B, 512]
        
        # Classification
        out = self.dnn(attn_out)
        return out

if __name__ == "__main__":
    model = iMHSA_ViT_DNN_Model()
    dummy = torch.randn(2, 3, 224, 224)
    print(model(dummy).shape)
