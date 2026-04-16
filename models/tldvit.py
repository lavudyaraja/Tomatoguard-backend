import torch
import torch.nn as nn

class TLDViTModel(nn.Module):
    """
    TLDViT: Tomato Leaf Disease Vision Transformer
    """
    def __init__(self, num_classes=10):
        super(TLDViTModel, self).__init__()
        
        # Custom patch embedding
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        # Transformer architecture
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=12
        )
        
        # Classification head tailored for Tomato Leaf Disease
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # [B, N, C]
        
        # Append class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, N+1, C]
        
        x = x.transpose(0, 1) # [N+1, B, C]
        x = self.transformer(x)
        
        # take the class token output
        cls_out = x[0] # [B, C]
        out = self.head(cls_out)
        return out

if __name__ == "__main__":
    model = TLDViTModel()
    dummy = torch.randn(2, 3, 224, 224)
    print(model(dummy).shape)
