import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== DropPath Utility ========
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    output = x / keep_prob * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ======== SE Block ========
class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.GELU(),
            nn.Linear(ch // r, ch), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


# ======== MBConv ========
class MBConv(nn.Module):
    def __init__(self, dim, expand=4, drop_path=0.):
        super().__init__()
        hid = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, 1, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.GELU(),
            SE(hid),
            nn.Conv2d(hid, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.net(x))


# ======== Relative Positional Bias ========
class RelPosBias(nn.Module):
    def __init__(self, win):
        super().__init__()
        self.win = win
        num = (2 * win - 1) * (2 * win - 1)
        self.table = nn.Parameter(torch.zeros(num, 1))
        nn.init.trunc_normal_(self.table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij'))
        coords = coords.reshape(2, -1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel[0] += win - 1
        rel[1] += win - 1
        rel[0] *= 2 * win - 1
        self.register_buffer('idx', rel.sum(0).long())

    def forward(self):
        return self.table[self.idx].squeeze(-1)


# ======== Multi-Axis Attention ========
class MaxViTAttn(nn.Module):
    def __init__(self, dim, heads=8, win=7, mode='block', drop_path=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.mode = mode
        self.win = win
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.pos = RelPosBias(win)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition(self, x, H, W):
        B, N, C = x.shape
        w = self.win
        x = x.view(B, H, W, C)
        if self.mode == 'block':
            x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(-1, w * w, C)
        else:
            x = x.view(B, H // w, w, W // w, w, C).permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(-1, (H // w) * (W // w), C)
        return x

    def _unpartition(self, x, B, H, W):
        w = self.win
        C = x.shape[-1]
        if self.mode == 'block':
            nH, nW = H // w, W // w
            x = x.view(B, nH, nW, w, w, C).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B, H * W, C)
        else:
            nH, nW = H // w, W // w
            x = x.view(B, w, w, nH, nW, C).permute(0, 3, 1, 4, 2, 5)
            x = x.reshape(B, H * W, C)
        return x

    def forward(self, x, H, W):
        B = x.shape[0]
        res = x
        x = self.norm(x)
        x = self._partition(x, H, W)
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.pos()
        if bias.shape[0] == N:
            attn = attn + bias.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x = self.proj(x)
        x = self._unpartition(x, B, H, W)
        return res + self.drop_path(x)


# ======== FFN ========
class FFN(nn.Module):
    def __init__(self, dim, mult=4, drop_path=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.net(x))


# ======== MaxViT Block ========
class MaxViTBlock(nn.Module):
    def __init__(self, dim, heads=8, win=7, drop_path=0.):
        super().__init__()
        self.mbconv = MBConv(dim, drop_path=drop_path)
        self.block_attn = MaxViTAttn(dim, heads, win, 'block', drop_path=drop_path)
        self.ffn1 = FFN(dim, drop_path=drop_path)
        self.grid_attn = MaxViTAttn(dim, heads, win, 'grid', drop_path=drop_path)
        self.ffn2 = FFN(dim, drop_path=drop_path)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.mbconv(x)
        xt = x.flatten(2).transpose(1, 2)
        xt = self.block_attn(xt, H, W)
        xt = self.ffn1(xt)
        xt = self.grid_attn(xt, H, W)
        xt = self.ffn2(xt)
        return xt.transpose(1, 2).view(B, C, H, W)


# ======== Downsample ========
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


# ======== FULL MaxViT ========
class MaxViT(nn.Module):
    def __init__(self, img_size=224, num_classes=11, dims=[64, 128, 256, 512],
                 depths=[2, 2, 5, 2], heads=[2, 4, 8, 16], win=7,
                 drop_path_rate=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.GELU()
        )

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            if i > 0:
                blocks.append(Downsample(dims[i - 1], dims[i]))
            for _ in range(depths[i]):
                cur_size = img_size // (4 * (2 ** i)) if i > 0 else img_size // 4
                w = min(win, cur_size)
                blocks.append(MaxViTBlock(dims[i], heads[i], w, drop_path=dpr[block_idx]))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dims[-1], num_classes)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.flatten(2).mean(dim=2)
        return self.head(self.norm(x))