"""timm-equivalent ViT-Tiny with Opacus-compatible token embeddings."""
import hashlib
import os
from pathlib import Path

import timm
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, length, dim = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(n, length, dim))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, patch_size=16, embed_dim=192, depth=12, heads=3,
                 mlp_ratio=4, num_classes=100, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.tokens = (image_size // patch_size) ** 2 + 1
        self.patch = nn.Linear(3 * patch_size ** 2, embed_dim)
        self.cls = nn.Embedding(1, embed_dim)
        self.position = nn.Embedding(self.tokens, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, heads, mlp_ratio)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        n = images.shape[0]
        patches = F.unfold(images, self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch(patches)
        cls = self.cls(torch.zeros(n, 1, device=images.device, dtype=torch.long))
        positions = torch.arange(self.tokens, device=images.device).repeat(n, 1)
        x = torch.cat((cls, x), dim=1) + self.position(positions)
        return self.head(self.norm(self.blocks(x))[:, 0])

    def load_backbone(self, reference):
        """Map every timm backbone tensor exactly; retain the random 100-way head."""
        mapped = {}
        for name, tensor in reference.state_dict().items():
            if name.startswith('head.'):
                continue
            if name == 'cls_token':
                name, tensor = 'cls.weight', tensor.reshape(1, -1)
            elif name == 'pos_embed':
                name, tensor = 'position.weight', tensor.squeeze(0)
            elif name == 'patch_embed.proj.weight':
                name, tensor = 'patch.weight', tensor.flatten(1)
            elif name == 'patch_embed.proj.bias':
                name = 'patch.bias'
            name = name.replace('.mlp.fc1.', '.mlp.0.').replace('.mlp.fc2.', '.mlp.2.')
            mapped[name] = tensor
        mapped.update({name: value for name, value in self.state_dict().items()
                       if name.startswith('head.')})
        self.load_state_dict(mapped, strict=True)


def checkpoint_path(model_path):
    path=Path(model_path)/'model.safetensors'
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pretrained_vit(model_path):
    """Same strict timm-to-Opacus backbone mapping as exp1e, explicit local load."""
    path = checkpoint_path(model_path)
    reference = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    reference.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    model = ViTTiny()
    model.load_backbone(reference)
    cfg = reference.pretrained_cfg
    metadata = {key: cfg[key] for key in ('mean', 'std', 'crop_pct', 'interpolation', 'hf_hub_id')}
    metadata.update(checkpoint_path=str(path),
                    checkpoint_sha256=checkpoint_sha256(path))
    return model, metadata


def initialization_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
