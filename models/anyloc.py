import torch
import torch.nn as nn
from models import aggregators

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class AnyModel(nn.Module):

    def __init__(self, 
                 model_name='dinov2_vitb14',
                 pretrained=True,
                 ):
                 
        super(AnyModel, self).__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.gem = aggregators.GeMPool()

    def forward(self, x):

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks:
                x = blk(x)
        x = x.detach()

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        g = self.gem(f)

        return g