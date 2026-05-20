from typing import Literal, Optional
import os

import torch
import torch.nn as nn
import numpy as np

from .vae import VAE, get_my_vae
from .distributions import DiagonalGaussianDistribution
from ..bigvgan import BigVGAN
from ..bigvgan_v2.bigvgan import BigVGAN as BigVGANv2




class AutoEncoderModule(nn.Module):

    def __init__(self,
                 *,
                 vae_ckpt_path,
                 vocoder_ckpt_path: Optional[str] = None,
                 mode: Literal['16k', '24k', '44k'],
                 need_vae_encoder: bool = True,
                 vae_stat_path: Optional[str] = None):
        super().__init__()
        self.vae: VAE = get_my_vae(mode).eval()
        vae_state_dict = torch.load(vae_ckpt_path, weights_only=True, map_location='cpu')
        self.vae.load_state_dict(vae_state_dict)
        self.vae.remove_weight_norm()
        
        if vae_stat_path and os.path.exists(vae_stat_path):
            stat_dict = torch.load(vae_stat_path, map_location='cpu')
            if isinstance(stat_dict, dict):
                if 'mean' in stat_dict:
                    mean = torch.tensor(stat_dict['mean'], dtype=torch.float32)
                    if len(mean.shape) == 1:
                        mean = mean.view(1, -1, 1)
                    self.vae.data_mean = nn.Buffer(mean)
                if 'std' in stat_dict:
                    std = torch.tensor(stat_dict['std'], dtype=torch.float32)
                    if len(std.shape) == 1:
                        std = std.view(1, -1, 1)
                    self.vae.data_std = nn.Buffer(std)
                elif 'var' in stat_dict:
                    var = torch.tensor(stat_dict['var'], dtype=torch.float32)
                    std = torch.sqrt(var)
                    if len(std.shape) == 1:
                        std = std.view(1, -1, 1)
                    self.vae.data_std = nn.Buffer(std)

        if mode == '16k' or mode == '24k':
            assert vocoder_ckpt_path is not None
            self.vocoder = BigVGAN(vocoder_ckpt_path).eval()
        elif mode == '44k':
            self.vocoder = BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x',
                                                     use_cuda_kernel=False)
            self.vocoder.remove_weight_norm()
        else:
            raise ValueError(f'Unknown mode: {mode}')

        for param in self.parameters():
            param.requires_grad = False

        if not need_vae_encoder:
            del self.vae.encoder

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.inference_mode()
    def vocode(self, spec: torch.Tensor) -> torch.Tensor:
        return self.vocoder(spec)
