"""
ACE-Step 加载
"""
import os
import sys
import torch
import torch.nn as nn


def init_acestep_handler(acestep_project_root: str,
                          config_path: str = "acestep-v15-turbo",
                          device: str = "cuda",
                          dtype: torch.dtype = torch.bfloat16,
                          offload_to_cpu: bool = False):
    if acestep_project_root not in sys.path:
        sys.path.insert(0, acestep_project_root)

    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    status_msg, ok = handler.initialize_service(
        project_root=acestep_project_root,
        config_path=config_path,
        device=device,
        offload_to_cpu=offload_to_cpu,
        use_flash_attention=False,   # 与 Ovi 的 flash-attn 隔离
    )
    if not ok:
        raise RuntimeError(f"ACE-Step handler init failed: {status_msg}")

    handler.dtype = dtype
    return handler


class CrossAttnAdapter(nn.Module):
    """两塔维度不一致时的投影适配:
       in_dim -> out_dim,LayerNorm 后再线性投影,初始化为小值,不破坏预训练特征。
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        return self.proj(self.norm(x))


@torch.no_grad()
def encode_audio_to_latent(handler, audio: torch.Tensor) -> torch.Tensor:
    """用 ACE-Step VAE 把 wav 编码到 latent。

    Args:
        handler: 已 init 的 AceStepHandler
        audio: [B, C, S] 48kHz 音频 tensor

    Returns:
        target_latents: [B, T, 64]
    """
    vae = handler.vae
    device = next(vae.parameters()).device
    audio = audio.to(device=device, dtype=vae.dtype)
    latent = vae.encode(audio).latent_dist.sample()
    return latent.transpose(1, 2).to(handler.dtype)   # [B, T, 64]


@torch.no_grad()
def decode_latent_to_audio(handler, latent: torch.Tensor) -> torch.Tensor:
    """用 ACE-Step VAE 把 latent 解码回 waveform。

    Args:
        latent: [B, T, 64]

    Returns:
        audio: [B, C, S] @ 48kHz
    """
    vae = handler.vae
    device = next(vae.parameters()).device
    # VAE decode 输入是 [B, 64, T]
    x = latent.transpose(1, 2).to(device=device, dtype=vae.dtype)
    return vae.decode(x).sample


def build_context_latents_default(handler, latent_length: int, device, dtype):
    """构造默认 context_latents(无 src audio,纯静音)。

    返回 [1, latent_length, 128]
    """
    silence = handler.silence_latent[:, :latent_length, :].to(device=device, dtype=dtype)
    if silence.shape[1] < latent_length:
        pad = latent_length - silence.shape[1]
        silence = torch.cat([silence,
                              silence[:, :pad, :].expand(1, -1, -1)], dim=1)
    chunk_masks = torch.ones(1, latent_length, 64, device=device, dtype=dtype)
    return torch.cat([silence, chunk_masks], dim=-1)   # [1, T, 128]
