"""
加载 ACE-Step DiT 主干 (AceStepDiTModel),并把它的每一层 cross_attn 接口
重定向到接受 Ovi video 塔传过来的 hidden state(原本是接歌词 encoder)。

核心:
  - ACE-Step 的 AceStepDiTLayer 自带 cross_attn(用 self.cross_attn,见 modeling_acestep_v15_turbo.py:467)
  - 它的 cross_attn 期望 encoder_hidden_states shape [B, L_enc, hidden_size]
  - 我们把 Ovi video tower 第 i 层的 hidden state 投影到 ACE-Step 维度后,
    作为 encoder_hidden_states 喂进去
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Optional


def load_acestep_dit(acestep_project_root: str, config_path: str = "acestep-v15-turbo",
                    device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """加载 ACE-Step DiT 主干 + config(权重从 checkpoints 拿)。

    返回 (dit_model, ace_config)。
    """
    if acestep_project_root not in sys.path:
        sys.path.insert(0, acestep_project_root)

    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepDiTModel
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from safetensors.torch import load_file

    ckpt_dir = os.path.join(acestep_project_root, "checkpoints", config_path)
    cfg = AceStepConfig.from_pretrained(ckpt_dir)
    # 强制 eager attn (Blackwell 上 flash-attn 装好后也可改成 flash_attention_2)
    cfg._attn_implementation = "eager"

    model = AceStepDiTModel(cfg)

    # 加载权重(ACE-Step 把 DiT 权重存在 model.safetensors 里,以 "dit." 为前缀)
    ckpt_file = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(ckpt_file):
        sd = load_file(ckpt_file)
        dit_sd = {k.replace("dit.", "", 1): v for k, v in sd.items() if k.startswith("dit.")}
        missing, unexpected = model.load_state_dict(dit_sd, strict=False)
        print(f"[ACE-Step DiT loaded] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg


class CrossAttnAdapter(nn.Module):
    """两个塔维度不一致时的投影适配:
       in_dim -> out_dim,LayerNorm 后再线性投影,初始化接近恒等。
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # 初始化为小值,避免一开始破坏预训练特征
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        return self.proj(self.norm(x))


def build_layer_mapping(num_video_layers: int, num_audio_layers: int):
    """video 层数和 audio 层数不一致时,把每个 video block 映射到最近的 audio block。
    返回 dict: video_layer_idx -> audio_layer_idx
    """
    mapping = {}
    for vi in range(num_video_layers):
        ai = int(round(vi * (num_audio_layers - 1) / max(num_video_layers - 1, 1)))
        mapping[vi] = ai
    return mapping
