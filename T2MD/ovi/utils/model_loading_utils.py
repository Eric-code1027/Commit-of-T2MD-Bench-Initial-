import torch 
import os
import json
from safetensors.torch import load_file

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.modules.mmaudio.features_utils import FeaturesUtils

def init_wan_vae_2_2(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['device'] = rank
    vae_pth = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config['vae_pth'] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model

def init_mmaudio_vae(ckpt_dir, rank=0, vae_ckpt_path=None, vocoder_ckpt_path=None, vae_stat_path=None):
    """
    初始化 MMAudio VAE
    
    Args:
        ckpt_dir: checkpoint 目录
        rank: 设备 rank
        vae_ckpt_path: 自定义 VAE 权重路径（可选）
        vocoder_ckpt_path: 自定义 vocoder 权重路径（可选）
        vae_stat_path: 统计文件路径（可选，用于加载 mean 和 std）
    """
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    # 如果提供了自定义路径，使用自定义路径；否则使用默认路径
    if vae_ckpt_path is not None:
        tod_vae_ckpt = vae_ckpt_path
    else:
        tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    
    if vocoder_ckpt_path is not None:
        bigvgan_vocoder_ckpt = vocoder_ckpt_path
    else:
        bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt
    if vae_stat_path is not None:
        vae_config['vae_stat_path'] = vae_stat_path

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae

def init_bigvgan_flow_vae_simple(vae_ckpt_path, vae_stat_path, device=0):
    """
    简化版本的 BigVGAN Flow VAE 加载函数
    直接使用 init_mmaudio_vae，但支持自定义路径和统计文件
    
    Args:
        vae_ckpt_path: VAE 权重文件路径（如 g_01240000）
        vae_stat_path: 统计文件路径（如 global_mean_var_124w.stat）
        device: 设备
    
    Returns:
        VAE 模型对象
    """
    # 注意：这里假设 vae_ckpt_path 包含 VAE 权重，vocoder 可能在同一个文件或需要单独指定
    # 如果 g_01240000 只包含 vocoder，需要单独指定 VAE 路径
    return init_mmaudio_vae(
        ckpt_dir="",  # 不使用默认路径
        rank=device,
        vae_ckpt_path=vae_ckpt_path,  # 如果这是 VAE 权重
        vocoder_ckpt_path=vae_ckpt_path,  # 如果 vocoder 也在同一个文件
        vae_stat_path=vae_stat_path
    )

def init_bigvgan_flow_vae(vae_ckpt_path, vae_stat_path, device=0):
    """
    初始化 BigVGAN Flow VAE（按 HunyuanVideo_pureTorch 的 export 代码加载）

    这类 checkpoint (g_01240000) 对应的是一体化的 BigVGANFlowVAE（包含 encoder/decoder/vocoder/flow 等），
    不能再拆成 MMAudio 的 VAE + BigVGANVocoder 来硬适配。

    Returns:
        一个带 wrapped_encode / wrapped_decode 的轻量 wrapper，接口与训练代码兼容：
        - wrapped_encode(audio) -> latent (B, C, T)
        - wrapped_decode(latent) -> audio (B, 1, T)
    """
    from ovi.modules.bigvgan_flow_vae.bigvgan_flow_vae_export import init_vae_stat

    base = init_vae_stat(vae_ckpt_path, vae_stat_path, device=device)

    class BigVGANFlowVAEWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def train(self, mode: bool = True):
            return super().train(False)

        @property
        def sampling_rate(self) -> int:
            return int(self.vae.h.sampling_rate)

        def _vae_param_device_dtype(self) -> tuple[torch.device, torch.dtype]:
            p = next(self.vae.parameters(), None)
            if p is not None:
                return p.device, p.dtype
            b = next(self.vae.buffers(), None)
            if b is not None:
                return b.device, b.dtype
            return torch.device("cpu"), torch.float32

        @torch.no_grad()
        def wrapped_encode(self, audio: torch.Tensor) -> torch.Tensor:
            # Expect audio as (B, L) or (L,) or (B, 1, L)
            if audio.ndim == 1:
                audio = audio.view(1, 1, -1)
            elif audio.ndim == 2:
                audio = audio.unsqueeze(1)
            elif audio.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected audio shape: {tuple(audio.shape)}")
            dev, dt = self._vae_param_device_dtype()
            audio = audio.to(device=dev, dtype=dt)
            return self.vae.encode(audio)  # (B, C, T)

        @torch.no_grad()
        def wrapped_decode(self, latent: torch.Tensor) -> torch.Tensor:
            # latent: (B, C, T)
            dev, dt = self._vae_param_device_dtype()
            latent = latent.to(device=dev, dtype=dt)
            return self.vae.decode(latent)  # (B, 1, L)

    return BigVGANFlowVAEWrapper(base)

def init_text_model(ckpt_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None)


    return text_encoder

def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config = "ovi/configs/model/dit/video.json"
    audio_config = "ovi/configs/model/dit/audio.json"
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    print(
        f"Score model (Fusion) all parameters:{params_all}"
    )

    return fusion_model, video_config, audio_config

def load_fusion_checkpoint(model, checkpoint_path, from_meta=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"): 
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else: 
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else: 
        raise RuntimeError("{checkpoint=} does not exists'")