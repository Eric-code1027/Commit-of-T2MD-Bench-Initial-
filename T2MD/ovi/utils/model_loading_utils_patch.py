
import os
import json
import torch
from ovi.modules.fusion_acestep import FusionAceStepModel


def init_fusion_acestep_model(
    acestep_project_root: str,
    video_config_path: str = "ovi/configs/model/dit/video.json",
    audio_config_path: str = "ovi/configs/model/dit/audio_acestep.json",
    device: str = "cuda",
):
    """构造 Wan video + ACE-Step audio 的双塔 fusion 模型。

    Returns:
        (fusion_model, video_config, audio_config)
    """
    assert os.path.exists(video_config_path), f"missing {video_config_path}"
    assert os.path.exists(audio_config_path), f"missing {audio_config_path}"
    with open(video_config_path) as f:
        video_config = json.load(f)
    with open(audio_config_path) as f:
        audio_config = json.load(f)

    fusion_model = FusionAceStepModel(
        video_config=video_config,
        audio_config=audio_config,
        acestep_project_root=acestep_project_root,
        device=device,
    )
    return fusion_model, video_config, audio_config
