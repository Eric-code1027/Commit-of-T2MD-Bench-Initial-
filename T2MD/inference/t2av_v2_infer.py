"""
T2MD-Bench 推理 v2(ACE-Step 替换 Ovi 音频分支):

输入: CSV 文件,每行 (text_prompt, image_path)
     text_prompt 形如 '<video_desc><AUDCAP>music_desc<ENDAUDCAP>'
输出: 对每条输入生成一个 mp4

视频: OviVideoOnlyEngine(fp8 + cpu_offload)
音频: ACEStepAudioBranch(视频生成完后延迟加载,与 Ovi 共用一张卡)
合成: ovi/utils/io_utils.py:save_video()
"""
import os
import sys
import re
import gc
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import (
    format_prompt_for_filename,
    validate_and_process_user_prompt,
)
from ovi.utils.utils import get_arguments
from ovi.ovi_video_only_engine import OviVideoOnlyEngine
from ovi.audio_branches.acestep_branch import ACEStepAudioBranch


AUDCAP_RE = re.compile(r"<AUDCAP>(.*?)<ENDAUDCAP>", re.DOTALL)


def split_caption(full_prompt: str):
    """从 '<video_desc><AUDCAP>music_desc<ENDAUDCAP>' 拆出两段"""
    m = AUDCAP_RE.search(full_prompt)
    if m:
        audio_cap = m.group(1).strip()
        video_cap = AUDCAP_RE.sub("", full_prompt).strip()
    else:
        audio_cap = full_prompt
        video_cap = full_prompt
    return video_cap, audio_cap


def _free_video_engine_gpu(video_engine):
    """把 Ovi 的所有 GPU 权重 offload 回 CPU,释放显存给 ACE-Step"""
    try:
        video_engine.offload_to_cpu(video_engine.model)
    except Exception as e:
        logging.warning(f"offload model failed: {e}")
    try:
        video_engine.offload_to_cpu(video_engine.vae_model_video.model)
    except Exception as e:
        logging.warning(f"offload vae failed: {e}")
    try:
        video_engine.offload_to_cpu(video_engine.text_model.model)
    except Exception as e:
        logging.warning(f"offload text failed: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main(config, args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # === 1. 决定卡分配 ===
    video_device = int(config.get("video_device", 0))
    audio_device = config.get("audio_device", "cuda:0")
    if isinstance(audio_device, int):
        audio_device = f"cuda:{audio_device}"
    logging.info(f"video on cuda:{video_device}, audio on {audio_device}")

    # === 2. 视频分支(立即加载到 GPU) ===
    torch.cuda.set_device(video_device)
    video_engine = OviVideoOnlyEngine(
        config=config, device=video_device, target_dtype=torch.bfloat16
    )

    # === 3. 音频分支参数(延迟加载,推理循环里第一次用到时才初始化) ===
    acestep_root = config.get("acestep_project_root")
    acestep_config = config.get("acestep_config_path", "acestep-v15-turbo")
    acestep_offload = bool(config.get("acestep_offload_to_cpu", False))
    assert acestep_root and os.path.isdir(acestep_root), (
        f"acestep_project_root invalid: {acestep_root}"
    )
    audio_branch = None   # 占位,首次进入循环时再实例化

    # === 4. 解析输入 csv ===
    text_prompts, image_paths = validate_and_process_user_prompt(
        config.get("text_prompt"),
        config.get("image_path"),
        mode=config.get("mode"),
    )
    if config.get("mode") != "i2v":
        image_paths = [None] * len(text_prompts)
    else:
        for p in image_paths:
            assert p and os.path.isfile(p), f"image_path invalid: {p}"

    output_dir = config.get("output_dir", "./outputs/t2av_v2_demo")
    os.makedirs(output_dir, exist_ok=True)

    duration = float(config.get("audio_duration", 5.04))

    # === 5. 推理循环 ===
    for idx, (full_prompt, image_path) in enumerate(
        tqdm(list(zip(text_prompts, image_paths)), desc="examples")
    ):
        video_cap, audio_cap = split_caption(full_prompt)
        seed = int(config.get("seed", 100)) + idx

        logging.info(f"[idx={idx}] video_cap (head): {video_cap[:120]}")
        logging.info(f"[idx={idx}] audio_cap (head): {audio_cap[:120]}")

        # ---- 视频 ----
        torch.cuda.set_device(video_device)
        gen_video, _flux_img = video_engine.generate(
            text_prompt=full_prompt,    # Ovi 训练时是完整 prompt,推理也用完整
            image_path=image_path,
            video_frame_height_width=config.get("video_frame_height_width"),
            seed=seed,
            solver_name=config.get("solver_name", "unipc"),
            sample_steps=config.get("num_steps", 50),
            shift=config.get("shift", 5.0),
            video_guidance_scale=config.get("video_guidance_scale", 4.0),
            slg_layer=config.get("slg_layer", 11),
            video_negative_prompt=config.get("video_negative_prompt", ""),
        )
        if gen_video is None:
            logging.error(f"[idx={idx}] video gen failed")
            continue

        # ---- 视频完成后,把 Ovi 所有 GPU 权重 offload 到 CPU,腾显存给 ACE-Step ----
        _free_video_engine_gpu(video_engine)
        logging.info(
            f"[idx={idx}] video done, GPU mem freed. "
            f"current allocated: {torch.cuda.memory_allocated(video_device)/1e9:.2f} GB"
        )

        # ---- 音频(首次进入才实例化 ACE-Step) ----
        if audio_branch is None:
            logging.info(f"[idx={idx}] lazy-loading ACE-Step on {audio_device} ...")
            audio_branch = ACEStepAudioBranch(
                acestep_project_root=acestep_root,
                config_path=acestep_config,
                device=audio_device,
                offload_to_cpu=acestep_offload,
            )

        gen_audio = audio_branch.generate(
            music_prompt=audio_cap,
            duration=duration,
            seed=seed,
            infer_steps=config.get("acestep_infer_steps", 8),
        )

        # ---- 长度对齐 ----
        n_frames = gen_video.shape[1]   # (C, F, H, W)
        target_audio_len = int(round(n_frames / 24.0 * 16000))
        if len(gen_audio) > target_audio_len:
            gen_audio = gen_audio[:target_audio_len]
        elif len(gen_audio) < target_audio_len:
            gen_audio = np.pad(gen_audio, (0, target_audio_len - len(gen_audio)))

        # ---- 写文件 ----
        formatted = format_prompt_for_filename(full_prompt)
        h, w = config.get("video_frame_height_width", [448, 832])
        out_path = os.path.join(
            output_dir, f"v2__{formatted}_{h}x{w}_{seed}.mp4"
        )
        save_video(out_path, gen_video, gen_audio, fps=24, sample_rate=16000)
        logging.info(f"saved -> {out_path}")


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config, args)