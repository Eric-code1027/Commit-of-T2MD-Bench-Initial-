"""
T2MD-Bench 统一推理脚本
"""
import os, sys, re, logging
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf

from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import (
    format_prompt_for_filename, validate_and_process_user_prompt,
    preprocess_image_tensor, snap_hw_to_multiple_of_32,
)
from ovi.utils.utils import get_arguments
from ovi.utils.model_loading_utils import (
    init_text_model, init_wan_vae_2_2, load_fusion_checkpoint,
)
from ovi.utils.model_loading_utils_patch import init_fusion_acestep_model
from ovi.utils.acestep_loader import (
    decode_latent_to_audio, build_context_latents_default,
)
from ovi.modules.fusion_acestep import build_ace_encoder_hidden
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


AUDCAP_RE = re.compile(r"<AUDCAP>(.*?)<ENDAUDCAP>", re.DOTALL)


def split_caption(full):
    m = AUDCAP_RE.search(full)
    if m:
        return AUDCAP_RE.sub("", full).strip(), m.group(1).strip()
    return full, full


def build_engine(config):
    device = int(config.get("video_device", 0))
    torch.cuda.set_device(device)

    model, video_cfg, audio_cfg = init_fusion_acestep_model(
        acestep_project_root=config["acestep_project_root"],
        device=f"cuda:{device}",
    )

    vae_v = init_wan_vae_2_2(config.ckpt_dir, rank=device)
    vae_v.model.requires_grad_(False).eval().to(torch.bfloat16)

    # Ovi 预训练权重(只装 video 塔部分)
    ovi_ckpt = config.get("ovi_ckpt", os.path.join(config.ckpt_dir, "Ovi", "model.safetensors"))
    if os.path.exists(ovi_ckpt):
        load_fusion_checkpoint(model, checkpoint_path=ovi_ckpt, from_meta=False, strict=False)

    # 训练好的 fusion / LoRA(可选)
    lora_ckpt = config.get("lora_ckpt")
    if lora_ckpt and os.path.exists(lora_ckpt):
        from safetensors.torch import load_file
        sd = load_file(lora_ckpt)
        miss, unexp = model.load_state_dict(sd, strict=False)
        logging.info(f"fusion ckpt loaded missing={len(miss)} unexpected={len(unexp)}")

    model = model.to(dtype=torch.bfloat16).to(f"cuda:{device}").eval()
    model.set_rope_params()

    text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=False)
    return model, vae_v, text_model, video_cfg, audio_cfg, device


@torch.inference_mode()
def generate_one(model, vae_v, text_model, prompt_full, image_path, config, device, seed):
    h, w = config.get("video_frame_height_width", [448, 832])
    h, w = snap_hw_to_multiple_of_32(h, w, area=max(h * w, 720 * 720))
    lh, lw = h // 16, w // 16
    L_v = 31

    # ---- text(T5 给 video 塔用) ----
    text_pos, text_neg = text_model([prompt_full, config.get("video_negative_prompt", "")],
                                    text_model.device)
    text_pos = text_pos.bfloat16().to(device)
    text_neg = text_neg.bfloat16().to(device)

    # ---- ACE-Step encoder_hidden_states(text + lyric 占位) ----
    ace_h_pos, ace_m_pos = build_ace_encoder_hidden(
        model, text_caption=prompt_full, lyrics="[Instrumental]", device=device,
    )
    ace_h_neg, ace_m_neg = build_ace_encoder_hidden(
        model, text_caption=config.get("video_negative_prompt", ""),
        lyrics="[Instrumental]", device=device,
    )

    # ---- i2v 首帧 ----
    is_i2v = image_path is not None
    img_lat = None
    if is_i2v:
        first_frame = preprocess_image_tensor(image_path, device, torch.bfloat16)
        img_lat = vae_v.wrapped_encode(first_frame[:, :, None]).bfloat16().squeeze(0)
        lh, lw = img_lat.shape[2], img_lat.shape[3]

    # ---- 初始 latent ----
    g = torch.Generator(device=device).manual_seed(seed)
    vid_noise = torch.randn((48, L_v, lh, lw), device=device, dtype=torch.bfloat16, generator=g)

    # ACE-Step audio latent 长度:duration(s) * 48000(sr) / ~512(VAE 下采样) ≈ duration * 93.75
    # 推荐方式:用 silence_latent 的 latent_length 当锚
    duration = float(config.get("audio_duration", 5.04))
    T_a = int(round(duration * 48000 / 512))   # ≈ 472 for 5.04s
    audio_noise = torch.randn((1, T_a, 64), device=device, dtype=torch.bfloat16, generator=g)

    ace_ctx = build_context_latents_default(
        model.acestep_handler, latent_length=T_a,
        device=device, dtype=torch.bfloat16,
    )
    ace_attn = torch.ones(1, T_a, device=device, dtype=torch.bfloat16)

    ph, pw = model.video_model.patch_size[1], model.video_model.patch_size[2]
    max_v_len = L_v * lh * lw // (ph * pw)

    # ---- scheduler ----
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(config.get("num_steps", 50), device=device,
                      shift=config.get("shift", 5.0))

    prev_audio_h = None
    cfg_v = config.get("video_guidance_scale", 4.0)
    cfg_a = config.get("audio_guidance_scale", 3.0)

    # ---- joint denoise loop ----
    for t in tqdm(sch.timesteps, desc="joint denoise"):
        ti = torch.full((1,), t, device=device)
        if is_i2v:
            vid_noise[:, :1] = img_lat

        # CFG 正向
        vp, ap, prev_audio_h = model(
            vid=[vid_noise], audio_latent=audio_noise, t=ti,
            vid_context=[text_pos],
            ace_encoder_hidden_states=ace_h_pos,
            ace_encoder_attention_mask=ace_m_pos,
            ace_context_latents=ace_ctx,
            ace_attention_mask=ace_attn,
            vid_seq_len=max_v_len,
            prev_audio_hidden=prev_audio_h,
        )
        # CFG 负向
        vn, an, _ = model(
            vid=[vid_noise], audio_latent=audio_noise, t=ti,
            vid_context=[text_neg],
            ace_encoder_hidden_states=ace_h_neg,
            ace_encoder_attention_mask=ace_m_neg,
            ace_context_latents=ace_ctx,
            ace_attention_mask=ace_attn,
            vid_seq_len=max_v_len,
            prev_audio_hidden=prev_audio_h,
        )

        v_pred = vn + cfg_v * (vp - vn)
        a_pred = an + cfg_a * (ap - an)

        vid_noise = sch.step(v_pred.unsqueeze(0), t, vid_noise.unsqueeze(0),
                              return_dict=False)[0].squeeze(0)
        audio_noise = sch.step(a_pred, t, audio_noise, return_dict=False)[0]

    if is_i2v:
        vid_noise[:, :1] = img_lat

    # ---- decode ----
    gen_video = vae_v.wrapped_decode(vid_noise.unsqueeze(0)).squeeze(0).cpu().float().numpy()
    audio_wav = decode_latent_to_audio(model.acestep_handler, audio_noise)
    # AutoencoderOobleck 输出 48kHz,sample_rate 默认 stereo → mono + 重采样到 16k
    audio_wav = audio_wav.squeeze(0).mean(0).cpu().float().numpy()
    import librosa
    audio_wav = librosa.resample(audio_wav, orig_sr=48000, target_sr=16000)

    # 长度对齐
    n_frames = gen_video.shape[1]
    target_len = int(round(n_frames / 24.0 * 16000))
    if len(audio_wav) > target_len:
        audio_wav = audio_wav[:target_len]
    elif len(audio_wav) < target_len:
        audio_wav = np.pad(audio_wav, (0, target_len - len(audio_wav)))
    return gen_video, audio_wav.astype(np.float32)


def main(config, args):
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s")
    model, vae_v, text_model, _, _, device = build_engine(config)

    prompts, images = validate_and_process_user_prompt(
        config.get("text_prompt"), config.get("image_path"),
        mode=config.get("mode"),
    )
    if config.get("mode") != "i2v":
        images = [None] * len(prompts)

    out_dir = config.get("output_dir", "./outputs/t2av_fusion")
    os.makedirs(out_dir, exist_ok=True)

    for idx, (p, img) in enumerate(tqdm(list(zip(prompts, images)), desc="examples")):
        seed = int(config.get("seed", 100)) + idx
        gv, ga = generate_one(model, vae_v, text_model, p, img, config, device, seed)
        h, w = config.get("video_frame_height_width", [448, 832])
        fn = format_prompt_for_filename(p)
        out = os.path.join(out_dir, f"fusion__{fn}_{h}x{w}_{seed}.mp4")
        save_video(out, gv, ga, fps=24, sample_rate=16000)
        logging.info(f"saved -> {out}")


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config, args)
