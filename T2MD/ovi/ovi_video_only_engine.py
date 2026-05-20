"""
仅视频生成引擎,从 ovi/ovi_fusion_engine.py 拆出。

与 OviFusionEngine 的区别:
- 不加载 MMAudio VAE(音频走 ACE-Step,VAE 不用)
- forward 时 audio=None,FusionModel 走 video-only 短路分支(fusion.py:268-274)
- 输出 (video_numpy, image),不返回 audio
"""
import os
import traceback
import logging
import torch
from tqdm import tqdm
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler

from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    init_text_model,
    init_wan_vae_2_2,
    load_fusion_checkpoint,
)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ovi.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ovi.utils.processing_utils import (
    clean_text,
    preprocess_image_tensor,
    snap_hw_to_multiple_of_32,
    scale_hw_to_area_divisible,
)


class OviVideoOnlyEngine:
    def __init__(self, config, device=0, target_dtype=torch.bfloat16):
        self.device = device
        self.target_dtype = target_dtype
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        meta_init = True

        # 1. 加载完整 FusionModel(权重里 video/audio 一起,必须整体加载)
        model, video_config, audio_config = init_fusion_score_model_ovi(
            rank=device, meta_init=meta_init
        )
        self.video_config = video_config
        self.audio_config = audio_config

        # 2. 视频 VAE(不加载 MMAudio VAE!)
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        # 3. T5
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # 4. 加载 fusion ckpt(根据 fp8 选 ckpt 文件)
        fp8 = config.get("fp8", False)
        ckpt_path = os.path.join(
            config.ckpt_dir, "Ovi",
            "model.safetensors" if not fp8 else "model_fp8_e4m3fn.safetensors",
        )
        ckpt_path = config.get("ovi_ckpt", ckpt_path)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Ovi ckpt not found: {ckpt_path}")
        load_fusion_checkpoint(model, checkpoint_path=ckpt_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model

        # 5. T2I2V 时的 Flux(我们用 i2v 不需要,但兼容保留)
        self.image_model = None
        if config.get("mode") == "t2i2v":
            self.image_model = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16
            )
            self.image_model.enable_model_cpu_offload(gpu_id=self.device)

        self.video_latent_channel = video_config.get("in_dim")
        self.video_latent_length = 31

        logging.info(
            f"OviVideoOnlyEngine ready. fp8={fp8}, cpu_offload={self.cpu_offload}, "
            f"GPU mem={torch.cuda.memory_allocated(device)/1e9:.2f}GB"
        )

    @torch.inference_mode()
    def generate(
        self,
        text_prompt,
        image_path=None,
        video_frame_height_width=None,
        seed=100,
        solver_name="unipc",
        sample_steps=50,
        shift=5.0,
        video_guidance_scale=4.0,
        slg_layer=11,
        video_negative_prompt="",
    ):
        try:
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift,
            )

            is_i2v = image_path is not None
            first_frame = None
            image = None

            if is_i2v and not self.image_model:
                first_frame = preprocess_image_tensor(
                    image_path, self.device, self.target_dtype
                )
            else:
                assert video_frame_height_width is not None
                video_h, video_w = video_frame_height_width
                snap_area = max(video_h * video_w, 720 * 720)
                video_h, video_w = snap_hw_to_multiple_of_32(
                    video_h, video_w, area=snap_area
                )
                video_latent_h, video_latent_w = video_h // 16, video_w // 16

            if self.cpu_offload:
                self.text_model.model = self.text_model.model.to(self.device)
            text_embs = self.text_model(
                [text_prompt, video_negative_prompt], self.text_model.device
            )
            text_embs = [emb.to(self.target_dtype).to(self.device) for emb in text_embs]
            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)
            text_pos, text_neg = text_embs[0], text_embs[1]

            if is_i2v:
                if self.cpu_offload:
                    self.vae_model_video.model = self.vae_model_video.model.to(self.device)
                with torch.no_grad():
                    latents_images = (
                        self.vae_model_video.wrapped_encode(first_frame[:, :, None])
                        .to(self.target_dtype)
                        .squeeze(0)
                    )
                video_latent_h, video_latent_w = (
                    latents_images.shape[2],
                    latents_images.shape[3],
                )
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)

            video_noise = torch.randn(
                (
                    self.video_latent_channel,
                    self.video_latent_length,
                    video_latent_h,
                    video_latent_w,
                ),
                device=self.device,
                dtype=self.target_dtype,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )

            ph = self.model.video_model.patch_size[1]
            pw = self.model.video_model.patch_size[2]
            max_seq_len_video = (
                video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (ph * pw)
            )

            if self.cpu_offload:
                self.model = self.model.to(self.device)
            with torch.amp.autocast(
                "cuda",
                enabled=self.target_dtype != torch.float32,
                dtype=self.target_dtype,
            ):
                for t_v in tqdm(timesteps_video, desc="video sampling"):
                    timestep_input = torch.full((1,), t_v, device=self.device)
                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # 正向: audio=None,fusion 走 video-only 分支(fusion.py:268-274)
                    pos_args = {
                        "audio_context": None,
                        "vid_context": [text_pos],
                        "vid_seq_len": max_seq_len_video,
                        "audio_seq_len": None,
                        "first_frame_is_clean": is_i2v,
                    }
                    pred_vid_pos, _ = self.model(
                        vid=[video_noise], audio=None, t=timestep_input, **pos_args
                    )

                    neg_args = {
                        "audio_context": None,
                        "vid_context": [text_neg],
                        "vid_seq_len": max_seq_len_video,
                        "audio_seq_len": None,
                        "first_frame_is_clean": is_i2v,
                        "slg_layer": slg_layer,
                    }
                    pred_vid_neg, _ = self.model(
                        vid=[video_noise], audio=None, t=timestep_input, **neg_args
                    )

                    pred = pred_vid_neg[0] + video_guidance_scale * (
                        pred_vid_pos[0] - pred_vid_neg[0]
                    )
                    video_noise = scheduler_video.step(
                        pred.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                    self.vae_model_video.model = self.vae_model_video.model.to(self.device)

                if is_i2v:
                    video_noise[:, :1] = latents_images

                video_latents = video_noise.unsqueeze(0)
                generated_video = self.vae_model_video.wrapped_decode(video_latents)
                generated_video = generated_video.squeeze(0).cpu().float().numpy()

                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)

            return generated_video, image

        except Exception:
            logging.error(traceback.format_exc())
            return None, None

    def offload_to_cpu(self, model):
        model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return model

    def get_scheduler_time_steps(
        self, sampling_steps, solver_name="unipc", device=0, shift=5.0
    ):
        torch.manual_seed(4)
        if solver_name == "unipc":
            sch = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sch.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sch.timesteps
        elif solver_name == "dpm++":
            sch = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(sch, device=device, sigmas=sampling_sigmas)
        elif solver_name == "euler":
            sch = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, _ = retrieve_timesteps(sch, sampling_steps, device=device)
        else:
            raise NotImplementedError(f"Solver {solver_name} not supported")
        return sch, timesteps