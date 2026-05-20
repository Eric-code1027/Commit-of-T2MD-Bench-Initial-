import os
import sys
import json
import uuid
import cv2
import glob
import torch
import logging
from textwrap import indent
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm
from ovi.modules.fusion import FusionModel
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint, init_bigvgan_flow_vae
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible

from optimum.quanto import freeze, qint8, quantize

DEFAULT_CONFIG = OmegaConf.load('ovi/configs/inference/inference_audio.yaml')


def init_score_audio(device, meta_init=False):
    video_config = None
    audio_config = "ovi/configs/model/dit/audio_latent64.json"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        with torch.device(device):
            fusion_model = FusionModel(video_config, audio_config)
    
    params_all = sum(p.numel() for p in fusion_model.parameters())
    
    print(
        f"Score model (Fusion) all parameters:{params_all}"
    )

    return fusion_model, video_config, audio_config

class OviAudioEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        meta_init = True
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_score_audio(device, meta_init=meta_init)

        fp8 = config.get("fp8", False)
        int8 = config.get("qint8", False)
        if fp8:
            assert not config.get("mode") == "t2i2v", "Image generation with FluxPipeline is not supported with fp8 quantization. This is because if you are unable to run the bf16 model, you likely cannot run image gen model"

        if not meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = (
                model.to(device=device if not self.cpu_offload else "cpu")
                .eval()
            )

        # Load VAEs
        vae_ckpt_path = config.get("audio_vae_ckpt", "./models/vae/g_01240000")
        vae_stat_path = config.get("audio_vae_stat", "./models/vae/global_mean_var_124w.stat")
        vae_model_audio = init_bigvgan_flow_vae(vae_ckpt_path, vae_stat_path, device=device)
        # vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        # vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # Find fusion ckpt in the same dir used by other components
        checkpoint_path = os.path.join(
            config.ckpt_dir,
            "Ovi",
            "model.safetensors" if not fp8 else "model_fp8_e4m3fn.safetensors",
        )
        checkpoint_path = config.get('ovi_ckpt', checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")


        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        ## Load t2i as part of pipeline
        self.image_model = None
        
        if config.get("mode") == "t2i2v":
            logging.info(f"Loading Flux Krea for first frame generation...")
            self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
            self.image_model.enable_model_cpu_offload(gpu_id=self.device) #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")

        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    audio_negative_prompt="",
                    audio_duration=5.04,
                ):

        params = {
            "Text Prompt": text_prompt,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Audio Negative Prompt": audio_negative_prompt,
            "Audio Duration": audio_duration,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )

            if self.cpu_offload:
                self.text_model.model = self.text_model.model.to(self.device)
            text_embeddings = self.text_model([text_prompt, audio_negative_prompt], self.text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)

            # Split embeddings
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_audio_neg = text_embeddings[1]

            audio_latent_length = int(audio_duration * 16000 / 512)
            audio_noise = torch.randn((audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            
            # Sampling loop
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_audio)
                self.model = self.model.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for i, t_a in tqdm(enumerate(timesteps_audio)):
                    timestep_input = torch.full((1,), t_a, device=self.device)

                    # Positive (conditional) forward pass
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': None,
                        'vid_seq_len': None,
                        'audio_seq_len': max_seq_len_audio,
                    }

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=None,
                        audio=[audio_noise],
                        t=timestep_input,
                        **pos_forward_args
                    )
                    
                    # Negative (unconditional) forward pass  
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': None,
                        'vid_seq_len': None,
                        'audio_seq_len': max_seq_len_audio,
                        'slg_layer': slg_layer
                    }
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        vid=None,
                        audio=[audio_noise],
                        t=timestep_input,
                        **neg_forward_args
                    )

                    # Apply classifier-free guidance
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                    self.vae_model_audio = self.vae_model_audio.to(self.device)

                # Decode audio
                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_audio)
            return generated_audio


        except Exception as e:
            logging.error(traceback.format_exc())
            return None
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
