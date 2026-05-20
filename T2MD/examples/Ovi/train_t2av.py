import torch, os, json
from diffsynth import load_state_dict
from diffsynth.utils import BasePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.trainers.t2mv_dataset import AudioVideoDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ['NCCL_TIMEOUT'] = '3600'

torch.set_num_threads(4)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import os
import sys
import uuid
import random
import time
import cv2
import glob
import torch
import logging
from textwrap import indent
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from diffusers import FluxPipeline
from tqdm import tqdm
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    init_text_model,
    init_mmaudio_vae,
    init_bigvgan_flow_vae,
    init_wan_vae_2_2,
    load_fusion_checkpoint,
)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible

from optimum.quanto import freeze, qint8, quantize

from diffsynth.schedulers.flow_match import FlowMatchScheduler

DEFAULT_CONFIG = OmegaConf.load('ovi/configs/training/finetune.yaml')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# 改动 #1: 通过 env var 控制是否走 video-only 训练
# ============================================================================
VIDEO_ONLY_TRAINING = True


class OviModel(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, config=DEFAULT_CONFIG, target_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        meta_init = True
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank='cpu', meta_init=meta_init)

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
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        # ====================================================================
        # 改动 #2: video-only 模式下完全跳过 audio VAE 加载,省 ~2GB 显存
        # ====================================================================
        if VIDEO_ONLY_TRAINING:
            print("[VIDEO_ONLY_TRAINING] Skipping audio VAE loading.")
            self.vae_model_audio = None
        else:
            audio_vae_ckpt = config.get("audio_vae_ckpt", None)
            audio_vae_stat = config.get("audio_vae_stat", None)
            if audio_vae_ckpt and audio_vae_stat:
                print(f"Using BigVGAN Flow VAE loading method")
                vae_model_audio = init_bigvgan_flow_vae(audio_vae_ckpt, audio_vae_stat, device=device)
                vae_model_audio.requires_grad_(False).eval()
                self.vae_model_audio = vae_model_audio
            else:
                vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
                vae_model_audio.requires_grad_(False).eval()
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

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")

        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu")
            model.set_rope_params()
        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        self.image_model = None
        if config.get("mode") == "t2i2v":
            logging.info(f"Loading Flux Krea for first frame generation...")
            self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
            self.image_model.enable_model_cpu_offload(gpu_id=self.device)

        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.audio_latent_length = 157
        self.video_latent_length = 31

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

    def offload_to_cpu(self, model):
        """ZeRO-3 模式下不真的搬模型,只清显存"""
        try:
            model.cpu()
        except Exception:
            pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return model

    def forward(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        # ---- 1. fusion 先下 CPU,腾位置给 T5 ----
        if self.cpu_offload:
            self.offload_to_cpu(self.model)
            self.text_model.model = self.text_model.model.to(self.device)

        text_embeddings = self.text_model(inputs['prompt'], self.device)
        text_embeddings = [emb.to(self.target_dtype) for emb in text_embeddings]

        # ---- 2. T5 下 CPU,fusion 上 GPU ----
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # ---- 3. VAE encode video ----
        latent_video = self.vae_model_video.wrapped_encode(inputs['video']).to(self.torch_dtype)
        noise_video = torch.randn_like(latent_video)
        vid_t = self.scheduler.add_noise(latent_video, noise_video, timestep)
        vid_target = self.scheduler.training_target(latent_video, noise_video, timestep)

        _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_video = vid_t.shape[2] * vid_t.shape[3] * vid_t.shape[4] // (_patch_size_h*_patch_size_w)

        # ---- 4. fusion forward(此时 GPU 上只有 fusion + activation) ----
        if VIDEO_ONLY_TRAINING:
            vid_pred, _ = self.model(
                vid=vid_t,
                audio=None,
                t=timestep,
                vid_context=text_embeddings,
                audio_context=None,
                vid_seq_len=max_seq_len_video,
                audio_seq_len=None,
            )
            weight = self.scheduler.training_weight(timestep)
            loss_video = torch.nn.functional.mse_loss(vid_pred[0].float(), vid_target[0].float()) * weight
            loss = loss_video
            if accelerator.is_main_process:
                print(f'[INFO] loss_video {loss_video.item():.4f}')
            torch.cuda.empty_cache()
            return loss

        # === 原 audio+video 联合路径,保留 ===
        latent_audio = self.vae_model_audio.wrapped_encode(inputs['audio']).transpose(1, 2).to(self.torch_dtype)
        noise_audio = torch.randn_like(latent_audio)
        audio_t = self.scheduler.add_noise(latent_audio, noise_audio, timestep)
        audio_target = self.scheduler.training_target(latent_audio, noise_audio, timestep)
        vid_pred, audio_pred = self.model(
            vid=vid_t, audio=audio_t, t=timestep,
            vid_context=text_embeddings, audio_context=text_embeddings,
            vid_seq_len=max_seq_len_video, audio_seq_len=audio_t.shape[1],
        )
        weight = self.scheduler.training_weight(timestep)
        loss_video = torch.nn.functional.mse_loss(vid_pred[0].float(), vid_target[0].float()) * weight
        loss_audio = torch.nn.functional.mse_loss(audio_pred[0].float(), audio_target[0].float()) * weight
        loss = 0.85 * loss_video + 0.15 * loss_audio
        if accelerator.is_main_process:
            print(f'[INFO] loss_audio {loss_audio.item():.3f} loss_video {loss_video.item():.3f}')
        return loss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        condition_dropout=0.0,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = OviModel(device='cuda')

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.pipe.model.gradient_checkpointing = self.use_gradient_checkpointing
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.condition_dropout = condition_dropout
        if condition_dropout > 0:
            print(f'Enable condition dropout: {condition_dropout}')

    def forward(self, data, inputs=None):
        if self.condition_dropout > 0:
            if random.random() < self.condition_dropout:
                data['prompt'] = ""
        loss = self.pipe(**data)
        return loss


if __name__ == "__main__":
    set_seed(777)
    parser = wan_parser()
    args = parser.parse_args()

    audio_vae_ckpt = getattr(args, "audio_vae_ckpt", None)
    audio_vae_stat = getattr(args, "audio_vae_stat", None)
    if audio_vae_ckpt is not None:
        DEFAULT_CONFIG.audio_vae_ckpt = audio_vae_ckpt
    if audio_vae_stat is not None:
        DEFAULT_CONFIG.audio_vae_stat = audio_vae_stat

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    if accelerator.is_main_process:
        print(f"=" * 60)
        print(f"VIDEO_ONLY_TRAINING = {VIDEO_ONLY_TRAINING}")
        print(f"  (set VIDEO_ONLY_TRAINING=0 to enable joint audio+video training)")
        print(f"=" * 60)

    dataset = AudioVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        csv_path=args.dataset_csv_path,
        dynamic_duration=args.dynamic_duration,
        repeat=args.dataset_repeat,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        condition_dropout=args.condition_dropout,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    def filter_collate(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        assert len(batch) == 1
        return batch[0]

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=filter_collate, num_workers=num_workers)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    optimizer.zero_grad()

    global_loss = 0
    rank = accelerator.process_index
    set_seed(777 + rank)
    start_time = time.time()
    for epoch_id in range(num_epochs):
        if accelerator.is_main_process:
            print(f'Start epoch {epoch_id}')
        for data in dataloader:
            if data is None:
                continue

            with accelerator.accumulate(model):
                loss = model(data)
                global_loss += loss.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    global_loss = accelerator.reduce(global_loss, reduction='mean') / accelerator.gradient_accumulation_steps
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    step = model_logger.num_steps
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    if accelerator.is_main_process:
                        print(f'Epoch {epoch_id} Step {step}: Loss {global_loss.item():.4f} \tNorm {gnorm:.3f} \tTime {elapsed_time:.2f}s')
                    global_loss = 0
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)