import torch, os, json
from diffsynth import load_state_dict
from diffsynth.utils import BasePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.trainers.t2a_dataset import build_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 方法1: 设置环境变量
os.environ["OMP_NUM_THREADS"] = "4"        # OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "4"        # MKL线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # OpenBLAS线程数
os.environ['NCCL_TIMEOUT'] = '3600'

# 在导入torch后也可以设置
torch.set_num_threads(4)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import os
import sys
import uuid
import random
import time
import cv2
import glob
import torch
import logging
import shutil
from textwrap import indent
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from diffusers import FluxPipeline
from tqdm import tqdm
from scipy.io import wavfile
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
from ovi.modules.fusion import FusionModel
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


def init_score_model_ovi(device, meta_init=False):
    video_config = None
    audio_config = DEFAULT_CONFIG.get("audio_dit_config", "ovi/configs/model/dit/audio.json")
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

class T2AModel(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, config=DEFAULT_CONFIG, target_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        meta_init = False
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_score_model_ovi(device, meta_init=meta_init)

        fp8 = config.get("fp8", False)
        int8 = config.get("qint8", False)
        if fp8:
            assert not config.get("mode") == "t2i2v", "Image generation with FluxPipeline is not supported with fp8 quantization. This is because if you are unable to run the bf16 model, you likely cannot run image gen model"

        if not meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = (
                model.to(device=device if not self.cpu_offload else "cpu")
                # .eval()
            )

        # Load VAEs
        # 如果提供了 audio_vae_ckpt 和 audio_vae_stat，使用 BigVGAN Flow VAE (BigVGANFlowVAE) 加载方式
        audio_vae_ckpt = config.get("audio_vae_ckpt", None)
        audio_vae_stat = config.get("audio_vae_stat", None)
        
        if audio_vae_ckpt and audio_vae_stat:
            from ovi.utils.model_loading_utils import init_bigvgan_flow_vae
            print(f"Using BigVGAN Flow VAE loading method")
            print(f"  VAE checkpoint: {audio_vae_ckpt}")
            print(f"  Stat file: {audio_vae_stat}")
            vae_model_audio = init_bigvgan_flow_vae(audio_vae_ckpt, audio_vae_stat, device=device)
        else:
            # 使用默认的 MMAudio VAE 加载方式
            vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        
        vae_model_audio.requires_grad_(False).eval()
        # NOTE: BigVGANFlowVAE contains vocoder; keep it in fp32 for stability/quality.
        if audio_vae_ckpt and audio_vae_stat:
            self.vae_model_audio = vae_model_audio
        else:
            self.vae_model_audio = vae_model_audio.bfloat16()

        # NOTE: this hack only applies to the legacy MMAudio VAE structure (vae_model_audio.tod.vae/decoder/vocoder).
        # BigVGANFlowVAEWrapper does not have these attributes.
        if not (audio_vae_ckpt and audio_vae_stat):
            with torch.no_grad():
                if hasattr(vae_model_audio, "tod") and hasattr(vae_model_audio.tod, "vae") and hasattr(vae_model_audio.tod.vae, "decoder"):
                    delattr(vae_model_audio.tod.vae, "decoder")
                if hasattr(vae_model_audio, "tod") and hasattr(vae_model_audio.tod, "vocoder"):
                    delattr(vae_model_audio.tod, "vocoder")
        # with torch.no_grad(): # TODO check if ok for vae reconstruction
        #     old = vae_model_audio.tod.vae.encoder.learnable_gain
        #     vae_model_audio.tod.vae.encoder.learnable_gain = nn.Parameter(old.view(1))
        #     vae_model_audio.tod.vae.encoder.learnable_gain.requires_grad = False   # 保持冻结

        # Load T5 text model
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # NOTE train from scratch
        # # Find fusion ckpt in the same dir used by other components
        # checkpoint_path = os.path.join(
        #     config.ckpt_dir,
        #     "Ovi",
        #     "model.safetensors" if not fp8 else "model_fp8_e4m3fn.safetensors",
        # )

        # if not os.path.exists(checkpoint_path):
        #     raise RuntimeError(f"No fusion checkpoint found in {config.ckpt_dir}")

        # load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu") # .eval()
            model.set_rope_params()
        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        # logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)


    def forward(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (len(inputs['audio']),))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        text_embeddings = self.text_model(inputs['prompt'], self.text_model.device)
        text_embeddings = [emb.to(self.target_dtype) for emb in text_embeddings]

        latents, targets = [], []
        for i, audio in enumerate(inputs['audio']):
            x0 = self.vae_model_audio.wrapped_encode(audio.float().to(self.device)).transpose(1, 2).squeeze(0)
            x1 = torch.randn_like(x0)
            xt = self.scheduler.add_noise(x0, x1, None, timestep_id[i])
            v = self.scheduler.training_target(x0, x1)
            latents.append(xt.to(self.torch_dtype))
            targets.append(v)
        xlens = [x.shape[0] for x in latents]

        vid_pred, audio_pred = self.model(
            vid=None,
            audio=latents,
            t=timestep,
            vid_context=None,
            audio_context=text_embeddings,
            vid_seq_len=None,
            audio_seq_len=max(xlens),
        )
        
        weight = self.scheduler.training_weight(None, timestep_id)
        losses = []
        for i in range(len(audio_pred)):
            loss = torch.nn.functional.mse_loss(audio_pred[i].float(), targets[i].float()) * weight[i]
            losses.append(loss)
        return loss.mean()


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=False, # NOTE
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        condition_dropout=0.0,
    ):
        super().__init__()
        # Load models
        self.pipe = T2AModel(device='cuda')
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.pipe.model.set_gradient_checkpointing(self.use_gradient_checkpointing)
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.condition_dropout = condition_dropout
        if condition_dropout > 0:
            print(f'Enable condition dropout: {condition_dropout}')

    def forward(self, data, inputs=None):
        if self.condition_dropout > 0:
            for i in range(len(data['prompt'])):
                if random.random() < self.condition_dropout:
                    data['prompt'][i] = ''
        loss = self.pipe(**data)
        return loss

    def test_vae(self, data):
        for i in range(len(data['uid'])):
            text_embeddings = self.pipe.text_model(data['prompt'][i], self.pipe.text_model.device)
            latent_audio = self.pipe.vae_model_audio.wrapped_encode(data['audio'][i].cuda()).transpose(1, 2)

            # Decode audio
            audio_latents_for_vae = latent_audio.transpose(1, 2)  # 1, c, l
            generated_audio = self.pipe.vae_model_audio.wrapped_decode(audio_latents_for_vae)
            generated_audio = generated_audio.squeeze().cpu().float().numpy()
            # import pdb; pdb.set_trace()
            uttid = data['uid'][i]
            shutil.copy(data['audio_path'][i], f'/tmp/test_vae/{uttid}.wav')
            wavfile.write(f'/tmp/test_vae/{uttid}-vae.wav', 16000, (generated_audio*32767).astype(np.int16))

if __name__ == "__main__":
    set_seed(777)
    parser = wan_parser()
    parser.add_argument(
        "--audio_dit_config",
        type=str,
        default=None,
        help="Path to OVI audio DiT config json. Must match VAE latent channels (e.g. 64).",
    )
    args = parser.parse_args()

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # fsdp_plugin = FullyShardedDataParallelPlugin(
    #         sharding_strategy="HYBRID_SHARD",
    #         auto_wrap_policy=size_based_auto_wrap_policy,
    #         use_orig_params=True,          # <-- 必须打开
    #         state_dict_type="FULL_STATE_DICT",
    #         cpu_offload=False
    # )
    # accelerator = Accelerator(fsdp_plugin=fsdp_plugin, gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16")
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=gradient_accumulation_steps,)
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 16

    # Override ckpt_dir if provided via command line
    ckpt_dir = getattr(args, 'ckpt_dir', None)
    if ckpt_dir is not None:
        DEFAULT_CONFIG.ckpt_dir = ckpt_dir
    
    # Override VAE loading config if provided
    audio_vae_ckpt = getattr(args, 'audio_vae_ckpt', None)
    audio_vae_stat = getattr(args, 'audio_vae_stat', None)

    # Override audio DiT config if provided
    audio_dit_config = getattr(args, "audio_dit_config", None)
    if audio_dit_config is not None:
        DEFAULT_CONFIG.audio_dit_config = audio_dit_config
    
    if audio_vae_ckpt is not None:
        DEFAULT_CONFIG.audio_vae_ckpt = audio_vae_ckpt
    if audio_vae_stat is not None:
        DEFAULT_CONFIG.audio_vae_stat = audio_vae_stat

    data_path = getattr(args, 'data_path', None)
    data_list = None
    conf = {
        'seed': 777,
        'data_path': data_path,
        'shuffle': True,
        'shuffle_conf': {
            'shuffle_size': 5000,
            'online': True,
        },
        'sort': True,
        'sort_conf': {
            'key': 'audio',
            'sort_size': 1000,
            'online': True,
        },
        'batch_conf': {
            'batch_type': 'dynamic',
            'batch_size': 120, # TODO change to 200, the same effective duration.
        },
    }
    dataset = build_dataset(data_list, conf)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=8 if num_workers > 0 else None,  # 100 易 OOM；多进程时 8 足够
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
    # launch_training_task(dataset, model, model_logger, args=args)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    optimizer.zero_grad()

    global_step = 0
    global_loss = 0
    global_duration = 0
    rank = accelerator.process_index
    set_seed(777 + rank) # NOTE make sure each rank different !!!
    start_time = time.time()
    for epoch_id in range(num_epochs):
        print(f'Start epoch {epoch_id}')
        for data in dataloader:
            # global_step += 1
            # elapsed_time = time.time() - start_time
            # start_time = time.time()
            # print(f'RANK{rank} Step{global_step} Time {elapsed_time:.2f}s')
            # continue

            with accelerator.accumulate(model):
                loss = model(data)
                global_loss += loss.detach()
                global_duration += sum(data['duration'])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    global_loss = accelerator.reduce(global_loss, reduction='mean') / accelerator.gradient_accumulation_steps
                    # global_duration = accelerator.reduce(global_duration, reduction='sum')
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    model_logger.on_step_end(accelerator, model, save_steps)
                    step = model_logger.num_steps
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    if accelerator.is_main_process:
                        print(f'Epoch {epoch_id} Step {step}: Loss {global_loss.item():.3f} \tNorm {gnorm:.3f} \tDuration {global_duration:.2f}s \tTime {elapsed_time:.2f}s')
                    global_loss = 0
                    global_duration = 0
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
