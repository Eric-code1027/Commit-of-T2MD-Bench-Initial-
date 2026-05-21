"""
T2MD-Bench 训练脚本(ACE-Step fusion 版,真实可跑路径)
"""
import os, json, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['NCCL_TIMEOUT'] = '3600'
torch.set_num_threads(4)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf

from diffsynth.utils import BasePipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.t2mv_dataset import AudioVideoDataset
from diffsynth.schedulers.flow_match import FlowMatchScheduler

# === Ovi 原有的工具(保留) ===
from ovi.utils.model_loading_utils import (
    init_text_model, init_wan_vae_2_2, load_fusion_checkpoint,
)
# === 新加的 ACE-Step fusion 入口 ===
from ovi.utils.model_loading_utils_patch import init_fusion_acestep_model
from ovi.utils.acestep_loader import (
    encode_audio_to_latent, build_context_latents_default,
)
from ovi.modules.fusion_acestep import build_ace_encoder_hidden

DEFAULT_CONFIG = OmegaConf.load("ovi/configs/training/finetune.yaml")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class OviAceStepModel(BasePipeline):
    """Wan video + ACE-Step audio 双塔训练模型。"""

    def __init__(self, device="cuda", target_dtype=torch.bfloat16, config=DEFAULT_CONFIG):
        super().__init__(
            device=device, torch_dtype=target_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.device = device
        self.target_dtype = target_dtype
        acestep_root = config["acestep_project_root"]

        # 1. fusion 主模型(Wan video + ACE-Step 全套)
        # 注意:init_fusion_acestep_model 内部已经通过 AceStepHandler 加载完
        # ACE-Step 所有组件(DiT/VAE/Encoder/text_encoder/lyric...)
        model, video_config, audio_config = init_fusion_acestep_model(
            acestep_project_root=acestep_root,
            device=device,
        )
        self.model = model
        self.video_config = video_config
        self.audio_config = audio_config

        # 2. 视频 VAE(沿用)
        self.vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        self.vae_model_video.model.requires_grad_(False).eval()
        self.vae_model_video.model = self.vae_model_video.model.bfloat16()

        # 3. T5 文本编码器(给 Ovi video 塔用,沿用)
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=False)

        # 4. 加载 Ovi 预训练权重(只灌 video 塔部分,strict=False 跳过 audio 塔)
        ovi_ckpt = os.path.join(config.ckpt_dir, "Ovi",
                                "model_fp8_e4m3fn.safetensors" if config.get("fp8") else "model.safetensors")
        if os.path.exists(ovi_ckpt):
            load_fusion_checkpoint(self.model, checkpoint_path=ovi_ckpt,
                                   from_meta=False, strict=False)

        if not config.get("fp8"):
            self.model = self.model.to(dtype=target_dtype)
        self.model = self.model.to(device)
        self.model.set_rope_params()

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

    def forward(self, **inputs):
        device = self.device
        # === 1. timestep ===
        t_max = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        t_min = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        t_id = torch.randint(t_min, t_max, (1,))
        t = self.scheduler.timesteps[t_id].to(self.target_dtype).to(device)

        # === 2. 视频侧:T5 text + Wan VAE encode ===
        text_emb = self.text_model([inputs["prompt"]], self.text_model.device)
        text_emb = [e.to(self.target_dtype).to(device) for e in text_emb]

        lat_v = self.vae_model_video.wrapped_encode(inputs["video"]).to(self.target_dtype)
        n_v = torch.randn_like(lat_v)
        vid_t = self.scheduler.add_noise(lat_v, n_v, t)
        vid_target = self.scheduler.training_target(lat_v, n_v, t)

        # === 3. 音频侧:ACE-Step VAE encode + 构造 encoder/context ===
        # inputs["audio"] : [B, C, S] @ 48kHz
        with torch.no_grad():
            lat_a = encode_audio_to_latent(self.model.acestep_handler,
                                            inputs["audio"])   # [B, T, 64]
        n_a = torch.randn_like(lat_a)
        audio_t = self.scheduler.add_noise(lat_a, n_a, t)
        audio_target = self.scheduler.training_target(lat_a, n_a, t)

        # ACE-Step encoder 输出(text + lyric 占位)
        ace_enc_h, ace_enc_m = build_ace_encoder_hidden(
            self.model, text_caption=inputs["prompt"], lyrics="[Instrumental]",
            device=device,
        )
        # 静音 src 的 context_latents
        ace_ctx = build_context_latents_default(
            self.model.acestep_handler, latent_length=lat_a.shape[1],
            device=device, dtype=self.target_dtype,
        )

        # === 4. fusion forward(单次同时算两塔) ===
        ph, pw = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_v_len = vid_t.shape[2] * vid_t.shape[3] * vid_t.shape[4] // (ph * pw)
        video_pred, audio_pred, _ = self.model(
            vid=[vid_t.squeeze(0)],
            audio_latent=audio_t,
            t=t,
            vid_context=text_emb,
            ace_encoder_hidden_states=ace_enc_h,
            ace_encoder_attention_mask=ace_enc_m,
            ace_context_latents=ace_ctx,
            vid_seq_len=max_v_len,
            prev_audio_hidden=None,    # 训练时 step 之间不传(单 step monte carlo)
        )

        w = self.scheduler.training_weight(t)
        loss_v = F.mse_loss(video_pred.float(), vid_target[0].float()) * w
        loss_a = F.mse_loss(audio_pred.float(), audio_target.float()) * w
        loss = 0.7 * loss_v + 0.3 * loss_a
        return loss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, lora_base_model=None, lora_target_modules="q,k,v,o", lora_rank=32,
                 use_gradient_checkpointing=True, condition_dropout=0.0, **kwargs):
        super().__init__()
        self.pipe = OviAceStepModel(device="cuda")
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models=None,
            lora_base_model=lora_base_model,
            lora_target_modules=lora_target_modules,
            lora_rank=lora_rank, lora_checkpoint=kwargs.get("lora_checkpoint"),
            enable_fp8_training=False,
        )
        self.pipe.model.set_gradient_checkpointing(use_gradient_checkpointing)
        self.condition_dropout = condition_dropout

    def forward(self, data, inputs=None):
        if self.condition_dropout > 0 and random.random() < self.condition_dropout:
            data["prompt"] = ""
        return self.pipe(**data)


if __name__ == "__main__":
    set_seed(777)
    args = wan_parser().parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = AudioVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        csv_path=args.dataset_csv_path,
        dynamic_duration=args.dynamic_duration,
        repeat=args.dataset_repeat,
    )

    model = WanTrainingModule(
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=True,
        condition_dropout=args.condition_dropout,
    )
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    opt = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ConstantLR(opt)

    def _collate(b):
        b = [x for x in b if x is not None]
        return b[0] if b else None

    dl = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=_collate,
                                      num_workers=args.dataset_num_workers)
    model, opt, dl, sch = accelerator.prepare(model, opt, dl, sch)
    opt.zero_grad()

    g_loss = 0.0
    set_seed(777 + accelerator.process_index)
    t0 = time.time()
    for ep in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"Start epoch {ep}")
        for data in dl:
            if data is None: continue
            with accelerator.accumulate(model):
                loss = model(data)
                g_loss += loss.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    g_loss = accelerator.reduce(g_loss, "mean") / accelerator.gradient_accumulation_steps
                    opt.step(); opt.zero_grad(); sch.step()
                    logger.on_step_end(accelerator, model, args.save_steps)
                    if accelerator.is_main_process:
                        print(f"Ep {ep} Step {logger.num_steps}: Loss {g_loss.item():.4f} "
                              f"Norm {gnorm:.3f} Time {time.time()-t0:.1f}s")
                    g_loss = 0.0; t0 = time.time()
        if args.save_steps is None:
            logger.on_epoch_end(accelerator, model, ep)
    logger.on_training_end(accelerator, model, args.save_steps)
