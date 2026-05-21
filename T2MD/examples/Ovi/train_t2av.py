"""
T2MD-Bench 训练脚本:

替换原 train_t2av.py:
  1. 删除 MMAudio VAE 加载逻辑;音频侧统一用 ACE-Step
  2. fusion 模型从 init_fusion_score_model_ovi 换成 init_fusion_acestep_model
  3. forward 时 audio latent 由 ACE-Step 的 audio tokenizer (DAC) 编码,不再用 MMAudio VAE
  4. 训练参数:base 塔全 frozen,只训新增的 cross-attn adapter + video 侧 *_fusion
"""
import os, json, random, time
import torch
import torch.nn as nn
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

# === 用新 fusion(替换原 Wan video + MMAudio audio) ===
from ovi.utils.model_loading_utils import (
    init_text_model, init_wan_vae_2_2, load_fusion_checkpoint,
)
from ovi.utils.model_loading_utils_patch import init_fusion_acestep_model

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
        self.cpu_offload = config.get("cpu_offload", False)
        acestep_root = config["acestep_project_root"]

        # 1. fusion 主模型(Wan video + ACE-Step DiT)
        model, video_config, audio_config = init_fusion_acestep_model(
            acestep_project_root=acestep_root, rank="cpu",
        )
        self.model = model
        self.video_config = video_config
        self.audio_config = audio_config

        # 2. 视频 VAE(沿用)
        self.vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        self.vae_model_video.model.requires_grad_(False).eval()
        self.vae_model_video.model = self.vae_model_video.model.bfloat16()

        # 3. 音频 tokenizer:从 ACE-Step 的 AceStepAudioTokenizer 加载(替换 MMAudio VAE)
        self.audio_tokenizer = self._load_acestep_audio_tokenizer(acestep_root, audio_config, device)

        # 4. T5(沿用)
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)

        # 5. 加载 Ovi video 塔预训权重(只加载 video_model 那部分,audio 塔已由 fusion 内部加载)
        ovi_ckpt = os.path.join(config.ckpt_dir, "Ovi",
                                "model_fp8_e4m3fn.safetensors" if config.get("fp8") else "model.safetensors")
        if os.path.exists(ovi_ckpt):
            load_fusion_checkpoint(self.model, checkpoint_path=ovi_ckpt, from_meta=False, strict=False)

        if not config.get("fp8"):
            self.model = self.model.to(dtype=target_dtype)
        self.model = self.model.to(device if not self.cpu_offload else "cpu")
        self.model.set_rope_params()

        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

    def _load_acestep_audio_tokenizer(self, acestep_root, audio_config, device):
        """加载 ACE-Step 的 AudioTokenizer(DAC),把 wav -> latent(ACE DiT 的输入空间)"""
        import sys
        if acestep_root not in sys.path:
            sys.path.insert(0, acestep_root)
        from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepAudioTokenizer
        from acestep.models.common.configuration_acestep_v15 import AceStepConfig
        ckpt_dir = os.path.join(acestep_root, "checkpoints", audio_config["ace_model_dir"])
        cfg = AceStepConfig.from_pretrained(ckpt_dir)
        tok = AceStepAudioTokenizer.from_pretrained(ckpt_dir, config=cfg)
        for p in tok.parameters():
            p.requires_grad = False
        return tok.to(device).eval()

    def forward(self, **inputs):
        # === timestep ===
        t_max = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        t_min = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        t_id = torch.randint(t_min, t_max, (1,))
        t = self.scheduler.timesteps[t_id].to(self.target_dtype).to(self.device)

        # === text ===
        text_emb = self.text_model(inputs["prompt"], self.text_model.device)
        text_emb = [e.to(self.target_dtype).to(self.device) for e in text_emb]

        # === video latent ===
        lat_v = self.vae_model_video.wrapped_encode(inputs["video"]).to(self.target_dtype)
        n_v = torch.randn_like(lat_v)
        vid_t = self.scheduler.add_noise(lat_v, n_v, t)
        vid_target = self.scheduler.training_target(lat_v, n_v, t)

        # === audio latent(用 ACE-Step tokenizer 把 wav 编码到 ACE DiT 输入空间) ===
        with torch.no_grad():
            lat_a = self.audio_tokenizer.encode(inputs["audio"]).to(self.target_dtype)
        n_a = torch.randn_like(lat_a)
        audio_t = self.scheduler.add_noise(lat_a, n_a, t)
        audio_target = self.scheduler.training_target(lat_a, n_a, t)

        # === fusion forward ===
        ph, pw = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_v = vid_t.shape[2] * vid_t.shape[3] * vid_t.shape[4] // (ph * pw)

        vid_pred, audio_pred = self.model(
            vid=vid_t, audio=audio_t, t=t,
            vid_context=text_emb, audio_context=text_emb,   # 简化:歌词 encoder 临时复用 T5
            vid_seq_len=max_seq_len_v,
            audio_seq_len=audio_t.shape[1],
        )

        w = self.scheduler.training_weight(t)
        loss_v = torch.nn.functional.mse_loss(vid_pred[0].float(), vid_target[0].float()) * w
        loss_a = torch.nn.functional.mse_loss(audio_pred[0].float(), audio_target[0].float()) * w
        loss = 0.7 * loss_v + 0.3 * loss_a
        if accelerator.is_main_process:
            print(f"[INFO] loss_v {loss_v.item():.4f} loss_a {loss_a.item():.4f}")
        return loss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, lora_base_model=None, lora_target_modules="q,k,v,o", lora_rank=32,
                 use_gradient_checkpointing=True, condition_dropout=0.0, **kwargs):
        super().__init__()
        self.pipe = OviAceStepModel(device="cuda")
        # 只对 cross-attn adapter + video *_fusion 加 LoRA(base 已 frozen)
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
    opt = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ConstantLR(opt)

    def _collate(b):
        b = [x for x in b if x is not None]
        return b[0] if b else None

    dl = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=_collate, num_workers=args.dataset_num_workers)
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
