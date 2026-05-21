"""
FusionAceStepModel: Ovi video 塔 + ACE-Step audio 塔的双向 cross-attn fusion。

与原 ovi/modules/fusion.py 的区别:
  - 原 audio_model 是 WanModel(MMAudio 那套),这里换成 AceStepDiTModel
  - 因为两塔维度/层数不同,加 CrossAttnAdapter 做投影
  - 每个 video block 找映射到的 audio block,各自把对方的 hidden state 作为 KV 做 cross-attn
  - 训练时:Ovi base + ACE-Step base 都 frozen(或 LoRA),只训新增的 cross-attn adapter

forward 输入输出与原 fusion 一致,这样外部 (train_t2av.py, infer) 几乎不用改调用方式。
"""
import torch
import torch.nn as nn
from typing import Optional

from ovi.modules.model import WanModel
from ovi.utils.acestep_loader import (
    load_acestep_dit, CrossAttnAdapter, build_layer_mapping,
)


class FusionAceStepModel(nn.Module):
    def __init__(self, video_config, audio_config, acestep_project_root: str):
        """
        video_config: dict, Wan video DiT 超参(沿用 ovi/configs/model/dit/video.json)
        audio_config: dict, ACE-Step DiT 超参(ovi/configs/model/dit/audio_acestep.json)
        acestep_project_root: ACE-Step-1.5-main 项目根目录(权重在 checkpoints/ 下)
        """
        super().__init__()

        # === 视频塔:Wan video DiT(原版) ===
        self.video_model = WanModel(**video_config)
        self.video_dim = video_config["dim"]
        self.num_video_layers = video_config["num_layers"]

        # === 音频塔:ACE-Step DiT(替换原 Wan audio 塔) ===
        self.audio_model, self.ace_config = load_acestep_dit(
            acestep_project_root,
            config_path=audio_config.get("ace_model_dir", "acestep-v15-turbo"),
            dtype=torch.bfloat16,
        )
        self.audio_dim = self.ace_config.hidden_size      # 2048
        self.num_audio_layers = self.ace_config.num_hidden_layers  # 24

        # === Cross-attn 维度对齐 adapter(新增,trainable) ===
        # Video hidden (3072) -> ACE-Step encoder_hidden_states (2048)
        self.v2a_adapter = CrossAttnAdapter(self.video_dim, self.audio_dim)
        # ACE-Step hidden (2048) -> Video tower 的 fusion KV (3072)
        self.a2v_adapter = CrossAttnAdapter(self.audio_dim, self.video_dim)

        # === Video 塔 cross-attn KV 投影(沿用原 ovi 的 k_fusion/v_fusion 注入方式) ===
        # 注:对应原 ovi/modules/fusion.py:40 inject_cross_attention_kv_projections
        from ovi.modules.model import WanLayerNorm, WanRMSNorm
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(self.video_dim, self.video_dim)
            vid_block.cross_attn.v_fusion = nn.Linear(self.video_dim, self.video_dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(self.video_dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(self.video_dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            )

        # video block <-> audio block 索引映射(30 video 层 ↔ 24 audio 层)
        self.layer_mapping = build_layer_mapping(self.num_video_layers, self.num_audio_layers)

        # === 训练策略:base 全冻结,只训 cross-attn adapter + video 的 *_fusion 投影 ===
        self._freeze_base_towers()

        self.gradient_checkpointing = False

    def _freeze_base_towers(self):
        """冻结两个塔的 base 权重,只有 cross-attn 相关参数 trainable。"""
        # Audio 塔全冻
        for p in self.audio_model.parameters():
            p.requires_grad = False
        # Video 塔:除了 *_fusion 命名的参数,其他冻
        for name, p in self.video_model.named_parameters():
            p.requires_grad = ("fusion" in name)

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable
        self.video_model.set_gradient_checkpointing(enable)
        # ACE-Step 自带 gradient_checkpointing
        self.audio_model.gradient_checkpointing = enable

    def set_rope_params(self):
        self.video_model.set_rope_params()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        vid,
        audio,                      # [B, T_audio, in_channels=8],ACE-Step latent
        t,
        vid_context,                # [B, L_text, dim] T5 embedding
        audio_context,              # ACE-Step 的 lyric encoder 输出;若没歌词传 zeros
        vid_seq_len,
        audio_seq_len=None,         # ACE-Step 用 attention_mask,这里保留兼容
        ace_extra_inputs: Optional[dict] = None,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        """
        ace_extra_inputs: ACE-Step 必需的额外输入(timestep_r, context_latents, encoder_attention_mask 等)
        训练时由 dataloader 准备好。
        """
        assert clip_fea is None and y is None

        # ---- 1. video 塔的 input embedding(沿用 Wan 原逻辑) ----
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len,
            clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean,
        )

        # ---- 2. audio 塔的 input embedding(ACE-Step DiTModel 内部做 patch + time_embed) ----
        # 这里我们调用 ACE-Step 内部的 input projection,不走它的 forward 顶层
        # (因为顶层 forward 直接跑完所有 layer,我们要在 layer 间穿插 cross-attn)
        ace = self.audio_model
        audio_x = ace.proj_in(audio)                                # [B, T/patch, hidden]
        temb_t, timestep_proj_t = ace.time_embed(t)
        # 没有 timestep_r 的话用 t 自己,timestep_r 是 ACE-Step 的双时间步条件,
        # 推理时由用户传或者默认置 0
        timestep_r = ace_extra_inputs.get("timestep_r", torch.zeros_like(t)) if ace_extra_inputs else torch.zeros_like(t)
        temb_r, timestep_proj_r = ace.time_embed_r(t - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # context_latents (ACE-Step 的 source/chunk_mask 条件,推理时由 dataloader 准备)
        if ace_extra_inputs is not None and "context_latents" in ace_extra_inputs:
            audio_x = torch.cat([ace_extra_inputs["context_latents"], audio_x], dim=-1)
        # 位置编码
        position_ids = torch.arange(audio_x.shape[1], device=audio_x.device).unsqueeze(0)
        audio_pos_embeddings = ace.rotary_emb(audio_x, position_ids)

        # ---- 3. 逐 video block 推进,中间插入双向 cross-attn ----
        kwargs = self._merge_kwargs(vid_kwargs)

        for vi in range(self.num_video_layers):
            if slg_layer and vi == slg_layer:
                continue
            vid_block = self.video_model.blocks[vi]
            ai = self.layer_mapping[vi]
            audio_layer = self.audio_model.layers[ai]

            # 3.1 vid self-attn + cross-attn(以 audio 当前态为 KV)
            vid = self._video_block_with_audio_xattn(
                vid_block, vid, audio_x, vid_e, kwargs
            )

            # 3.2 ACE-Step layer(以 video 当前态当 encoder_hidden_states)
            # 注:ACE-Step DiTLayer 已自带 cross_attn 接口,直接复用
            encoder_hidden = self.v2a_adapter(vid)            # [B, L_vid, audio_dim]
            audio_x = audio_layer(
                audio_x,
                position_embeddings=audio_pos_embeddings,
                temb=timestep_proj,
                attention_mask=None,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=None,
            )[0]

        # ---- 4. 视频塔 output ----
        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs["grid_sizes"], vid_e)

        # ---- 5. 音频塔 output(ACE-Step 的 norm_out + proj_out) ----
        shift, scale = (ace.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        audio_x = (ace.norm_out(audio_x) * (1 + scale) + shift).type_as(audio_x)
        audio_out = ace.proj_out(audio_x)

        return vid, audio_out

    def _video_block_with_audio_xattn(self, vid_block, vid, audio_x, vid_e, kwargs):
        """单层 video block 内部:self-attn -> cross-attn(text + audio) -> ffn。
           复用原 ovi single_fusion_block_forward 的设计(见 fusion.py:161),
           但 target_seq 由 a2v_adapter 投影后的 audio_x 提供。
        """
        # 计算 modulation
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e = vid_block.modulation(vid_e).chunk(6, dim=2)

        # self-attn
        vid_y = vid_block.self_attn(
            vid_block.norm1(vid).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            kwargs["seq_lens"], kwargs["grid_sizes"], kwargs["freqs"],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vid = vid + vid_y * e[2].squeeze(2)

        # cross-attn:text + audio(以 audio_x 经 a2v 投影后做 KV)
        audio_kv = self.a2v_adapter(audio_x)
        vid = self._video_cross_attn(
            vid_block, vid, audio_kv,
            kwargs["context"], kwargs["context_lens"],
            kwargs["grid_sizes"], kwargs["freqs"], e,
        )
        return vid

    def _video_cross_attn(self, vid_block, vid, audio_kv, context, context_lens,
                          grid_sizes, freqs, vid_e):
        """video 侧的 cross-attn,KV 来自 audio_kv(已投影到 video_dim)。
           简化版,不做 sequence parallel。
        """
        from ovi.modules.attention import flash_attention
        cross = vid_block.cross_attn
        b, n, d = vid.size(0), cross.num_heads, cross.head_dim

        # text branch
        if hasattr(cross, "k_img"):
            q, k, v, k_img, v_img = cross.qkv_fn(vid_block.norm3(vid), context)
        else:
            q, k, v = cross.qkv_fn(vid_block.norm3(vid), context)
            k_img = v_img = None
        x = flash_attention(q, k, v, k_lens=context_lens)
        if k_img is not None:
            x = x + flash_attention(q, k_img, v_img, k_lens=None)

        # audio branch(走 *_fusion 投影)
        a_in = cross.pre_attn_norm_fusion(audio_kv)
        k_a = cross.norm_k_fusion(cross.k_fusion(a_in)).view(b, -1, n, d)
        v_a = cross.v_fusion(a_in).view(b, -1, n, d)
        x = x + flash_attention(q, k_a, v_a, k_lens=None)

        x = x.flatten(2)
        x = cross.o(x)

        # ffn
        y = vid_block.ffn(
            vid_block.norm2(vid + x).bfloat16() * (1 + vid_e[4].squeeze(2)) + vid_e[3].squeeze(2)
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return vid + x + y * vid_e[5].squeeze(2)

    def _merge_kwargs(self, vid_kwargs):
        """把 vid_kwargs 字典展开成 forward 内部更直观的名字。"""
        return {
            "seq_lens": vid_kwargs["seq_lens"],
            "grid_sizes": vid_kwargs["grid_sizes"],
            "freqs": vid_kwargs["freqs"],
            "context": vid_kwargs["context"],
            "context_lens": vid_kwargs["context_lens"],
        }

    def init_weights(self):
        self.video_model.init_weights()
        # video 侧 *_fusion 权重缩小(沿用原 ovi 做法)
        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)
