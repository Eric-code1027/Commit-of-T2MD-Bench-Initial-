"""
FusionAceStepModel: Ovi video DiT (Wan) + ACE-Step audio DiT 双塔 fusion。
"""
import torch
import torch.nn as nn
from typing import Optional

from ovi.modules.model import WanModel, WanLayerNorm, WanRMSNorm
from ovi.modules.attention import flash_attention
from ovi.utils.acestep_loader import (
    init_acestep_handler, CrossAttnAdapter, build_context_latents_default,
)


class FusionAceStepModel(nn.Module):
    def __init__(self, video_config, audio_config, acestep_project_root: str,
                 device: str = "cuda"):
        super().__init__()

        # === 视频塔(Wan video DiT,原版) ===
        self.video_model = WanModel(**video_config)
        self.video_dim = video_config["dim"]
        self.num_video_layers = video_config["num_layers"]

        # === 音频塔(ACE-Step,完整 handler) ===
        self._ace_handler = init_acestep_handler(
            acestep_project_root,
            config_path=audio_config.get("ace_model_dir", "acestep-v15-turbo"),
            device=device,
            dtype=torch.bfloat16,
        )
        self.audio_dim = self._ace_handler.config.hidden_size           # 2048
        self.num_audio_layers = self._ace_handler.config.num_hidden_layers  # 24

        # === 双向 cross-attn adapter(trainable) ===
        # Video hidden (3072) -> ACE-Step encoder_hidden_states (2048)
        self.v2a_adapter = CrossAttnAdapter(self.video_dim, self.audio_dim)
        # ACE-Step hidden (2048) -> Video fusion KV (3072)
        self.a2v_adapter = CrossAttnAdapter(self.audio_dim, self.video_dim)

        # === Video 塔 cross-attn 注入 *_fusion(沿用 Ovi 原 inject 方式) ===
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(self.video_dim, self.video_dim)
            vid_block.cross_attn.v_fusion = nn.Linear(self.video_dim, self.video_dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(self.video_dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(self.video_dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            )

        self._freeze_base()
        self.gradient_checkpointing = False

    @property
    def acestep_dit(self):
        return self._ace_handler.model.decoder

    @property
    def acestep_encoder(self):
        return self._ace_handler.model.encoder

    @property
    def acestep_vae(self):
        return self._ace_handler.vae

    @property
    def acestep_handler(self):
        return self._ace_handler

    def _freeze_base(self):
        """两塔 base 全冻结,只训 *_fusion + adapter。"""
        for p in self.acestep_dit.parameters():
            p.requires_grad = False
        for p in self.acestep_encoder.parameters():
            p.requires_grad = False
        for p in self.acestep_vae.parameters():
            p.requires_grad = False
        for name, p in self.video_model.named_parameters():
            p.requires_grad = ("fusion" in name)

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable
        self.video_model.set_gradient_checkpointing(enable)
        self.acestep_dit.gradient_checkpointing = enable

    def set_rope_params(self):
        self.video_model.set_rope_params()

    # ------------------------------------------------------------------
    # forward:塔级 cross-attn
    # ------------------------------------------------------------------
    def forward(
        self,
        vid,                            # list[ [C,F,H,W] ]  (Ovi 视频塔输入,latent)
        audio_latent,                   # [B, T, 64]   ACE-Step VAE encode 出的 audio latent
        t,                              # [B]   timestep
        vid_context,                    # T5 text embedding(list, 1 个 element)
        ace_encoder_hidden_states,      # [B, L_enc, 2048]  ACE-Step encoder 输出(text+lyric+refer)
        ace_encoder_attention_mask,     # [B, L_enc]
        ace_context_latents,            # [B, T, 128]   ACE-Step src+chunk_mask
        ace_attention_mask=None,        # [B, T]   audio 侧 attn mask
        vid_seq_len=None,
        prev_audio_hidden: Optional[torch.Tensor] = None,
        clip_fea=None, y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        """
        prev_audio_hidden: 上一个 denoise step 的 audio DiT 最后 hidden state,
            用来作为 video 塔的 fusion KV。第一个 step 传 None 表示不做 cross-attn。

        返回:
            video_pred: [C, F, H, W] 视频塔的 flow 预测
            audio_pred: [B, T, 64]   音频塔的 flow 预测
            audio_hidden_for_next:    本 step 的 audio hidden(下个 step 用)
        """
        # ---------------- 1. ACE-Step DiT forward(完整跑 24 层) ----------------
        # 注:这里 video 信息通过 ace_encoder_hidden_states 上 concat 进来,详见 build_ace_encoder_hidden
        # decoder 接口签名见 acestep/training/trainer.py:517 实证
        if ace_attention_mask is None:
            ace_attention_mask = torch.ones(
                audio_latent.shape[0], audio_latent.shape[1],
                device=audio_latent.device, dtype=audio_latent.dtype,
            )
        ace_out = self.acestep_dit(
            hidden_states=audio_latent,
            timestep=t,
            timestep_r=t,
            attention_mask=ace_attention_mask,
            encoder_hidden_states=ace_encoder_hidden_states,
            encoder_attention_mask=ace_encoder_attention_mask,
            context_latents=ace_context_latents,
            use_cache=False,
            past_key_values=None,
            return_hidden_states=False,
        )

        audio_pred = ace_out[0] if isinstance(ace_out, tuple) else ace_out.last_hidden_state

        vid_x, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len,
            clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean,
        )

        # 用 prev step 的 audio hidden(如果 None 就用本 step 的 audio_hidden)
        audio_for_kv = prev_audio_hidden if prev_audio_hidden is not None else audio_hidden
        audio_kv = self.a2v_adapter(audio_for_kv)   # [B, T_a, 3072]

        for vi in range(self.num_video_layers):
            if slg_layer and vi == slg_layer:
                continue
            vid_block = self.video_model.blocks[vi]
            vid_x = self._video_block_with_audio_xattn(
                vid_block, vid_x, audio_kv, vid_e, vid_kwargs,
            )

        video_pred = self.video_model.post_transformer_block_out(
            vid_x, vid_kwargs["grid_sizes"], vid_e,
        )

        return video_pred, audio_pred, audio_hidden

    def _video_block_with_audio_xattn(self, vid_block, vid, audio_kv, vid_e, kwargs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e = vid_block.modulation(vid_e).chunk(6, dim=2)

        # self-attn
        y = vid_block.self_attn(
            vid_block.norm1(vid).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            kwargs["seq_lens"], kwargs["grid_sizes"], kwargs["freqs"],
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vid = vid + y * e[2].squeeze(2)

        # cross-attn(text + audio)
        cross = vid_block.cross_attn
        b, n, d = vid.size(0), cross.num_heads, cross.head_dim
        normed = vid_block.norm3(vid)

        # text branch(原 Ovi 的 qkv_fn)
        if hasattr(cross, "k_img"):
            q, k_text, v_text, k_img, v_img = cross.qkv_fn(normed, kwargs["context"])
        else:
            q, k_text, v_text = cross.qkv_fn(normed, kwargs["context"])
            k_img = v_img = None
        x = flash_attention(q, k_text, v_text, k_lens=kwargs["context_lens"])
        if k_img is not None:
            x = x + flash_attention(q, k_img, v_img, k_lens=None)

        # audio branch(走 *_fusion 投影)
        a_in = cross.pre_attn_norm_fusion(audio_kv)
        k_a = cross.norm_k_fusion(cross.k_fusion(a_in)).view(b, -1, n, d)
        v_a = cross.v_fusion(a_in).view(b, -1, n, d)
        x = x + flash_attention(q, k_a, v_a, k_lens=None)

        x = x.flatten(2)
        x = cross.o(x)
        vid = vid + x

        # ffn
        ff = vid_block.ffn(
            vid_block.norm2(vid).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return vid + ff * e[5].squeeze(2)

    def init_weights(self):
        self.video_model.init_weights()
        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)


@torch.no_grad()
def build_ace_encoder_hidden(fusion_model: FusionAceStepModel,
                              text_caption: str,
                              lyrics: str = "[Instrumental]",
                              device: str = "cuda"):
    handler = fusion_model.acestep_handler
    tok = handler.text_tokenizer
    text_enc = handler.text_encoder

    # ---- text ----
    inputs = tok(text_caption, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = text_enc(**inputs, output_hidden_states=False)
    text_hidden = out.last_hidden_state.to(handler.dtype)
    text_mask = inputs["attention_mask"].to(handler.dtype)

    # ---- lyric:简化,用 zeros 占位(实际 ACE-Step 有完整 lyric tokenization 管线) ----
    # 真实部署时应该用 handler 的 lyric encoder。这里 [Instrumental] 时全 0 是合法的。
    lyric_hidden = torch.zeros(1, 1, text_hidden.shape[-1],
                                device=device, dtype=handler.dtype)
    lyric_mask = torch.zeros(1, 1, device=device, dtype=handler.dtype)

    # ---- refer audio: zeros ----
    refer_hidden = torch.zeros(1, 1, 64, device=device, dtype=handler.dtype)
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

    encoder_hidden, encoder_mask = handler.model.encoder(
        text_hidden_states=text_hidden,
        text_attention_mask=text_mask,
        lyric_hidden_states=lyric_hidden,
        lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer_hidden,
        refer_audio_order_mask=refer_order_mask,
    )
    return encoder_hidden, encoder_mask
