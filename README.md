# ACE-Step Fusion 改造说明

## 改造目标

把 Ovi 原 "Wan video DiT ↔ Wan(MMAudio) audio DiT" 双塔架构,替换为
**"Wan video DiT ↔ ACE-Step DiT"** 双塔,在塔间做 cross-attention 信息交互。
训练与推理共享同一个 fusion 模型。

## 架构说明

###  ACE-Step fusion(塔级 cross-attn)
由于 ACE-Step 的 `AceStepDiTModel` 是 monolithic 调用(24 层封装在
`forward` 内部,从外部无法在层间插入交互),无法做到逐层 cross-attn。
因此采用 **塔级 cross-attn**:

- 每个 denoise step:
  1. **ACE-Step DiT 一次跑完 24 层**,得到 audio flow 预测 + audio hidden
  2. audio hidden 经 `a2v_adapter`(2048→3072 LayerNorm + Linear)投影,
     作为 video DiT **每一层** cross-attn 的 fusion KV(走 Ovi 原有
     `*_fusion` 注入机制)
  3. Video DiT 跑完 30 层,得到 video flow 预测 + video hidden
  4. video hidden 经 `v2a_adapter`(3072→2048)投影后,通过 `build_ace_encoder_hidden`
     与 text/lyric encoder 输出拼接,作为下一个 step ACE-Step 的
     `encoder_hidden_states`

### 维度/层数对齐

| 项 | Video 塔 (Wan) | Audio 塔 (ACE-Step turbo) |
|---|---|---|
| hidden_size | 3072 | 2048 |
| num_layers | 30 | 24 |
| attention impl | flash_attention | sdpa / eager |
| 训练时是否 frozen | base frozen,只训 *_fusion | 全 frozen |

两塔维度差异通过 `CrossAttnAdapter` 解决;层数不再做对齐(塔级 cross-attn,
不需要层映射)。

## 新增/修改的文件

### 新增

| 文件 | 作用 |
|---|---|
| `ovi/configs/model/dit/audio_acestep.json` | ACE-Step 关键超参摘要(hidden_size, num_hidden_layers, VAE latent dim 等) |
| `ovi/utils/acestep_loader.py` | 通过官方 `AceStepHandler.initialize_service()` 加载完整音频栈(DiT/VAE/Encoder/text_encoder/lyric);提供 `CrossAttnAdapter`、`encode_audio_to_latent`、`decode_latent_to_audio`、`build_context_latents_default` 工具 |
| `ovi/modules/fusion_acestep.py` | **核心**:`FusionAceStepModel`(双塔 + adapter + *_fusion 注入);`build_ace_encoder_hidden`(组装 ACE-Step DiT 需要的 encoder_hidden_states) |
| `ovi/utils/model_loading_utils_patch.py` | 入口函数 `init_fusion_acestep_model()`,替代原 `init_fusion_score_model_ovi()` |

### 修改

| 文件 | 改动点 |
|---|---|
| `examples/Ovi/train_t2av.py` | 删 MMAudio VAE 加载;audio wav→latent 改用 ACE-Step VAE(`encode_audio_to_latent`);`encoder_hidden_states` 用 ACE-Step encoder 算;fusion forward 一次返回 (video_pred, audio_pred);loss = 0.7 × MSE(video) + 0.3 × MSE(audio) |
| `inference/t2av_v2_infer.py` | 删独立 ACE-Step wrapper 调用;denoise loop 改成两塔同步推进;CFG 两塔都做;decode 时 video 走 Wan VAE,audio 走 ACE-Step VAE(`decode_latent_to_audio`) |

### 配置文件需新增字段

`ovi/configs/training/finetune.yaml` 和 `ovi/configs/inference/inference_v2.yaml` 均需新增:

```yaml
acestep_project_root: /root/ACE-Step-1.5-main
```

(指向 ACE-Step-1.5-main 项目根目录,handler 通过这个找权重)

### 废弃(原独立音频分支已废弃)

| 文件 / 目录 | 原作用 |
|---|---|
| `ovi/modules/mmaudio/` | 原 MMAudio VAE + DiT 实现 |
| `ovi/ovi_video_only_engine.py` | 旧视频独立推理引擎 |
| `ovi/audio_branches/acestep_branch.py` | 旧"ACE-Step 独立推理"wrapper |

## 训练/推理可训练参数

`FusionAceStepModel._freeze_base()` 内:

- **冻结**:`acestep_dit`、`acestep_encoder`、`acestep_vae`、`video_model` 除
  `*_fusion` 之外的全部参数
- **trainable**:
  - `video_model.blocks[i].cross_attn.{k_fusion, v_fusion, pre_attn_norm_fusion, norm_k_fusion}`(沿用原 Ovi 注入的 *_fusion 模块)
  - `v2a_adapter`、`a2v_adapter`(LayerNorm + Linear 投影)

LoRA 训练时,`--lora_target_modules` 仍可设为 `q,k,v,o,ffn.0,ffn.2`,
LoRA 只会注入到 video_model 那些已经 trainable 的层。

## 使用

```bash
# 训练
bash examples/Ovi/run_overfit_dance10.sh

# 推理(用训练好的 fusion ckpt)
python inference/t2av_v2_infer.py \
    --config-file ovi/configs/inference/inference_v2.yaml
```
