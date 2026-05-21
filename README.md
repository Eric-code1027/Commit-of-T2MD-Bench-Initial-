# ACE-Step Fusion 改造说明

改动:把 Ovi 原有的 "Wan video DiT ↔ Wan(MMAudio) audio DiT" 双塔
替换为 **"Wan video DiT ↔ ACE-Step DiT"** 双塔,通过双向 cross-attention
在每一层做信息交互。训练和推理使用同一个 fusion 模型,音频不再独立生成。

## 新增/修改的文件

### 新增

| 文件 | 作用 |
|---|---|
| `ovi/configs/model/dit/audio_acestep.json` | ACE-Step DiT 的超参摘要(层数 24、hidden 2048 等),供 fusion 构 cross-attn adapter 用 |
| `ovi/utils/acestep_loader.py` | 加载 ACE-Step DiT 主干权重,定义 `CrossAttnAdapter`(两塔维度对齐用)和 `build_layer_mapping`(30 视频层 ↔ 24 音频层的索引映射) |
| `ovi/modules/fusion_acestep.py` | **核心**:新的 `FusionAceStepModel`。video 塔走 Wan,audio 塔走 ACE-Step,每层做双向 cross-attn。base 塔全 frozen,只训新增 cross-attn 参数 |
| `ovi/utils/model_loading_utils_patch.py` | 新增 `init_fusion_acestep_model()`,替代原 `init_fusion_score_model_ovi()` |

### 修改

| 文件 | 改动点 |
|---|---|
| `examples/Ovi/train_t2av.py` | 删除 MMAudio VAE 加载;音频侧改用 ACE-Step 的 `AceStepAudioTokenizer` 把 wav → latent;fusion 模型改用 `FusionAceStepModel`;loss 仍是 video + audio MSE 联合训练 |
| `inference/t2av_v2_infer.py` | 删掉"先视频后音频"的分步推理逻辑;改成 video latent 和 audio latent 在同一个 denoise loop 里同步更新,CFG 也是两塔同时算 |

### 删除(原 Ovi 独立音频分支已废弃)

| 文件 / 目录 | 原作用 |
|---|---|
| `ovi/modules/mmaudio/` | 原 MMAudio VAE + DiT 实现 |
| `ovi/ovi_video_only_engine.py` | 旧的"视频独立推理"引擎 |
| `ovi/audio_branches/acestep_branch.py` | 旧的"ACE-Step 独立推理"wrapper |

## 训练/推理使用

训练:
```bash
bash examples/Ovi/run_overfit_dance10.sh
```
配置文件 `ovi/configs/training/finetune.yaml` 里需新增字段:
```yaml
acestep_project_root: /root/ACE-Step-1.5-main
```

推理:
```bash
python inference/t2av_v2_infer.py \
    --config-file ovi/configs/inference/inference_v2.yaml
```
inference yaml 同样需要 `acestep_project_root` 字段。
