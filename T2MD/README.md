# 环境
```bash
cuda 12.5
conda create -n ovi python=3.12.2
pip install -r requirements.txt
```
# 下载权重

从./Ovi/ckpts网址下载

```bash
|-ckpt
    |-MMAudio (Ovi原版音频VAE)
    |-Wan2.2-TI2V-5B (Ovi原版视频VAE、文本编码器umT5)
    |-Ovi (T2AV主干权重，Ovi原始权重)
    |-T2MV (T2MV主干权重，基于Ovi微调，音乐生成能力弱)
    |-T2A (T2AS权重，后面更新成T2ASM，24卡10天)
    |-VAE-ASM (Sound+Speech+Music VAE - wave 24khz, latent 50hz)
```
T2A数据构成：./AudioSummary/DiffSynth-Studio-Output/train_t2a_input/train_t2a_input.list
共2亿，其中：
1）影视剧T2A 80M（仅中文）
2）Audio 40M 
3）Music 40M 
4）TTS 40M（影视剧TTS 10M + 基础TTS 20M + 专项TTS 1M + 方言 10M）

训练时长：256卡*3天

# 数据
从./code/4s_pipeline/crawler_code/a_v_caption_filtered.jsonl网址下载，同时还要下载对应的音视频

# t2av训练
原料: ./code/4s_pipeline/crawler_code/a_v_caption_filtered.jsonl
数据格式: audio_path, video_path, audio_caption, video_caption
修改dataloader: ./AudioSummary/DiffSynth-Studio/diffsynth/trainers/t2mv_dataset.py
```bash
bash ./code/4s_pipeline/crawler_code/DiffSynth-Studio/examples/Ovi/run_multinode.sh
```
注意默认加载的是新的vae，如果基于ovi微调，请使用原来的mmaudio的vae，也就是说注视掉audio_vae_ckpt

#  t2av推理
原料：./AudioSummary/DiffSynth-Studio/inference/av_input.csv
```bash
torchrun --nnodes 1 --nproc_per_node 8 inference/t2av_infer.py --config-file ovi/configs/inference/inference_fusion.yaml
```

# t2a训练
```bash
./AudioSummary/DiffSynth-Studio/examples/Ovi/train_t2a.py (有待改)
```
训好的t2as模型：/apdcephfs_nj4/share_301739632/nickkhuang/exp/WAN5B_TTS_T2A_wd_bsz120shf_parafile/step-15000.safetensors


# t2a推理
修改./AudioSummary/DiffSynth-Studio/ovi/configs/inference/inference_audio.yaml其中ovi_ckpt放t2a的权重
```bash
torchrun --nnodes 1 --nproc_per_node 8 inference/t2a_infer.py --config-file ovi/configs/inference/inference_audio.yaml
```

# 数据
## 结构
分辨率和采样率会在加载时自动处理
```
dataset_base_path/                      
├── evan_metadata_s2v_with_prompt.csv   # 元数据 CSV
├── videos/                             # .mp4
│   ├── xxx.mp4
│   └── ...
└── audios/                             # .wav
    ├── xxx.wav
    └── ...
```
## Csv格式
```
video,input_audio,prompt
videos/clip_001.mp4,audios/clip_001.wav,a person is speaking
videos/clip_002.mp4,audios/clip_002.wav,a person is speaking
```

# s2v-5b训练
```
bash ./AudioSummary/DiffSynth-Studio/examples/wanvideo/model_training/taiji/Evan-Wan2.2-S2V-5B-multi-node.sh
```


# s2v-5b推理
```
bash ./AudioSummary/DiffSynth-Studio/inference/s2v_infer.py
```

# s2v-5b轨迹蒸馏训练
```
bash ./AudioSummary/DiffSynth-Studio/examples/wanvideo/model_training/taiji/Evan-Wan2.2-S2V-TI-5B-multi-node.sh
```

# s2v-5b轨迹蒸馏推理
关闭cfg，改步数为2
```
bash ./AudioSummary/DiffSynth-Studio/inference/s2v_infer.py
```


# 快速理解代码
./AudioSummary/DiffSynth-Studio/ovi/utils/model_loading_utils.py
init_wan_vae_2_2() - 初始化视频 VAE
init_mmaudio_vae() - 初始化音频 VAE
init_text_model() - 初始化 T5 文本编码器
init_fusion_score_model_ovi() - 初始化融合模型
load_fusion_checkpoint() - 加载融合模型检查点

./AudioSummary/DiffSynth-Studio/ovi/ovi_fusion_engine.py
OviFusionEngine generate

./AudioSummary/DiffSynth-Studio/ovi/modules/fusion.py
self.model() forward核心代码
