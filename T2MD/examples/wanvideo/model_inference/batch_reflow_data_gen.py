import os
import sys
import json
import pathlib
import pandas as pd
import torch
import librosa
from PIL import Image
from tqdm import tqdm

sys.path.append("./github_projects/from_modelscope/DiffSynth-Studio")
from diffsynth.utils.data import VideoData
from diffsynth.pipelines.wan_video_s2v import WanVideoS2VPipeline, ModelConfig

# ======================== Config ========================
DEBUG = True
MAX_SAMPLES = 100000

CSV_PATH = "./train_datasets/wan_s2v/distill_video_dataset/example_video_dataset_large/evan_metadata_s2v_with_prompt_overfit.csv"
BASE_PATH = "./train_datasets/wan_s2v/distill_video_dataset/example_video_dataset_large"
OUTPUT_ROOT = "./train_datasets/wan_s2v/distill_video_dataset"
JSONL_PATH = os.path.join(OUTPUT_ROOT, "reflow_data.jsonl")

SFT_CKPT = "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-5B_full_multi_node_time4/step-19000.safetensors"

NUM_FRAMES = 121
HEIGHT = 832
WIDTH = 448
NUM_INFERENCE_STEPS = 40
CFG_SCALE = 5.0
PROMPT = "a person is speaking"
NEGATIVE_PROMPT = ""

NOISE_DIR = os.path.join(OUTPUT_ROOT, "noise")
LATENT_DIR = os.path.join(OUTPUT_ROOT, "wan_latent")
FIRST_FRAME_DIR = os.path.join(OUTPUT_ROOT, "first_frame_latent")

# ======================== Init ========================
os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(LATENT_DIR, exist_ok=True)
os.makedirs(FIRST_FRAME_DIR, exist_ok=True)

pipe = WanVideoS2VPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": SFT_CKPT}),
        ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="./github_projects/Wan2.2/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/google/umt5-xxl/"),
    audio_processor_config=ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/"),
)

# ======================== Load CSV ========================
df = pd.read_csv(CSV_PATH)
if DEBUG:
    df = df.head(10)
    print(f"[DEBUG] 仅处理前 {len(df)} 条样本")
elif MAX_SAMPLES is not None:
    df = df.head(MAX_SAMPLES)
    print(f"限制处理前 {len(df)} 条样本")

# ======================== Batch Inference ========================
jsonl_records = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating reflow data"):
    row_dict = row.to_dict()

    video_rel = row_dict["video"]
    audio_rel = row_dict["input_audio"]
    prompt = row_dict.get("prompt", PROMPT)

    video_path = os.path.join(BASE_PATH, video_rel)
    audio_path = os.path.join(BASE_PATH, audio_rel)

    video_data = VideoData(video_path, height=HEIGHT, width=WIDTH)
    input_image = video_data[0]

    input_audio, sample_rate = librosa.load(audio_path, sr=16000)

    video_stem = pathlib.Path(video_rel).stem

    noise, latent, first_frame_latent = pipe.generate_reflow_pair(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=input_image,
        seed=idx,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        audio_sample_rate=sample_rate,
        input_audio=input_audio,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
    )

    noise_save = noise.to(dtype=torch.bfloat16).cpu()
    latent_save = latent.to(dtype=torch.bfloat16).cpu()

    noise_rel = f"noise/{video_stem}.pt"
    latent_rel = f"wan_latent/{video_stem}.pt"

    torch.save(noise_save, os.path.join(OUTPUT_ROOT, noise_rel))
    torch.save(latent_save, os.path.join(OUTPUT_ROOT, latent_rel))

    ff_latent_rel = None
    if first_frame_latent is not None:
        ff_latent_save = first_frame_latent.to(dtype=torch.bfloat16).cpu()
        ff_latent_rel = f"first_frame_latent/{video_stem}.pt"
        torch.save(ff_latent_save, os.path.join(OUTPUT_ROOT, ff_latent_rel))

    record = dict(row_dict)
    record["distill_noise"] = noise_rel
    record["distill_latent"] = latent_rel
    record["first_frame_latent"] = ff_latent_rel
    jsonl_records.append(record)

    if (idx + 1) % 10 == 0 or idx == len(df) - 1:
        print(f"[{idx + 1}/{len(df)}] Saved {video_stem}")

# ======================== Write JSONL ========================
with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for record in jsonl_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done. JSONL written to {JSONL_PATH}, total {len(jsonl_records)} records.")
