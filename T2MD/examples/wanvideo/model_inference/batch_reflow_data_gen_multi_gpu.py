import os
import sys
import json
import pathlib
import pandas as pd
import torch
import torch.multiprocessing as mp
import librosa
from PIL import Image

sys.path.append("./github_projects/from_modelscope/DiffSynth-Studio")
from diffsynth.utils.data import VideoData
from diffsynth.pipelines.wan_video_s2v import WanVideoS2VPipeline, ModelConfig

# ======================== Config ========================
DEBUG = True
USE_TQDM = False
WORLD_SIZE = 8
MAX_SAMPLES = 100000

CSV_PATH = "./train_datasets/wan_s2v/example_video_dataset_large/evan_metadata_s2v_with_prompt_overfit.csv"
BASE_PATH = "./train_datasets/wan_s2v/example_video_dataset_large"
OUTPUT_ROOT = "./train_datasets/wan_s2v/distill_video_dataset"
JSONL_PATH = os.path.join(OUTPUT_ROOT, "reflow_data.jsonl")

SFT_CKPT = "./released_models/wan_s2v/si2v_5b_stage2/step-19000.safetensors"

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


def _identity_progress(iterable, **kwargs):
    return iterable


def worker(rank, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    if USE_TQDM:
        from tqdm import tqdm
        progress_bar_cmd = tqdm
    else:
        progress_bar_cmd = _identity_progress

    # -------------------- Load model --------------------
    pipe = WanVideoS2VPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": SFT_CKPT}),
            ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"),
            ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="./github_projects/Wan2.2/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/google/umt5-xxl/"),
        audio_processor_config=ModelConfig(path="./github_projects/from_modelscope/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/"),
    )

    # -------------------- Load & split data --------------------
    df = pd.read_csv(CSV_PATH)
    if DEBUG:
        df = df.head(10)
    elif MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)

    all_indices = list(range(len(df)))
    my_indices = all_indices[rank::world_size]
    total_local = len(my_indices)

    print(f"[Rank {rank}] Assigned {total_local} samples", flush=True)

    # -------------------- Inference --------------------
    local_records = []

    for done, local_i in enumerate(my_indices):
        row_dict = df.iloc[local_i].to_dict()
        original_idx = local_i

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
            seed=original_idx,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            audio_sample_rate=sample_rate,
            input_audio=input_audio,
            num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_scale=CFG_SCALE,
            progress_bar_cmd=progress_bar_cmd,
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
        record["_original_idx"] = original_idx
        local_records.append(record)

        print(f"[Rank {rank}] [{done + 1}/{total_local}] Saved {video_stem}", flush=True)

    # -------------------- Write per-rank JSONL --------------------
    tmp_jsonl = os.path.join(OUTPUT_ROOT, f"reflow_data_rank{rank}.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for record in local_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Rank {rank}] Done. Wrote {len(local_records)} records to {tmp_jsonl}", flush=True)


def merge_jsonl(world_size):
    all_records = []
    for rank in range(world_size):
        tmp_jsonl = os.path.join(OUTPUT_ROOT, f"reflow_data_rank{rank}.jsonl")
        if not os.path.exists(tmp_jsonl):
            continue
        with open(tmp_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    all_records.sort(key=lambda r: r["_original_idx"])

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for record in all_records:
            del record["_original_idx"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for rank in range(world_size):
        tmp_jsonl = os.path.join(OUTPUT_ROOT, f"reflow_data_rank{rank}.jsonl")
        if os.path.exists(tmp_jsonl):
            os.remove(tmp_jsonl)

    print(f"Merged JSONL written to {JSONL_PATH}, total {len(all_records)} records.")


if __name__ == "__main__":
    os.makedirs(NOISE_DIR, exist_ok=True)
    os.makedirs(LATENT_DIR, exist_ok=True)
    os.makedirs(FIRST_FRAME_DIR, exist_ok=True)

    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    merge_jsonl(WORLD_SIZE)
