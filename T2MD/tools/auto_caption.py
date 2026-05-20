"""
自动给 10 条跳舞数据打 caption(视频 + 音频),输出符合 t2mv_dataset.py 格式的 jsonl。

依赖:
    pip install transformers accelerate librosa decord
模型(首次运行自动下,各 ~16GB):
    - Qwen/Qwen2.5-VL-7B-Instruct  (视频 caption)
    - Qwen/Qwen2-Audio-7B-Instruct (音频 caption)

如果你显存不够同时跑两个,可以分两次跑(先视频,显存释放后再音频)。
"""
import os, json, gc, glob
import torch
from pathlib import Path

DATA_DIR = "datasets/dance_10"
OUT_JSONL = f"{DATA_DIR}/dance_caption.jsonl"
N = 10

VIDEO_PROMPT = (
    "Describe this dance video in 2-3 sentences. Focus on: "
    "(1) the dancer's appearance and clothing, "
    "(2) specific dance moves and body language, "
    "(3) the scene, lighting and camera angle. "
    "Be concrete and visual."
)

AUDIO_PROMPT = (
    "Describe this music clip in 1-2 sentences. Include: "
    "genre, tempo (BPM if estimable), main instruments, mood, and energy level."
)


def video_caption_all():
    """用 Qwen2.5-VL 给 10 条视频打 caption"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading Qwen2.5-VL-7B (first time will download ~16GB) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    captions = {}
    for i in range(1, N + 1):
        name = f"{i:03d}"
        vpath = os.path.abspath(f"{DATA_DIR}/raw/{name}.mp4")
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": vpath, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": VIDEO_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        captions[name] = gen
        print(f"[VIDEO {name}] {gen[:100]}...")

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return captions


def audio_caption_all():
    """用 Qwen2-Audio 给 10 条音频打 caption"""
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    import librosa
    print("Loading Qwen2-Audio-7B (first time will download ~16GB) ...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    captions = {}
    for i in range(1, N + 1):
        name = f"{i:03d}"
        apath = f"{DATA_DIR}/audio/{name}.wav"
        wav, sr = librosa.load(apath, sr=processor.feature_extractor.sampling_rate)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": apath},
                {"type": "text", "text": AUDIO_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[wav], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        gen = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        captions[name] = gen
        print(f"[AUDIO {name}] {gen[:100]}...")

    del model, processor
    gc.collect(); torch.cuda.empty_cache()
    return captions


def write_jsonl(video_caps, audio_caps):
    """严格按 t2mv_dataset.py:240-264 的格式写 jsonl"""
    base = os.path.abspath(DATA_DIR)
    lines = []
    for i in range(1, N + 1):
        name = f"{i:03d}"
        vc = video_caps[name]
        ac = audio_caps[name]
        item = {
            "video_path": f"{base}/raw/{name}.mp4",
            "audio_path": f"{base}/audio/{name}.wav",
            "audio_caption": ac,
            "video_caption": {
                "caption": json.dumps({"medium_caption": vc})
            },
        }
        lines.append(json.dumps(item, ensure_ascii=False))
    Path(OUT_JSONL).write_text("\n".join(lines))
    print(f"\n[done] wrote {OUT_JSONL}")
    print("verify with: head -1 " + OUT_JSONL)


if __name__ == "__main__":
    video_caps = video_caption_all()
    audio_caps = audio_caption_all()
    write_jsonl(video_caps, audio_caps)