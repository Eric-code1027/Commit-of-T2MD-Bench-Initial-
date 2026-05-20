# This script can generate a single video clip.
# If you need generate long videos, please refer to `Wan2.2-S2V-14B_multi_clips.py`.
import os
import sys
# 添加项目根目录到路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import torch
from PIL import Image
import librosa
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_s2v import WanVideoS2VPipeline, ModelConfig
from modelscope import dataset_snapshot_download

# 模型路径使用相对路径，可通过环境变量 CKPT_DIR 覆盖
_ckpt_dir = os.environ.get("CKPT_DIR", os.path.join(_project_root, "ckpts"))
pipe = WanVideoS2VPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": os.environ.get("SFT_CKPT_PATH", os.path.join(_project_root, "ckpts", "Wan2.2-S2V-5B", "step-19000.safetensors"))}),
        ModelConfig(path=os.path.join(_ckpt_dir, "Wan2.2-S2V-14B", "wav2vec2-large-xlsr-53-english", "model.safetensors")),
        ModelConfig(path=os.path.join(_ckpt_dir, "Wan2.2-S2V-14B", "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=os.path.join(_ckpt_dir, "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")),
    ],
    tokenizer_config=ModelConfig(path=os.path.join(_ckpt_dir, "Wan2.2-S2V-14B", "google", "umt5-xxl", "")),
    audio_processor_config=ModelConfig(path=os.path.join(_ckpt_dir, "Wan2.2-S2V-14B", "wav2vec2-large-xlsr-53-english", "")),
)


num_frames = 121 # 4n+1
height = 832
width = 448

prompt = "a person is speaking"
negative_prompt = ""
_input_dir = os.environ.get("S2V_INPUT_DIR", os.path.join(_project_root, "outputs", "s2v_input"))
input_image = Image.open(os.path.join(_input_dir, "evan_448x832.jpg")).convert("RGB").resize((width, height))
# s2v audio input, recommend 16kHz sampling rate
audio_path = os.path.join(_input_dir, "paoxie_5s_16k.wav")
input_audio, sample_rate = librosa.load(audio_path, sr=16000)



result = pipe.test_backward_simulation_s2v(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    cfg_scale=5.0,
    num_inference_steps=40,
    sigma_shift=5.0,
)
video = result[1:]



out_dir = os.environ.get("S2V_OUTPUT_DIR", os.path.join(_project_root, "outputs", "s2v_output"))
os.makedirs(out_dir, exist_ok=True)
index = len(os.listdir(out_dir))
save_video_with_audio(video, f"{out_dir}/video_Wan2.2-S2V-5B-org-{index}.mp4", audio_path, fps=24, quality=5)

