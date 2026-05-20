# This script can generate a single video clip.
# If you need generate long videos, please refer to `Wan2.2-S2V-14B_multi_clips.py`.
import os
import sys
sys.path.append("./github_projects/from_modelscope/DiffSynth-Studio") # FIXME: remove this
import torch
from PIL import Image
import librosa
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_s2v import WanVideoS2VPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoS2VPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-5B_full_multi_node_time4/step-15000.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-5B_full-PIM_multi_node_time2/step-12000.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-TI-D2Step-5B_full_multi_node2/step-2000.safetensors"}),
        ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-5B_full_multi_node_time4/step-19000.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-DD-5B_full_multi_node/step-8000.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./github_projects/from_modelscope/DiffSynth-Studio/models/train/Evan-Wan2.2-S2V-TI-D2Step-5B_full/step-400-ema.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-DMD-V3-5B_full_multi_node/step-3000.safetensors"}),
        # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-TI-D2Step-5B_full_multi_node2/step-1000.safetensors"}),
        ModelConfig(path="./Ovi/ckpts/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"),
        ModelConfig(path="./Ovi/ckpts/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="./Ovi/ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path="./Ovi/ckpts/Wan2.2-S2V-14B/google/umt5-xxl/"),
    audio_processor_config=ModelConfig(path="./Ovi/ckpts/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/"),
)

# pipe = WanVideoS2VPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./exp_dir/wan_s2v/talk_head_70w/Evan-Wan2.2-S2V-5B_full_multi_node_time4/step-19000.safetensors"}),
#         # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./github_projects/from_modelscope/DiffSynth-Studio/models/train/Evan-Wan2.2-S2V-DMD-5B_full/step-1000.safetensors"}),
#         # ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./released_models/wan_s2v/si2v_5b_stage2/step-1000-distill.safetensors"}),
#         ModelConfig(path="dummy_s2v_5b_model", infer_kwargs={"sft_ckpt_path": "./github_projects/from_modelscope/DiffSynth-Studio/models/train/Evan-Wan2.2-S2V-TI-SelfForcing-5B/step-800.safetensors"}),
#         ModelConfig(path="./release_models/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"),
#         ModelConfig(path="./release_models/Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
#         ModelConfig(path="./release_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
#     ],
#     tokenizer_config=ModelConfig(path="./release_models/Wan2.2-S2V-14B/google/umt5-xxl/"),
#     audio_processor_config=ModelConfig(path="./release_models/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/"),
# )

num_frames = 121 # 4n+1
height = 832
width = 448

prompt = "a person is speaking"
negative_prompt = ""
# 从视频读取第一帧作为input_image
# video_path = './train_datasets/wan_s2v/example_video_dataset_large/videos/00002b2b0983c5f328433e08a4da03e5_5_fb06fe7280e915d708f730cdabbaf8c4_vtrack_f5s.mp4'
# video_data = VideoData(video_path, height=height, width=width)
# input_image = video_data[0]  # 获取第一帧
input_image = Image.open("./github_projects/from_modelscope/DiffSynth-Studio/outputs/images/evan_448x832.jpg").convert("RGB").resize((width, height))
# s2v audio input, recommend 16kHz sampling rate
# audio_path = './train_datasets/wan_s2v/example_video_dataset_large/audios/00002b2b0983c5f328433e08a4da03e5_5_fb06fe7280e915d708f730cdabbaf8c4_atrack_f5s.mp3'
# audio_path = './github_projects/from_modelscope/DiffSynth-Studio/outputs/audios/hello_5s_16k.wav'
audio_path = './github_projects/from_modelscope/DiffSynth-Studio/outputs/audios/paoxie_5s_16k.wav'
input_audio, sample_rate = librosa.load(audio_path, sr=16000)

# Speech-to-video (streaming)
# result = pipe.generate_streaming(
#     prompt=prompt,
#     input_image=input_image,
#     negative_prompt=negative_prompt,
#     seed=0,
#     num_frames=num_frames,
#     height=height,
#     width=width,
#     audio_sample_rate=sample_rate,
#     input_audio=input_audio,
#     num_inference_steps=40,
#     sigma_shift=5.0,
#     chunk_frames=25,
#     inject_motion_latents=True,
#     motion_latent_frames=2,
#     audio_encode_mode="once",
#     fps=24,
# )
# video = result["video"]

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
    num_inference_steps=4,
    sigma_shift=5.0,
)
video = result[1:]

# result = pipe(
#     prompt=prompt,
#     input_image=input_image,
#     negative_prompt=negative_prompt,
#     seed=0,
#     num_frames=num_frames,
#     height=height,
#     width=width,
#     audio_sample_rate=sample_rate,
#     input_audio=input_audio,
#     num_inference_steps=40,
#     sigma_shift=5.0,
# )
# video = result[1:]

out_dir = "./github_projects/from_modelscope/DiffSynth-Studio/outputs/inference_dmd_d4step"
os.makedirs(out_dir, exist_ok=True)
index = len(os.listdir(out_dir))
save_video_with_audio(video, f"{out_dir}/video_Wan2.2-S2V-5B-evan-{index}.mp4", audio_path, fps=24, quality=5)

# save_video_with_audio(video[1:], f"{out_dir}/video_Wan2.2-S2V-5B-evan-{index}.mp4", audio_path, fps=24, quality=5)
# s2v will use the first (num_frames) frames as reference. height and width must be the same as input_image. And fps should be 16, the same as output video fps.
# pose_video_path = 'data/example_video_dataset/wans2v/pose.mp4'
# pose_video = VideoData(pose_video_path, height=height, width=width)

# # Speech-to-video with pose
# video = pipe(
#     prompt=prompt,
#     input_image=input_image,
#     negative_prompt=negative_prompt,
#     seed=0,
#     num_frames=num_frames,
#     height=height,
#     width=width,
#     audio_sample_rate=sample_rate,
#     input_audio=input_audio,
#     s2v_pose_video=pose_video,
#     num_inference_steps=40,
# )
# save_video_with_audio(video[1:], "video_2_Wan2.2-S2V-14B.mp4", audio_path, fps=24, quality=5)


# ============================================================
# FlashHead Mode Inference Examples
# ============================================================
# FlashHead mode allows the model to optionally use motion guidance during inference.
# When motion_video is provided, the model uses it as guidance.
# When motion_video is None, the model generates freely (similar to training with motion dropout).

# Example 1: FlashHead mode without motion (model generates freely)
# result = pipe(
#     prompt=prompt,
#     input_image=input_image,
#     negative_prompt=negative_prompt,
#     seed=0,
#     num_frames=num_frames,
#     height=height,
#     width=width,
#     audio_sample_rate=sample_rate,
#     input_audio=input_audio,
#     num_inference_steps=40,
#     sigma_shift=5.0,
#     # FlashHead mode parameters
#     inject_motion_as_prefix=True,  # Enable FlashHead mode
#     motion_prefix_frames=2,         # Number of motion frames in latent domain
#     # motion_video=None,  # Not providing motion_video (model generates freely)
# )
# video = result[1:]
# save_video_with_audio(video, f"{out_dir}/video_Wan2.2-S2V-5B-flashhead_no_motion-{index}.mp4", audio_path, fps=24, quality=5)

# Example 2: FlashHead mode with motion guidance (provide a few frames as motion hint)
# motion_video_path = "/path/to/motion_reference_video.mp4"
# motion_video = VideoData(motion_video_path, height=height, width=width)
# motion_frames = motion_video[:9]  # Take first ~9 frames (2 latent frames * 4 + 1)
# result = pipe(
#     prompt=prompt,
#     input_image=input_image,
#     negative_prompt=negative_prompt,
#     seed=0,
#     num_frames=num_frames,
#     height=height,
#     width=width,
#     audio_sample_rate=sample_rate,
#     input_audio=input_audio,
#     num_inference_steps=40,
#     sigma_shift=5.0,
#     # FlashHead mode parameters
#     inject_motion_as_prefix=True,  # Enable FlashHead mode
#     motion_prefix_frames=2,         # Number of motion frames in latent domain
#     motion_video=motion_frames,     # Provide motion frames for guidance
# )
# video = result[1:]
# save_video_with_audio(video, f"{out_dir}/video_Wan2.2-S2V-5B-flashhead_with_motion-{index}.mp4", audio_path, fps=24, quality=5)
