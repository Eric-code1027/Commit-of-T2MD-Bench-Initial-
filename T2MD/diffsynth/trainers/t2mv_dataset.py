
import os
import sys

# Append project root to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import json
import torch
import torchaudio
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import imageio


class ImageCropAndResize:
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class AudioVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 base_path=None, metadata_path=None,
                 csv_path="./code/4s_pipeline/crawler_code/a_v_caption_filtered.jsonl",
                 dynamic_duration=False,
                 repeat=1,
                 height=480, width=480,
                 target_audio_length=157,  # 降采样后的音频点数
                 target_video_frames=30,   # 降采样后的视频帧数, 30 -> 117 frames (4.875s) for 5s video
                 audio_sample_rate=16000,
                 video_fps=24,
                 audio_downsample_factor=512,
                 video_downsample_factor=4,
                ):
        # 强制使用指定的数据集路径
        # self.csv_path = "./code/4s_pipeline/crawler_code/a_v_caption_filtered.jsonl"
        self.csv_path = csv_path
        print(f"[Dataset] Using dataset file: {self.csv_path}")
        
        self.base_path = base_path
        self.repeat = repeat
        self.load_metadata(self.csv_path)
        self.dynamic_duration = dynamic_duration

        self.target_audio_length = target_audio_length
        self.target_video_frames = target_video_frames
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        self.audio_downsample_factor = audio_downsample_factor
        self.video_downsample_factor = video_downsample_factor
        if height is not None and width is not None:
            self.frame_processor = ImageCropAndResize(height, width, height*width, 32, 32)
        else:
            self.frame_processor = ImageCropAndResize(None, None, 720*720, 32, 32)
        
        # 计算原始需要的音频长度和视频帧数
        self.raw_audio_length = target_audio_length * audio_downsample_factor
        self.raw_video_frames = (target_video_frames-1) * video_downsample_factor + 1
        
        # 计算持续时间（秒） - 确保音频和视频持续时间一致
        self.duration = self.raw_video_frames / video_fps # 5.04s
        
        # 重新计算实际的原始长度
        self.actual_raw_audio_length = int(self.duration * audio_sample_rate)
        self.actual_raw_video_frames = self.raw_video_frames
        
        print(f"Duration: {self.duration:.2f}s")
        print(f"Raw audio samples: {self.actual_raw_audio_length}")
        print(f"Raw video frames: {self.actual_raw_video_frames}")

    def load_metadata(self, metadata_path):
        self.data = []
        if metadata_path.endswith(".jsonl"):
            with open(metadata_path, 'r') as f:
                for line in f:
                    try:
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                self.data = json.load(f)
        else:
            # Fallback for CSV if needed, but primary use case is JSONL now
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        print(f"[Dataset] Loaded {len(self.data)} samples.")

    def load_audio(self, audio_path):
        """加载并处理音频"""
        try:
            # 加载音频
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return torch.zeros(self.actual_raw_audio_length)

            waveform, orig_sr = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 重采样到16kHz
            if orig_sr != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, self.audio_sample_rate)
                waveform = resampler(waveform)
            
            if self.dynamic_duration:
                pass
            else:
                # 截取指定长度的音频
                if waveform.shape[1] > self.actual_raw_audio_length:
                    # 从头截取，与视频保持一致
                    waveform = waveform[:, :self.actual_raw_audio_length]
                else:
                    # 如果音频太短，进行填充
                    padding = self.actual_raw_audio_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
        
            return waveform.squeeze(0)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(self.actual_raw_audio_length) # Return correct length zeros
    
    def load_video(self, video_path):
        """加载并处理视频"""
        try:
            # from pudb.remote import set_trace ; set_trace()
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return None

            cap = cv2.VideoCapture(video_path)
            
            # 获取视频信息
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算实际需要采样的时间点
            if video_fps < 1: # Sanity check
                 print(f'Video {video_path} invalid FPS: {video_fps}')
                 return None

            if video_fps < self.video_fps:
                # Allow slightly lower FPS but warn? Or strict? Keeping strict for now based on prev logic
                # print(f'Video {video_path} FPS {video_fps} too small')
                # return None
                pass # 暂时允许，后面会按时间采样
            
            sample_interval_f = 1.0
            if video_fps > self.video_fps:
                sample_interval_f = video_fps / self.video_fps
            
            frames_to_read = self.actual_raw_video_frames
            
            # 简单的时间对齐读取
            frames = []
            for i in range(frames_to_read):
                frame_idx = int(i * sample_interval_f)
                if frame_idx >= total_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 转换BGR到RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = self.frame_processor(frame)
                    frame = np.array(frame).transpose(2, 0, 1) # c h w
                    frame_tensor = torch.from_numpy(frame).float() / 255.
                    frame_tensor = frame_tensor * 2. - 1.
                    frames.append(frame_tensor)
                else:
                    break
            
            cap.release()

            if len(frames) < frames_to_read:
                # print(f"Video {video_path} not enough frames: {len(frames)}/{frames_to_read}")
                return None

            return torch.stack(frames, dim=1) # c f h w
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    def __getitem__(self, data_id):
        # 循环获取直到成功
        for _ in range(10): # 尝试10次，避免死循环
            idx = (data_id + _) % len(self.data)
            item = self.data[idx]
            
            video_path = item.get('video_path')
            audio_path = item.get('audio_path')
            # print(f"video_path: {video_path}, audio_path: {audio_path}")
            
            if not video_path or not audio_path:
                print(f"video_path or audio_path is None")
                continue

            # 处理 Prompt
            try:
                audio_cap = item.get('audio_caption', '')
                # audio_caption 可能包含 assistant\n ... 等前缀，根据用户需求这里直接用 full content
                # 或者需要清洗？用户示例直接是 content。这里假设直接用字符串。
                
                video_cap_obj = item.get('video_caption', {})
                if isinstance(video_cap_obj, str):
                     # 可能是字符串形式的 json
                     try:
                        video_cap_obj = json.loads(video_cap_obj)
                     except:
                        pass

                video_cap_str = video_cap_obj.get('caption', '')
                # caption 字段也是 json string
                if isinstance(video_cap_str, str):
                    try:
                        video_cap_inner = json.loads(video_cap_str)
                        video_cap = video_cap_inner.get('medium_caption', '')
                    except:
                        video_cap = video_cap_str # Fallback
                else:
                    video_cap = ""

                prompt = f"<audio_cap>{audio_cap}</audio_cap>\n<sep>\n<video_cap>{video_cap}</video_cap>"
                # print(f"prompt: {prompt}")
                # print("########################################################")
            except Exception as e:
                print(f"Error parsing caption for {video_path}: {e}")
                continue

            # print("success1")
            print(f"video_path: {video_path}")
            # 加载音视频
            video = self.load_video(video_path)
            if video is None:
                print(f"video is None")
                continue
            
            # print("success2")
            audio = self.load_audio(audio_path)
            # audio 返回 zeros 如果失败，这里可以检查一下是否全0且需要过滤? 
            # 暂时假设 audio 加载失败返回 zeros 也是允许的(静音)，或者可以在 load_audio 里处理

            # print("success3")
            return {
                'video': video.unsqueeze(0),      # [1, C, F, H, W] (unsqueeze to match expected input if needed, or check consumer)
                                                  # 注意：之前的代码 __getitem__ 里做了 unsqueeze(0)
                                                  # data['video'] = self.load_video(...).unsqueeze(0)
                'audio': audio.unsqueeze(0),      # [1, L]
                'prompt': prompt,
                'video_path': video_path,
                'audio_path': audio_path
            }
        
        return None # Should be handled by collate_fn

    def __len__(self):
        return int(len(self.data) * self.repeat)

    def __call__(self, video_path, audio_path):
        """测试用"""
        audio = self.load_audio(audio_path)
        video = self.load_video(video_path)
        return {
            'video': video,
            'audio': audio,
            'video_path': video_path,
            'audio_path': audio_path
        }

if __name__ == "__main__":
    print("Starting dataset test...")
    # 测试实例化
    dataset = AudioVideoDataset()
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # 获取第一个样本
        print("Fetching first sample...")
        sample = dataset[1]
        
        if sample:
            print("Sample fetched successfully:")
            print(f"Video shape: {sample['video'].shape}")
            print(f"Audio shape: {sample['audio'].shape}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Video Path: {sample['video_path']}")
            
            from ovi.utils.io_utils import save_video
            # 注意：save_video 通常期望 [C, F, H, W] (或者 [F, C, H, W]?) 和 [L]
            # dataset返回的是 unsqueeze过的 [1, ...], 需要 squeeze 回来
            vid_tensor = sample['video'].squeeze(0)
            aud_tensor = sample['audio'].squeeze(0)
            
            output_path = './code/4s_pipeline/crawler_code/DiffSynth-Studio/mytest/test_dataset_output.mp4'
            print(f"Saving debug video to {output_path}...")
            # 假设 save_video 接受 numpy 或 tensor
            # 根据之前的代码: save_video('/tmp/Adata.mp4', generated_video, generated_audio, ...)
            # generated_video 是 numpy
            
            vid_np = vid_tensor.cpu().numpy()
            aud_np = aud_tensor.cpu().numpy()
            
            save_video(output_path, vid_np, aud_np, fps=24, sample_rate=16000)
            print("Video saved.")
        else:
            print("Failed to fetch sample (returned None).")
    else:
        print("Dataset is empty.")
