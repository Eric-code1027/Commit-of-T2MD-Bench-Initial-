
import os
import sys
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


import requests
import os
import tempfile
from urllib.parse import urlparse
import shutil

def download_file_from_url_safe(url, local_path=None):
    """
    安全下载文件，使用临时文件避免文件破碎
    
    Args:
        url (str): 要下载的文件的URL
        local_path (str): 本地保存路径，如果为None则使用URL中的文件名
    
    Returns:
        str: 下载文件的本地路径
    """
    temp_file = None
    try:
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 发送HEAD请求获取文件信息
        head_response = requests.head(url, headers=headers, allow_redirects=True)
        
        # 确定本地文件名
        if local_path is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_file.mp4"
            local_path = filename
        
        # 确保目录存在
        os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else '.', exist_ok=True)
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb', 
            delete=False,
            suffix='.tmp',
            dir=os.path.dirname(local_path) or '.'
        )
        temp_path = temp_file.name
        
        # print(f"开始下载，临时文件: {temp_path}")
        
        # 发送GET请求下载到临时文件
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # 获取文件总大小（如果服务器提供）
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        # 下载文件
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                downloaded_size += len(chunk)
                
                # # 显示下载进度
                # if total_size > 0:
                #     percent = (downloaded_size / total_size) * 100
                #     print(f"\r下载进度: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        # print("\n下载完成，正在保存...")
        temp_file.close()
        
        # 原子操作：重命名临时文件为目标文件
        shutil.move(temp_path, local_path)
        
        print(f"文件已成功下载到: {local_path}")
        return local_path
        
    except Exception as e:
        # 发生错误时清理临时文件
        if temp_file and not temp_file.closed:
            temp_file.close()
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)
        print(f"下载失败: {e}")
        return None


class AudioVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 base_path=None, metadata_path=None,
                 csv_path=None,
                 dynamic_duration=False,
                 repeat=1,
                 target_audio_length=157,  # 降采样后的音频点数
                 target_video_frames=31,   # 降采样后的视频帧数
                 audio_sample_rate=16000,
                 video_fps=24,
                 audio_downsample_factor=512,
                 video_downsample_factor=4,
                ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        if metadata_path:
            self.load_metadata(os.path.join(base_path, metadata_path))
        if csv_path:
            self.load_metadata(csv_path)
        self.cos_data = (csv_path is not None)
        self.dynamic_duration = dynamic_duration

        self.target_audio_length = target_audio_length
        self.target_video_frames = target_video_frames
        self.audio_sample_rate = audio_sample_rate
        self.video_fps = video_fps
        self.audio_downsample_factor = audio_downsample_factor
        self.video_downsample_factor = video_downsample_factor
        self.frame_processor = ImageCropAndResize(None, None, 720*720, 32, 32) 
        
        # 计算原始需要的音频长度和视频帧数
        self.raw_audio_length = target_audio_length * audio_downsample_factor
        self.raw_video_frames = (target_video_frames-1) * video_downsample_factor + 1
        
        # 计算持续时间（秒） - 确保音频和视频持续时间一致
        self.duration = self.raw_video_frames / video_fps # 5.04s
        # self.duration = min(
        #     self.raw_audio_length / audio_sample_rate,
        #     self.raw_video_frames / video_fps
        # )
        
        # 重新计算实际的原始长度
        self.actual_raw_audio_length = int(self.duration * audio_sample_rate)
        self.actual_raw_video_frames = self.raw_video_frames
        
        print(f"Duration: {self.duration:.2f}s")
        print(f"Raw audio samples: {self.actual_raw_audio_length}")
        print(f"Raw video frames: {self.actual_raw_video_frames}")

    def load_metadata(self, metadata_path):
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def load_audio(self, audio_path):
        """加载并处理音频"""
        try:
            # 加载音频
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
                    # 从中间截取
                    start = (waveform.shape[1] - self.actual_raw_audio_length) // 2
                    waveform = waveform[:, start:start + self.actual_raw_audio_length]
                else:
                    # 如果音频太短，进行填充
                    padding = self.actual_raw_audio_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
        
            # 归一化
            # waveform = waveform.float() / 32767.

            return waveform.squeeze(0)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(self.target_audio_length)
    
    def load_test(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frame_idx in range(total_frames):
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # 20s
            ret, frame = cap.read() # 1.4s
            if ret:
                frames.append(frame)
        cap.release()
        # reader = imageio.get_reader(video_path) # 3.15s
        # num_frames = int(reader.count_frames())
        # frames = []
        # for frame_id in range(num_frames):
        #     frame = reader.get_data(frame_id)
        #     frame = Image.fromarray(frame)
        #     # frame = self.frame_processor(frame)
        #     frames.append(frame)
        # reader.close()

    def load_video(self, video_path):
        """加载并处理视频"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频信息
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算实际需要采样的时间点
            if video_fps < self.video_fps:
                print(f'Video {video_path} FPS {video_fps} too small')
                return None
            sample_interval_f = 1
            frames_to_read = total_frames
            if video_fps > self.video_fps:
                # 计算采样间隔（保持浮点精度）
                sample_interval_f = video_fps / self.video_fps
                frames_to_read = int(total_frames / sample_interval_f)
            
            if self.dynamic_duration:
                pass
            else:
                frames_to_read = min(frames_to_read, self.actual_raw_video_frames)
                if frames_to_read < self.actual_raw_video_frames:
                    print(f'Video {video_path} duration {frames_to_read/self.video_fps}s too small')
                    return None

            j = 0
            raw_frames = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                frame_pos = int(j * sample_interval_f)
                if i == frame_pos:
                    raw_frames.append(frame)
                    j += 1
            # if len(raw_frames) != frames_to_read:
                # import pdb; pdb.set_trace()
            # assert len(raw_frames) == frames_to_read

            frames = []
            for frame in raw_frames:
                # 转换BGR到RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                frame = self.frame_processor(frame)
                frame = np.array(frame).transpose(2, 0, 1) # c h w
                frame_tensor = torch.from_numpy(frame).float() / 255.
                frame_tensor = frame_tensor * 2. - 1.
                frames.append(frame_tensor) # c h w
            
            cap.release()
            return torch.stack(frames, dim=1) # c f h w
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")

    def load_video_deprecated(self, video_path):
        """加载并处理视频"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频信息
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算实际需要采样的时间点
            if video_fps < self.video_fps:
                print(f'Video {video_path} FPS {video_fps} too small')
                return None
            sample_interval_f = 1
            frames_to_read = total_frames
            if video_fps > self.video_fps:
                # 计算采样间隔（保持浮点精度）
                sample_interval_f = video_fps / self.video_fps
                frames_to_read = int(total_frames / sample_interval_f)
            
            if self.dynamic_duration:
                pass
            else:
                frames_to_read = min(frames_to_read, self.actual_raw_video_frames)
                if frames_to_read < self.actual_raw_video_frames:
                    print(f'Video {video_path} duration {frames_to_read/self.video_fps}s too small')
                    return None

            frames = []
            
            for i in range(frames_to_read):
                # 精确计算帧位置
                frame_pos = int(i * sample_interval_f)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 转换BGR到RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                frame = self.frame_processor(frame)

                # 转换为tensor并归一化到[-1, 1]
                frame = np.array(frame).transpose(2, 0, 1) # c h w
                frame_tensor = torch.from_numpy(frame).float() / 255.
                frame_tensor = frame_tensor * 2. - 1.
                frames.append(frame_tensor) # c h w
            
            cap.release()
            return torch.stack(frames, dim=1) # c f h w
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        if self.cos_data:
            vid = data['videoid']
            if float(data['duration']) < 3: return None
            if float(data['duration']) > 10: return None
            if int(data['frame_rate']) < 24: return None
            if int(data['shape_l']) * int(data['shape_w']) < 620*620: return None
            struct_caption = json.loads(data['structure_caption'])
            try:
                caption = struct_caption['medium_caption'] + '<AUDCAP>' + data['audio_caption_new'] + '<ENDAUDCAP>'
            except Exception as e:
                print(e)
                return None
            data['prompt'] = caption
            data['audio'] = self.load_audio(data['audio_path']).unsqueeze(0)
            url = data['video_cos_url']
            local_path = f'./datasets/video_sft_100k/{vid}.mp4'
            if not os.path.exists(local_path):
                if download_file_from_url_safe(url, local_path) is None: return None
            data['video_path'] = local_path
            video = self.load_video(local_path)
            if video is None: return None
            data['video'] = video.unsqueeze(0)
        else:
            data['audio_path'], data['video_path'] = data['audio'], data['video']
            data['audio'] = self.load_audio(os.path.join(self.base_path, data['audio'])).unsqueeze(0)
            data['video'] = self.load_video(os.path.join(self.base_path, data['video'])).unsqueeze(0)
        # TODO check if ok
        if data['audio'] is None or data['video'] is None:
            return None
        return data

    def __len__(self):
        return int(len(self.data) * self.repeat)
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

    def __call__(self, video_path, audio_path):
        """同时加载视频和音频，确保时间对齐"""
        audio = self.load_audio(audio_path)
        video = self.load_video(video_path)
        
        # 验证时间对齐
        audio_duration = len(audio) / self.audio_sample_rate
        video_duration = video.shape[1] / self.video_fps
        
        print(f"Final audio duration: {audio_duration:.3f}s, points: {len(audio)}")
        print(f"Final video duration: {video_duration:.3f}s, frames: {video.shape[1]}")
        
        return {
            'video': video,      # [C, 31, H, W]
            'audio': audio,      # [157]
            'video_path': video_path,
            'audio_path': audio_path
        }

    def filter(self, data):
        vid = data['videoid']
        if float(data['duration']) < 3: return None
        if float(data['duration']) > 10: return None
        if int(data['frame_rate']) < 24: return None
        if int(data['shape_l']) * int(data['shape_w']) < 620*620: return None
        struct_caption = json.loads(data['structure_caption'])
        try:
            caption = struct_caption['medium_caption'] + '<AUDCAP>' + data['audio_caption_new'] + '<ENDAUDCAP>'
        except Exception as e:
            print(e)
            return None
        data['prompt'] = caption
        url = data['video_cos_url']
        local_path = f'./datasets/video_sft_100k/{vid}.mp4'
        if not os.path.exists(local_path):
            if download_file_from_url_safe(url, local_path) is None: return None
        return True

# 使用示例
if __name__ == "__main__":
    import sys
    from ovi.utils.io_utils import save_video
    loader = AudioVideoDataset(dynamic_duration=True, csv_path='./mini_100k_full_info.csv')

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import pandas as pd

    def process_chunk(chunk):
        """处理一个数据块"""
        results = []
        for data in chunk:
            if loader.filter(data.copy()):
                results.append(data)
        return results

    # 将数据分成块
    def chunk_data(data_list, chunk_size=1000):
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i:i + chunk_size]

    max_workers = min(mp.cpu_count(), 32)
    chunk_size = len(loader.data) // max_workers

    filtered_data = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunks = list(chunk_data(loader.data, chunk_size))
        
        # 处理每个块
        for result_chunk in executor.map(process_chunk, chunks):
            filtered_data.extend(result_chunk)

    # filtered_data = []
    # for data in loader.data:
    #     if loader.filter(data.copy()):
    #         filtered_data.append(data)
    pd.DataFrame(filtered_data).to_csv('./datasets/mini_100k_filtered.csv', index=False)
    exit()

    import time

    file = sys.argv[1]
    start = time.time()
    loader.load_video(file)
    print(time.time() - start)
    exit()

    # 假设视频和音频文件路径
    video_file = "/root/Ovi/outputs/t2v_720x720_seed103/Gen8_A_medium_shot_shows_a_woman_and_a_man,_both_adorne_720x720_103_0.mp4"
    audio_file = video_file + '.wav'
    
    data = loader(video_file, audio_file)
    
    print(f"Video shape: {data['video'].shape}")  # 应该是 [3, 31, H, W]
    print(f"Audio shape: {data['audio'].shape}")  # 应该是 [157]
    import pdb; pdb.set_trace()
    save_video('/tmp/Adata.mp4', data['video'].numpy(), data['audio'].numpy(), fps=24, sample_rate=16000)

# Audio: 16k, [-1, 1], pad {(nfft-hop)/2, (1024-256)/2} (mod 256)
# Video: 24fps, 121frames(1+30*4), preprocess_image_tensor, [-1, 1], shape [bchw]

## step1: 音视频读取 -> 写入文件
## step2: 音视频分别过 VAE -> 写入文件
