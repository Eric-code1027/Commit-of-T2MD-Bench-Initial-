import os
import subprocess
import math
import csv
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 输入输出路径，可通过环境变量或命令行参数指定
import argparse
_parser = argparse.ArgumentParser(description="裁剪和修剪视频文件")
_parser.add_argument("--input_dir", "-i", default=os.environ.get("INPUT_DIR", "./videos"), help="输入视频目录")
_parser.add_argument("--output_dir", "-o", default=os.environ.get("OUTPUT_DIR", "./processed"), help="输出目录")
_args = _parser.parse_args()
input_dir = _args.input_dir
output_base_dir = _args.output_dir

# 输出子目录
videos_dir = os.path.join(output_base_dir, "videos")
audios_dir = os.path.join(output_base_dir, "audios")
metadata_path = os.path.join(output_base_dir, "metadata_s2v.csv")

# 目标参数
target_height = 832
target_width = 448
target_duration = 5  # 秒

# Prompt配置
default_prompt = "a person is speaking"

# 进程数（可以根据CPU核心数调整）
num_processes = min(cpu_count(), 64)  # 最多使用64个进程

# 视频文件扩展名
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

def get_video_files(directory):
    """递归获取目录下的所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(root, file))
    return video_files

def get_video_info(video_path):
    """获取视频的宽度和高度"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            width, height = map(int, result.stdout.strip().split(','))
            return width, height
    except Exception as e:
        print(f"获取视频信息失败 {video_path}: {e}")
    return None, None

def crop_and_extract_video_track(input_path, output_path, target_w, target_h, duration):
    """
    裁剪视频并提取视频轨道（无音频）
    硬性要求输出121帧在24fps下
    """
    # 获取原始视频尺寸
    orig_width, orig_height = get_video_info(input_path)
    if orig_width is None or orig_height is None:
        return False, "无法获取视频信息"
    
    # 计算中心裁剪参数
    scale_w = target_w / orig_width
    scale_h = target_h / orig_height
    scale = max(scale_w, scale_h)  # 选择较大的缩放比例，确保能覆盖目标区域
    
    # 使用ceil向上取整，确保缩放后尺寸不小于目标尺寸
    scaled_w = math.ceil(orig_width * scale)
    scaled_h = math.ceil(orig_height * scale)
    
    # 额外安全检查：确保缩放后尺寸不小于目标尺寸
    scaled_w = max(scaled_w, target_w)
    scaled_h = max(scaled_h, target_h)
    
    # 计算裁剪起始位置（中心裁剪）
    crop_x = (scaled_w - target_w) // 2
    crop_y = (scaled_h - target_h) // 2
    
    # 硬性要求121帧
    # 策略优化：
    # 1. 输入读取限制放宽到 10秒
    # 2. 先在末尾补足 1秒 的重复帧 (tpad)，确保即使源视频只有 5秒(120帧)，现在也有 6秒了。
    # 3. 再进行截取 (trim)，只取前 121 帧。
    # 效果：
    # - 长视频(>5.05s) -> 输出 121 帧真实动态
    # - 标准视频(5s)   -> 输出 120 帧真实动态 + 1 帧重复
    # - 短视频(<5s)    -> 输出 全部真实动态 + 补足到 121 帧 (如果补1秒够的话，不够还会少，但通常输入都>5s)
    
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-t', '10',  # 输入选项：只读取前10秒，足够长
        '-i', input_path,
        # 滤镜链：缩放裁剪 -> 转24fps -> 末尾补1秒(clone) -> 截取前121帧 -> 重置时间戳
        '-vf', f'scale={scaled_w}:{scaled_h}:flags=bicubic,crop={target_w}:{target_h}:{crop_x}:{crop_y},fps=24,tpad=stop_mode=clone:stop_duration=1,trim=end_frame=121,setpts=PTS-STARTPTS',
        '-an',  # 去除音频
        '-c:v', 'libx264',  # 使用H.264编码
        '-crf', '18',  # 质量参数
        '-preset', 'medium',
        '-frames:v', '121', # 强制输出标记为121帧
        '-y',  # 覆盖输出文件
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        if result.returncode == 0:
            return True, None
        else:
            error_msg = f"FFmpeg失败 (返回码: {result.returncode})"
            if result.stderr:
                error_msg += f"\n错误: {result.stderr[:500]}"
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "处理超时(5分钟)"
    except Exception as e:
        return False, f"异常: {str(e)}"

def extract_audio_track(input_path, output_path, duration):
    """
    提取音频轨道并转换为16kHz采样率的mp3（前N秒）
    """
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', input_path,
        '-t', str(duration),  # 只取前5秒
        '-vn',  # 不包含视频
        '-ar', '16000',  # 16kHz采样率
        '-ac', '1',  # 单声道
        '-c:a', 'libmp3lame',  # mp3编码
        '-b:a', '128k',  # 比特率
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, None
        else:
            return False, f"FFmpeg失败: {result.stderr[:200]}"
    except Exception as e:
        return False, f"异常: {str(e)}"

def process_single_video(args):
    """
    处理单个视频的worker函数，用于多进程
    完成：裁剪 + 视频轨道提取 + 音频轨道提取
    """
    video_path, index, videos_dir, audios_dir, target_w, target_h, duration = args
    
    # 从原始文件名生成输出文件名（去掉扩展名）
    original_name = Path(video_path).stem
    video_output_name = f"{original_name}_vtrack_f5s.mp4"
    audio_output_name = f"{original_name}_atrack_f5s.mp3"
    
    video_output_path = os.path.join(videos_dir, video_output_name)
    audio_output_path = os.path.join(audios_dir, audio_output_name)
    
    # 如果输出文件已存在，跳过
    if os.path.exists(video_output_path) and os.path.exists(audio_output_path):
        return {
            'status': 'skip',
            'video_path': video_output_name,
            'audio_path': audio_output_name
        }
    
    # 步骤1: 裁剪视频并提取视频轨道（无音频）
    video_success, video_error = crop_and_extract_video_track(
        video_path, video_output_path, target_w, target_h, duration
    )
    if not video_success:
        return {'status': 'fail', 'video': video_path, 'error': f'视频轨道处理失败: {video_error}'}
    
    # 步骤2: 提取音频轨道
    audio_success, audio_error = extract_audio_track(video_path, audio_output_path, duration)
    if not audio_success:
        # 如果音频提取失败，删除已生成的视频文件
        if os.path.exists(video_output_path):
            os.remove(video_output_path)
        return {'status': 'fail', 'video': video_path, 'error': f'音频轨道提取失败: {audio_error}'}
    
    return {
        'status': 'success',
        'video_path': video_output_name,
        'audio_path': audio_output_name
    }

def main():
    # 创建输出目录
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(audios_dir, exist_ok=True)
    
    print("=" * 80)
    print("Soul数据集视频音频分离处理流程：裁剪 + 轨道分离 + CSV生成")
    print("=" * 80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"目标尺寸: {target_width}x{target_height}")
    print(f"目标时长: {target_duration}秒")
    print(f"目标帧数: 121帧 @ 24fps")
    print(f"默认Prompt: '{default_prompt}'")
    print("=" * 80)
    
    print(f"\n正在扫描视频文件: {input_dir}")
    video_files = get_video_files(input_dir)
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"将处理所有找到的视频")
    
    if len(video_files) == 0:
        print("没有找到视频文件！")
        return
    
    # 准备任务参数列表
    tasks = [
        (video_path, idx + 1, videos_dir, audios_dir, target_width, target_height, target_duration)
        for idx, video_path in enumerate(video_files)
    ]
    
    print(f"使用 {num_processes} 个进程并行处理...")
    
    # 使用多进程处理，实时统计成功数
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_videos = []
    metadata_rows = []
    processed_count = 0
    
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered逐个获取结果
        pbar = tqdm(total=len(video_files), desc="处理进度")
        
        for result in pool.imap_unordered(process_single_video, tasks, chunksize=1):
            processed_count += 1
            
            if result['status'] == 'success':
                success_count += 1
                metadata_rows.append({
                    'video': f"videos/{result['video_path']}",
                    'input_audio': f"audios/{result['audio_path']}",
                    's2v_pose_video': '',
                    'prompt': default_prompt
                })
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count
                })
                    
            elif result['status'] == 'skip':
                skip_count += 1
                metadata_rows.append({
                    'video': f"videos/{result['video_path']}",
                    'input_audio': f"audios/{result['audio_path']}",
                    's2v_pose_video': '',
                    'prompt': default_prompt
                })
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count
                })
                
            elif result['status'] == 'fail':
                fail_count += 1
                failed_videos.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count
                })
        
        pbar.close()
    
    # 生成CSV元数据文件
    if metadata_rows:
        print(f"\n正在生成元数据文件: {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['video', 'input_audio', 's2v_pose_video', 'prompt']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"元数据文件已生成: {metadata_path}")
        print(f"CSV记录数: {len(metadata_rows)}")
    
    # 输出结果
    print(f"\n" + "=" * 80)
    print(f"处理完成！")
    print(f"=" * 80)
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总共: {len(video_files)}")
    print(f"\n输出目录:")
    print(f"  视频轨道: {videos_dir}")
    print(f"  音频轨道: {audios_dir}")
    print(f"  元数据CSV: {metadata_path}")
    print("=" * 80)
    
    # 如果有失败的视频，保存到日志文件
    if failed_videos:
        error_log_path = os.path.join(output_base_dir, "failed_videos.log")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"失败视频列表 (共{fail_count}个)\n")
            f.write("=" * 80 + "\n\n")
            for item in failed_videos:
                f.write(f"视频: {item['video']}\n")
                if 'error' in item:
                    f.write(f"错误: {item['error']}\n")
                f.write("-" * 80 + "\n")
        print(f"\n失败视频列表已保存到: {error_log_path}")
        print(f"\n最近的失败示例:")
        for item in failed_videos[:3]:  # 只显示前3个
            print(f"  - {os.path.basename(item['video'])}")
            if 'error' in item:
                error_preview = item['error'].split('\n')[0][:100]
                print(f"    {error_preview}")

if __name__ == "__main__":
    main()
