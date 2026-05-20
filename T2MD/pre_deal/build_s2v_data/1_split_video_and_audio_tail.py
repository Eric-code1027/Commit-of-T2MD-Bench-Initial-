import os
import subprocess
import csv
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 输入输出路径
input_dir = "./train_datasets/wan_s2v/raw_data_cropped_832_448_5s"
output_base_dir = "./train_datasets/wan_s2v/example_video_dataset"

# 输出子目录
videos_dir = os.path.join(output_base_dir, "videos")
audios_dir = os.path.join(output_base_dir, "audios")
metadata_path = os.path.join(output_base_dir, "evan_metadata_s2v.csv")

# 进程数
num_processes = min(cpu_count(), 32)

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

def get_video_metadata(video_path):
    """获取视频的元数据：宽度、高度、帧数、帧率、时长"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate,duration',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            width = int(parts[0])
            height = int(parts[1])
            
            # 帧率处理
            frame_rate_str = parts[2]
            if '/' in frame_rate_str:
                num, den = map(float, frame_rate_str.split('/'))
                frame_rate = num / den
            else:
                frame_rate = float(frame_rate_str)
            
            # 帧数
            try:
                nb_frames = int(parts[3])
            except:
                nb_frames = None
            
            # 时长
            try:
                duration = float(parts[4]) if len(parts) > 4 else None
            except:
                duration = None
            
            return width, height, frame_rate, nb_frames, duration
    except Exception as e:
        print(f"获取视频元数据失败 {video_path}: {e}")
    return None, None, None, None, None

def extract_video_track(input_path, output_path):
    """提取视频轨道（不包含音频）"""
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', input_path,
        '-an',  # 不包含音频
        '-c:v', 'copy',  # 直接复制视频流，不重新编码
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

def extract_audio_track(input_path, output_path):
    """提取音频轨道并转换为16kHz采样率的mp3"""
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', input_path,
        '-vn',  # 不包含视频
        '-ar', '16000',  # 16kHz采样率
        '-ac', '1',  # 单声道（可选，如果需要立体声改为2）
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
    """处理单个视频的worker函数"""
    video_path, index, videos_dir, audios_dir = args
    
    # 生成输出文件名
    video_output_name = f"clip_{index:03d}.mp4"
    audio_output_name = f"audio_{index:03d}.mp3"
    
    video_output_path = os.path.join(videos_dir, video_output_name)
    audio_output_path = os.path.join(audios_dir, audio_output_name)
    
    # 如果输出文件已存在，跳过
    if os.path.exists(video_output_path) and os.path.exists(audio_output_path):
        # 获取元数据
        width, height, fps, nb_frames, duration = get_video_metadata(video_output_path)
        if width is not None:
            return {
                'status': 'skip',
                'video_path': video_output_name,
                'audio_path': audio_output_name,
                'width': width,
                'height': height,
                'fps': fps,
                'nb_frames': nb_frames,
                'duration': duration
            }
    
    # 提取视频轨道
    video_success, video_error = extract_video_track(video_path, video_output_path)
    if not video_success:
        return {'status': 'fail', 'video': video_path, 'error': f'视频轨道提取失败: {video_error}'}
    
    # 提取音频轨道
    audio_success, audio_error = extract_audio_track(video_path, audio_output_path)
    if not audio_success:
        # 如果音频提取失败，删除已生成的视频文件
        if os.path.exists(video_output_path):
            os.remove(video_output_path)
        return {'status': 'fail', 'video': video_path, 'error': f'音频轨道提取失败: {audio_error}'}
    
    # 获取视频元数据
    width, height, fps, nb_frames, duration = get_video_metadata(video_output_path)
    if width is None:
        return {'status': 'fail', 'video': video_path, 'error': '无法获取视频元数据'}
    
    return {
        'status': 'success',
        'video_path': video_output_name,
        'audio_path': audio_output_name,
        'width': width,
        'height': height,
        'fps': fps,
        'nb_frames': nb_frames,
        'duration': duration
    }

def main():
    # 创建输出目录
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(audios_dir, exist_ok=True)
    
    print(f"正在扫描视频文件: {input_dir}")
    video_files = get_video_files(input_dir)
    print(f"找到 {len(video_files)} 个视频文件")
    
    if len(video_files) == 0:
        print("没有找到视频文件！")
        return
    
    # 准备任务参数列表
    tasks = [
        (video_path, idx + 1, videos_dir, audios_dir)
        for idx, video_path in enumerate(video_files)
    ]
    
    print(f"使用 {num_processes} 个进程并行处理...")
    
    # 使用多进程处理
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_videos = []
    metadata_rows = []
    
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(total=len(video_files), desc="处理进度")
        
        for result in pool.imap(process_single_video, tasks, chunksize=1):
            if result['status'] == 'success':
                success_count += 1
                metadata_rows.append({
                    'video': f"videos/{result['video_path']}",
                    'input_audio': f"audios/{result['audio_path']}",
                    's2v_pose_video': ''
                })
                
            elif result['status'] == 'skip':
                skip_count += 1
                metadata_rows.append({
                    'video': f"videos/{result['video_path']}",
                    'input_audio': f"audios/{result['audio_path']}",
                    's2v_pose_video': ''
                })
                
            elif result['status'] == 'fail':
                fail_count += 1
                failed_videos.append(result)
            
            pbar.update(1)
            pbar.set_postfix({
                '成功': success_count,
                '跳过': skip_count,
                '失败': fail_count
            })
        
        pbar.close()
    
    # 生成CSV元数据文件
    if metadata_rows:
        print(f"\n正在生成元数据文件: {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['video', 'input_audio', 's2v_pose_video']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"元数据文件已生成: {metadata_path}")
    
    # 输出结果
    print(f"\n处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总共: {len(video_files)}")
    print(f"\n输出目录:")
    print(f"  视频: {videos_dir}")
    print(f"  音频: {audios_dir}")
    print(f"  元数据: {metadata_path}")
    
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
        for item in failed_videos[:3]:
            print(f"  - {os.path.basename(item['video'])}")
            if 'error' in item:
                error_preview = item['error'].split('\n')[0][:100]
                print(f"    {error_preview}")

if __name__ == "__main__":
    main()