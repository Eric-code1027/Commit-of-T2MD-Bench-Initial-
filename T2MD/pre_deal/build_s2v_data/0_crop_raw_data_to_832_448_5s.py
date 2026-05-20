import os
import subprocess
import math
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 输入输出路径
input_dir = "./train_datasets/hunyuan_vid"
output_dir = "./train_datasets/wan_s2v/raw_data_cropped_832_448_5s"

# 目标参数
target_height = 832
target_width = 448
target_duration = 5  # 秒
max_videos_to_scan = int(1.5 * 10000)  # 扫描15000条视频
max_success_count = 10000  # 成功处理10000条后停止

# 进程数（可以根据CPU核心数调整）
num_processes = min(cpu_count(), 32)  # 最多使用32个进程

# 视频文件扩展名
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

def get_video_files(directory, max_count=None):
    """递归获取目录下的所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(root, file))
                if max_count and len(video_files) >= max_count:
                    return video_files
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

def crop_video(input_path, output_path, target_w, target_h, duration):
    """
    使用ffmpeg进行中心裁剪和时长截取
    """
    # 获取原始视频尺寸
    orig_width, orig_height = get_video_info(input_path)
    if orig_width is None or orig_height is None:
        return False, "无法获取视频信息"
    
    # 计算中心裁剪参数
    # 首先计算需要缩放到的尺寸（保持宽高比，确保至少一边能覆盖目标尺寸）
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
    
    # 构建ffmpeg命令
    # 1. 取前5秒 (-t 5)
    # 2. 缩放视频（使用ceil确保向上取整）
    # 3. 中心裁剪到目标尺寸
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',  # 只输出错误信息
        '-i', input_path,
        '-t', str(duration),  # 只取前5秒
        '-vf', f'scale={scaled_w}:{scaled_h}:flags=bicubic,crop={target_w}:{target_h}:{crop_x}:{crop_y}',
        '-c:v', 'libx264',  # 使用H.264编码
        '-crf', '18',  # 质量参数
        '-preset', 'medium',
        '-c:a', 'aac',  # 音频编码
        '-b:a', '128k',
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
                error_msg += f"\n错误: {result.stderr[:500]}"  # 只取前500字符
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "处理超时(5分钟)"
    except Exception as e:
        return False, f"异常: {str(e)}"

def process_single_video(args):
    """
    处理单个视频的worker函数，用于多进程
    """
    video_path, input_dir, output_dir, target_w, target_h, duration = args
    
    # 保持原始文件名
    relative_path = os.path.relpath(video_path, input_dir)
    output_path = os.path.join(output_dir, relative_path)
    
    # 创建输出子目录
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        return {'status': 'fail', 'video': video_path, 'error': f'创建目录失败: {e}'}
    
    # 如果输出文件已存在，跳过
    if os.path.exists(output_path):
        return {'status': 'skip', 'video': video_path}
    
    # 处理视频
    success, error_msg = crop_video(video_path, output_path, target_w, target_h, duration)
    if success:
        return {'status': 'success', 'video': video_path}
    else:
        return {'status': 'fail', 'video': video_path, 'error': error_msg}

def main():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在扫描视频文件: {input_dir}")
    video_files = get_video_files(input_dir, max_videos_to_scan)
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"目标: 成功处理 {max_success_count} 个视频")
    
    if len(video_files) == 0:
        print("没有找到视频文件！")
        return
    
    # 准备任务参数列表
    tasks = [
        (video_path, input_dir, output_dir, target_width, target_height, target_duration)
        for video_path in video_files
    ]
    
    print(f"使用 {num_processes} 个进程并行处理...")
    
    # 使用多进程处理，实时统计成功数
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_videos = []
    processed_count = 0
    
    with Pool(processes=num_processes) as pool:
        # 使用imap_unordered逐个获取结果，可以提前终止
        pbar = tqdm(total=max_success_count, desc="成功处理")
        
        for result in pool.imap_unordered(process_single_video, tasks, chunksize=1):
            processed_count += 1
            
            if result['status'] == 'success':
                success_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count,
                    '已处理': processed_count
                })
                
                # 达到目标后停止
                if success_count >= max_success_count:
                    print(f"\n已成功处理 {max_success_count} 个视频，停止处理。")
                    pool.terminate()  # 终止进程池
                    break
                    
            elif result['status'] == 'fail':
                fail_count += 1
                failed_videos.append(result)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count,
                    '已处理': processed_count
                })
                
            elif result['status'] == 'skip':
                skip_count += 1
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': fail_count,
                    '跳过': skip_count,
                    '已处理': processed_count
                })
        
        pbar.close()
    
    # 输出结果
    print(f"\n处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总共处理: {processed_count}/{len(video_files)}")
    print(f"输出目录: {output_dir}")
    
    # 如果有失败的视频，保存到日志文件
    if failed_videos:
        error_log_path = os.path.join(output_dir, "failed_videos.log")
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
