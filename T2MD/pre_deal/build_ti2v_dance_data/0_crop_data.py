import os
import subprocess
import math
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 输入输出路径
input_jsonl = ".//code/4s_pipeline/crawler_code/a_v_caption_bit.jsonl"
output_video_dir = ".//datasets/dance_from_xiaoda/cropped_832x448_video"
output_jsonl = ".//datasets/dance_from_xiaoda/a_v_caption_bit_cropped.jsonl"

# 目标参数
target_height = 832  # 竖屏高度
target_width = 448   # 竖屏宽度
target_frames = 121  # 精确121帧

# 进程数
num_processes = min(cpu_count(), 32)

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

def crop_video(input_path, output_path, target_w, target_h, num_frames):
    """
    使用ffmpeg进行中心裁剪和帧数截取
    使用滤镜链确保精确输出指定帧数
    """
    # 获取原始视频尺寸
    orig_width, orig_height = get_video_info(input_path)
    if orig_width is None or orig_height is None:
        return False, "无法获取视频信息"
    
    # 计算中心裁剪参数
    scale_w = target_w / orig_width
    scale_h = target_h / orig_height
    scale = max(scale_w, scale_h)
    
    scaled_w = math.ceil(orig_width * scale)
    scaled_h = math.ceil(orig_height * scale)
    
    scaled_w = max(scaled_w, target_w)
    scaled_h = max(scaled_h, target_h)
    
    crop_x = (scaled_w - target_w) // 2
    crop_y = (scaled_h - target_h) // 2
    
    # 滤镜链：缩放裁剪 -> 转24fps -> 末尾补帧 -> 截取前121帧 -> 重置时间戳
    filter_chain = f'scale={scaled_w}:{scaled_h}:flags=bicubic,crop={target_w}:{target_h}:{crop_x}:{crop_y},fps=24,tpad=stop_mode=clone:stop_duration=1,trim=end_frame={num_frames},setpts=PTS-STARTPTS'
    
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-t', '10',  # 输入选项：只读取前10秒，足够长
        '-i', input_path,
        '-vf', filter_chain,
        '-an',  # 去除音频
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'medium',
        '-frames:v', str(num_frames),  # 强制输出标记为指定帧数
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
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

def process_single_video(args):
    """
    处理单个视频的worker函数
    """
    video_path, output_dir, target_w, target_h, num_frames = args
    
    # 使用原文件名作为输出文件名
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    
    # 如果输出文件已存在，跳过
    if os.path.exists(output_path):
        return {'status': 'skip', 'video': video_path, 'output_path': output_path}
    
    # 处理视频
    success, error_msg = crop_video(video_path, output_path, target_w, target_h, num_frames)
    if success:
        return {'status': 'success', 'video': video_path, 'output_path': output_path}
    else:
        return {'status': 'fail', 'video': video_path, 'error': error_msg}

def main():
    # 创建输出目录
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # 读取jsonl文件
    print(f"正在读取jsonl文件: {input_jsonl}")
    data_list = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    
    print(f"共读取 {len(data_list)} 条数据")
    
    if len(data_list) == 0:
        print("没有找到数据！")
        return
    
    # 准备任务参数列表
    tasks = [
        (item['video_path'], output_video_dir, target_width, target_height, target_frames)
        for item in data_list
    ]
    
    print(f"使用 {num_processes} 个进程并行处理...")
    
    # 使用多进程处理
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_videos = []
    processed_count = 0
    results = []
    
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(total=len(tasks), desc="处理进度")
        
        for result in pool.imap_unordered(process_single_video, tasks, chunksize=1):
            processed_count += 1
            pbar.update(1)
            
            if result['status'] == 'success':
                success_count += 1
                results.append(result)
            elif result['status'] == 'skip':
                skip_count += 1
                results.append(result)
            elif result['status'] == 'fail':
                fail_count += 1
                failed_videos.append(result)
            
            pbar.set_postfix({
                '成功': success_count,
                '失败': fail_count,
                '跳过': skip_count
            })
        
        pbar.close()
    
    # 构建路径映射（原始路径 -> 新路径）
    path_mapping = {}
    for r in results:
        if r['status'] in ['success', 'skip']:
            path_mapping[r['video']] = r['output_path']
    
    # 更新jsonl中的路径
    print(f"\n正在更新jsonl文件...")
    updated_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data_list:
            orig_path = item['video_path']
            if orig_path in path_mapping:
                item['video_path'] = path_mapping[orig_path]
                updated_count += 1
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出结果
    print(f"\n处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总共处理: {processed_count}/{len(data_list)}")
    print(f"更新jsonl条目: {updated_count}")
    print(f"输出视频目录: {output_video_dir}")
    print(f"输出jsonl: {output_jsonl}")
    
    # 如果有失败的视频，保存到日志文件
    if failed_videos:
        error_log_path = os.path.join(output_video_dir, "failed_videos.log")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"失败视频列表 (共{fail_count}个)\n")
            f.write("=" * 80 + "\n\n")
            for item in failed_videos:
                f.write(f"视频: {item['video']}\n")
                if 'error' in item:
                    f.write(f"错误: {item['error']}\n")
                f.write("-" * 80 + "\n")
        print(f"\n失败视频列表已保存到: {error_log_path}")

if __name__ == "__main__":
    main()
