import subprocess
import os
from pathlib import Path

def resize_and_center_crop(input_video, output_video, target_width=448, target_height=832):
    """
    对视频进行等比拉伸和中心裁剪（使用ffmpeg）
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        target_width: 目标宽度
        target_height: 目标高度
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # 使用ffmpeg的scale和crop滤镜
    # scale=-1:'ih*max(448/iw,832/ih)' 表示等比拉伸，使得缩放后的尺寸能够覆盖目标区域
    # crop=448:832 表示从中心裁剪出448x832的区域
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f"scale='iw*max({target_width}/iw,{target_height}/ih)':'ih*max({target_width}/iw,{target_height}/ih)',crop={target_width}:{target_height}",
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',  # 覆盖输出文件
        output_video
    ]
    
    print(f"执行命令: {' '.join(ffmpeg_cmd)}")
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print(f"目标尺寸: {target_width}x{target_height}")
    
    try:
        # 执行ffmpeg命令
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print(f"处理完成！输出视频: {output_video}")
        else:
            print(f"错误: ffmpeg执行失败")
            print(f"错误信息: {result.stderr}")
            
    except FileNotFoundError:
        print("错误: 找不到ffmpeg命令，请确保已安装ffmpeg")
    except Exception as e:
        print(f"错误: {str(e)}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="对视频进行等比拉伸和中心裁剪")
    parser.add_argument("--input", "-i", type=str, default=os.environ.get("INPUT_VIDEO", "input.mp4"),
                        help="输入视频路径，也可通过环境变量 INPUT_VIDEO 指定")
    parser.add_argument("--output", "-o", type=str, default=os.environ.get("OUTPUT_VIDEO", "output_448x832.mp4"),
                        help="输出视频路径，也可通过环境变量 OUTPUT_VIDEO 指定")
    parser.add_argument("--width", type=int, default=448, help="目标宽度")
    parser.add_argument("--height", type=int, default=832, help="目标高度")
    args = parser.parse_args()
    resize_and_center_crop(args.input, args.output, target_width=args.width, target_height=args.height)
