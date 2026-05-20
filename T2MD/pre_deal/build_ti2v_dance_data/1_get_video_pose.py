import os
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import av

# 模型路径：可通过环境变量 POSE_MODEL_PATH 或 --model_path 指定
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
_default_model_dir = os.path.join(_project_root, "models", "mediapipe")
model_path = os.environ.get("POSE_MODEL_PATH", os.path.join(_default_model_dir, "pose_landmarker.task"))
os.makedirs(os.path.dirname(model_path), exist_ok=True)
if not os.path.exists(model_path):
    print("Downloading pose_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded to {model_path}")

# 创建 PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = PoseLandmarker.create_from_options(options)

# 输入输出视频路径：可通过环境变量或命令行参数指定
_default_input = os.path.join(_project_root, "datasets", "dance_from_xiaoda", "cropped_832x448_video", "000005_s000_no_vocals.mp4")
_default_output = os.path.join(_project_root, "datasets", "dance_from_xiaoda", "cropped_832x448_video_pose", "000005_s000_no_vocals.mp4")
parser = argparse.ArgumentParser(description="为视频添加姿态关键点")
parser.add_argument("--input", "-i", default=os.environ.get("INPUT_VIDEO", _default_input), help="输入视频路径")
parser.add_argument("--output", "-o", default=os.environ.get("OUTPUT_VIDEO", _default_output), help="输出视频路径")
args = parser.parse_args()
input_video = args.input
output_video = args.output
os.makedirs(os.path.dirname(output_video), exist_ok=True)

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 使用 av 库输出视频（H.264 编码，兼容性好）
container = av.open(output_video, mode='w')
stream = container.add_stream('libx264', rate=fps)
stream.width = width
stream.height = height
stream.pix_fmt = 'yuv420p'
stream.codec_context.thread_count = 4

# PoseLandmarker 的关键点连接关系 (33 个关键点)
POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (7, 11), (11, 12), (12, 13),
    (7, 14), (14, 15), (15, 16),
    (11, 23), (23, 24), (24, 12),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)
]

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 转换为 MediaPipe Image 格式
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 处理帧 (VIDEO 模式需要传入 timestamp_ms)
    timestamp_ms = int(frame_idx * 1000 / fps)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # 绘制骨架
    if results.pose_landmarks:
        for pose_landmarks in results.pose_landmarks:
            # 绘制关键点
            for landmark in pose_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # 绘制连接线
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]
                
                start_x = int(start_point.x * width)
                start_y = int(start_point.y * height)
                end_x = int(end_point.x * width)
                end_y = int(end_point.y * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    # 转换为 RGB 写入 av 容器
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
    for packet in stream.encode(av_frame):
        container.mux(packet)
    
    frame_idx += 1

# 刷新编码器
for packet in stream.encode():
    container.mux(packet)

cap.release()
container.close()
landmarker.close()
print(f"Done! Output saved to {output_video}")
