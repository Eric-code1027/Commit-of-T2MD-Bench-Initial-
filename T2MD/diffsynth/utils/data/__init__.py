"""Re-export VideoData and save_video_with_audio for backward compatibility with diffsynth.utils.data"""
from ...data import VideoData, save_video, save_frames, merge_video_audio, save_video_with_audio

__all__ = ["VideoData", "save_video", "save_frames", "merge_video_audio", "save_video_with_audio"]
