import imageio
from PIL import Image
import numpy as np
from typing import List
from moviepy.editor import ImageSequenceClip

def read_video(video_path):
    return imageio.get_reader(video_path)

def extract_frames(video_path):
    """
    逐帧读取视频,返回PIL Image列表
    Args:
        video_path: 视频文件路径
    Returns:
        frames: PIL Image列表
    """
    reader = read_video(video_path)
    frames = []
    
    for frame in reader:
        # 将视频帧从numpy数组转换为PIL Image
        pil_frame = Image.fromarray(np.uint8(frame))
        frames.append(pil_frame)
        
    reader.close()
    return frames

def frames_with_angle(frames, first_frame_angle):
    """
    计算每帧的旋转角度，返回角度和帧的配对列表
    Args:
        frames: PIL Image列表
        first_frame_angle: 第一帧的起始角度
    Returns:
        frame_angle_pairs: 包含(angle, frame)元组的列表

    范围位0-360
    """
    frame_angle_pairs = []
    angle_step = 360 / len(frames)

    for i, frame in enumerate(frames):
        angle = first_frame_angle + i * angle_step  # 转换为整数角度

        if angle < 0:
            angle += 360
        if angle > 360:
            angle -= 360

        frame_angle_pairs.append((angle, frame))
        
    return frame_angle_pairs

def create_video_from_frames(frames: List[Image.Image], output_path: str, fps: int = 16, duration: float = 1.0):
    """
    将PIL Image列表转换为指定fps和时长的视频，如果帧数不足则循环播放
    Args:
        frames: PIL Image列表
        output_path: 输出视频路径
        fps: 目标帧率，默认16
        duration: 视频时长（秒），默认1.0
    """
    if not frames:
        raise ValueError("frames列表不能为空")
    
    # 计算需要的总帧数
    total_frames = int(fps * duration)
    
    # 准备帧数据
    frame_data = []
    for i in range(total_frames):
        # 使用取模运算循环使用帧
        frame_idx = i % len(frames)
        frame_data.append(np.array(frames[frame_idx]))
    
    # 写入视频
    imageio.mimsave(output_path, frame_data, fps=fps, quality=8)

def write_video(vclip, fps=25, save_path='res.mp4'):
    if isinstance(vclip, list):
        vclip = ImageSequenceClip(vclip, fps=fps)
    vclip.write_videofile(save_path, codec="libx264")