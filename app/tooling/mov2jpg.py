import cv2
import os
from pathlib import Path

def extract_frames_opencv(fps_target=1):
    # 源视频路径
    input_path = Path(r"D:\biaozhu\biaozhu_qingyu\process")
    # 统一输出路径
    output_base = Path(r"D:\biaozhu\biaozhu_qingyu\process_out")
    
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        return

    # 创建统一的输出根目录
    output_base.mkdir(parents=True, exist_ok=True)

    for video_file in input_path.iterdir():
        if video_file.suffix.lower() in video_extensions:
            # 获取视频名（例如 260414_01），作为图片命名的前缀
            video_name_stem = video_file.stem
            
            # 打开视频文件
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"❌ 无法打开视频: {video_file.name}")
                continue

            # 获取视频自带的帧率
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            # 计算每隔多少帧提取一次
            hop = round(video_fps / fps_target)
            if hop < 1: hop = 1
            
            print(f"⏳ 正在处理: {video_file.name} (原始FPS: {video_fps}, 提取间隔: {hop}帧)")

            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 只保存符合间隔的帧
                if frame_count % hop == 0:
                    saved_count += 1
                    
                    img_name = output_base / f"{video_name_stem}_{saved_count:05d}.jpg"
                    
                    # 保存图片，100代表最高质量
                    cv2.imwrite(str(img_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                frame_count += 1
            
            cap.release()
            print(f"✅ {video_file.name} 完成！提取了 {saved_count} 张图片")

if __name__ == "__main__":
    extract_frames_opencv()