# pip install decord turbojpeg
import os
import time
from pathlib import Path
from decord import VideoReader, cpu
from turbojpeg import TurboJPEG
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

jpeg = TurboJPEG()

def save_img_turbo(path, frame):
    with open(path, 'wb') as f:
        f.write(jpeg.encode(frame, quality=90))

def process_video_god_mode(video_info):
    video_file, output_base, fps_target = video_info
    
    vr = VideoReader(str(video_file), ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    
    hop = round(video_fps / fps_target)
    indices = list(range(0, len(vr), hop))
    
    print(f"{video_file.name} | 待提取: {len(indices)} 帧")

    with ThreadPoolExecutor(max_workers=10) as io_executor:
        for i, frame_idx in enumerate(indices):
            # 获取帧（decord 的 get_batch 或直接索引非常快）
            frame = vr[frame_idx].asnumpy()
            # 转换为 BGR (decord 默认 RGB)
            frame = frame[:, :, ::-1]
            
            img_name = output_base / f"{video_file.stem}_{i+1:05d}.jpg"
            io_executor.submit(save_img_turbo, img_name, frame)

    return f" {video_file.name} 完成"

def main():
    input_path = Path(r"D:\biaozhu\biaozhu_qingyu\process") 
    output_base = Path(r"D:\biaozhu\biaozhu_qingyu\process_out") 
    output_base.mkdir(parents=True, exist_ok=True)

    video_tasks = [(f, output_base, 1) for f in input_path.iterdir() if f.suffix.lower() in {".mp4", ".mov"}]
    
    if not video_tasks:
        print("📂 文件夹里没找到视频，请检查路径！")
        return

    print(f"🚀 开始提取 {len(video_tasks)} 个视频...")
    start = time.time()

    with ProcessPoolExecutor(max_workers=6) as p_executor:
        results = list(p_executor.map(process_video_god_mode, video_tasks))
        
    for res in results:
        print(res)

    print(f"\n✨ 总耗时: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()  