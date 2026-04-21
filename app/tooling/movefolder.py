import os
import shutil

# 1. 定义源路径和目标路径
src_base = r'D:\biaozhu\biaozhu_qingyu\video_all'
dst_base = r'D:\biaozhu\biaozhu_qingyu\process'

# 2. 指定需要提取视频的文件夹列表
target_folders = ['260414', '260415']

# 3. 指定视频后缀（可根据实际情况增删）
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

def flatten_export():
    # 确保目标根目录存在
    if not os.path.exists(dst_base):
        os.makedirs(dst_base)
        print(f"创建目标目录: {dst_base}")

    for folder_name in target_folders:
        folder_path = os.path.join(src_base, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"跳过: 找不到文件夹 {folder_path}")
            continue

        print(f"正在处理文件夹: {folder_name}...")
        
        # 遍历子文件夹内的所有文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查是否为视频文件
                if file.lower().endswith(video_extensions):
                    src_file_path = os.path.join(root, file)
                    dst_file_path = os.path.join(dst_base, file)
                    
                    # 如果文件名重复，自动重命名防止覆盖
                    if os.path.exists(dst_file_path):
                        name, ext = os.path.splitext(file)
                        dst_file_path = os.path.join(dst_base, f"{name}_{folder_name}{ext}")

                    try:
                        shutil.copy2(src_file_path, dst_file_path)
                        print(f"  已导出: {file}")
                    except Exception as e:
                        print(f"  导出 {file} 失败: {e}")

if __name__ == "__main__":
    flatten_export()
    print("\n所有指定文件夹内的视频已导出到 process 根目录。")