import os

def rename_folders(base_path, prefix):
    # 获取所有文件夹
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # 排序
    folders.sort()

    # 重命名
    for idx, folder in enumerate(folders, start=1):
        old_path = os.path.join(base_path, folder)
        new_name = f"{prefix}_{idx}"
        new_path = os.path.join(base_path, new_name)

        os.rename(old_path, new_path)
        print(f"{folder} -> {new_name}")

    print("重命名完成")


def main(base_path,prefix):
    rename_folders(base_path, prefix)


if __name__ == "__main__":
    
    base_path = r"D:\biaozhu\biaozhu_qingyu\0_biaozhu_ing"
    prefix = "DJI_20260414"

    main(base_path, prefix)