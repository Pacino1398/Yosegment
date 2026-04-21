from __future__ import annotations

from pathlib import Path


def rename_images_in_folder(
    folder: Path,
    sort_by="name"
):
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
        ".tiff", ".tif", ".raw", ".cr2", ".nef", ".heic"
    }

    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not images:
        return []

    # 排序
    if sort_by == "time":
        images.sort(key=lambda p: p.stat().st_mtime)
    elif sort_by == "size":
        images.sort(key=lambda p: p.stat().st_size)
    else:
        images.sort(key=lambda p: p.name.lower())

    folder_name = folder.name
    rename_plan = []

    for idx, img in enumerate(images, start=1):
        new_name = f"{folder_name}_{idx}{img.suffix.lower()}"
        rename_plan.append((img, folder / new_name))

    # 执行重命名
    for old_path, new_path in rename_plan:
        old_path.rename(new_path)
        print(f"{old_path.name} -> {new_path.name}")

    return rename_plan


def process_all_folders(base_path, sort_by="name"):
    base = Path(base_path).resolve()

    if not base.exists():
        print(f"路径不存在: {base}")
        return

    folders = [f for f in base.iterdir() if f.is_dir()]
    folders.sort(key=lambda x: x.name.lower())

    print(f"发现文件夹数量: {len(folders)}")
    print("=" * 60)

    total = 0

    for folder in folders:
        print(f"\n处理文件夹: {folder.name}")

        plan = rename_images_in_folder(
            folder,
            sort_by=sort_by
        )

        total += len(plan)

    print("=" * 60)
    print(f"全部完成，共处理 {total} 张图片")


def main(base_path):

    process_all_folders(
        base_path,
        sort_by="name"
    )


if __name__ == "__main__":
    base_path = r"D:\biaozhu\biaozhu_qingyu\0_biaozhu_ing"
    main(base_path)