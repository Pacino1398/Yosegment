from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_TO_GROUP_ID = {
    "deliver_point": 0,
    "car": 1,
    "cover": 2,
    "road_sign": 3,
    "tree": 4,
    "person": 5,
    "forest": 6,
    "road": 7,
    "road_base": 8,
    "house": 9,
}
JSON_SUFFIX = ".json"


def fix_single_json(file_path: str | Path):
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        modified = False
        fix_log = []

        if "shapes" in data and isinstance(data["shapes"], list):
            check_items = data["shapes"]
        elif "annotations" in data and isinstance(data["annotations"], list):
            check_items = data["annotations"]
        else:
            check_items = [data]

        for idx, item in enumerate(check_items):
            label = item.get("label")
            if not label:
                continue

            if label not in LABEL_TO_GROUP_ID:
                fix_log.append(f"[对象{idx}] 未知label：{label}")
                continue

            correct_id = LABEL_TO_GROUP_ID[label]
            item["group_id"] = correct_id
            modified = True
            fix_log.append(f"[对象{idx}] → group_id = {correct_id}")

        if modified:
            with path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)

        return modified, fix_log
    except Exception as exc:
        return False, [f"错误：{exc}"]


def batch_fix_json(target_dir: str | Path):
    directory = Path(target_dir)
    if not directory.is_dir():
        print(f"文件夹不存在：{directory}")
        return 1

    total = 0
    fixed = 0
    log_list = []

    print("=" * 70)
    print("自动修正 group_id（根据 label 匹配）")
    print(f"目录：{directory}")
    print("=" * 70)

    for json_file in sorted(directory.glob(f"*{JSON_SUFFIX}")):
        total += 1
        is_modified, log = fix_single_json(json_file)
        if is_modified:
            line = f"✅ 已修正：{json_file.name} | {' | '.join(log)}"
            fixed += 1
        else:
            line = f"✔ 正常：{json_file.name} | {' | '.join(log)}"
        print(line)
        log_list.append(line)

    log_path = directory / "group_id修正日志.txt"
    log_path.write_text("\n".join(log_list), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"完成 | 总文件：{total} | 已修复：{fixed}")
    print(f"日志：{log_path}")
    print("=" * 70)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="根据 label 批量修正标注 JSON 的 group_id。")
    parser.add_argument("target_dir", type=Path, help="包含 JSON 文件的目录")
    return parser.parse_args()


def main():
    args = parse_args()
    raise SystemExit(batch_fix_json(args.target_dir))


if __name__ == "__main__":
    main()
