#!/usr/bin/env python3
"""解压 soul 文件夹下除 batch_1.tar 之外的所有 tar 文件（多进程，调用 tar 命令）
使用方式: INPUT_DIR=xxx python untar_soul_data.py
或: python untar_soul_data.py --input_dir xxx
"""

import argparse
import multiprocessing
import os
import subprocess
from pathlib import Path

EXCLUDE_TAR = "batch_1.tar"
NUM_WORKERS = 8  # 并行进程数


def extract_one(args):
    tar_path, extract_dir = args
    try:
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(extract_dir)],
            check=True,
            capture_output=True,
        )
        return tar_path.name, None
    except subprocess.CalledProcessError as e:
        return tar_path.name, (e.stderr.decode() if e.stderr else str(e))
    except Exception as e:
        return tar_path.name, str(e)


def main():
    parser = argparse.ArgumentParser(description="解压 soul 文件夹下的 tar 文件")
    parser.add_argument("--input_dir", "-i", type=str, default=os.environ.get("INPUT_DIR", "./soul_data"),
                        help="包含 tar 文件的目录，也可通过环境变量 INPUT_DIR 指定")
    args = parser.parse_args()
    input_dir = args.input_dir
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return

    tar_files = [f for f in input_path.iterdir() if f.suffix == ".tar" and f.name != EXCLUDE_TAR]
    tar_files.sort()

    if not tar_files:
        print(f"未找到需要解压的 tar 文件 (已排除 {EXCLUDE_TAR})")
        return

    print(f"共找到 {len(tar_files)} 个待解压的 tar 文件，使用 {NUM_WORKERS} 个进程并行解压")
    tasks = [(f, input_path) for f in tar_files]

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(extract_one, tasks)

    for name, err in results:
        if err is None:
            print(f"完成: {name}")
        else:
            print(f"失败: {name}, 错误: {err}")


if __name__ == "__main__":
    main()
