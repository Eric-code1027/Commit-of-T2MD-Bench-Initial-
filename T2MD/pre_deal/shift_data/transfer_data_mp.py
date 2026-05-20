#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程文件拷贝脚本
将源目录下的所有文件拷贝到目标目录，保留子目录结构
使用方式: SOURCE_DIR=xxx TARGET_DIR=xxx python transfer_data_mp.py
或: python transfer_data_mp.py --source_dir xxx --target_dir xxx
"""

import os
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# 进程数量
NUM_PROCESSES = 64


def copy_file(args):
    """
    拷贝单个文件
    
    Args:
        args: (源文件路径, 目标文件路径)
    
    Returns:
        tuple: (是否成功, 文件路径, 错误信息)
    """
    src_file, dst_file = args
    
    try:
        # 创建目标目录
        dst_dir = os.path.dirname(dst_file)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 拷贝文件
        shutil.copy2(src_file, dst_file)
        
        return True, src_file, None
    except Exception as e:
        return False, src_file, str(e)


def get_all_files(source_dir):
    """
    获取源目录下的所有文件
    
    Args:
        source_dir: 源目录路径
    
    Returns:
        list: 所有文件的路径列表
    """
    file_list = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            file_list.append(src_file_path)
    
    return file_list


def prepare_copy_tasks(source_dir, target_dir):
    """
    准备拷贝任务列表
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    
    Returns:
        list: [(源文件路径, 目标文件路径), ...]
    """
    tasks = []
    
    # 获取所有文件
    all_files = get_all_files(source_dir)
    
    print(f"找到 {len(all_files)} 个文件需要拷贝")
    
    # 准备拷贝任务
    for src_file in all_files:
        # 计算相对路径
        rel_path = os.path.relpath(src_file, source_dir)
        
        # 目标文件路径
        dst_file = os.path.join(target_dir, rel_path)
        
        tasks.append((src_file, dst_file))
    
    return tasks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多进程文件拷贝脚本")
    parser.add_argument("--source_dir", "-s", type=str, default=os.environ.get("SOURCE_DIR", "."),
                        help="源目录路径，也可通过环境变量 SOURCE_DIR 指定")
    parser.add_argument("--target_dir", "-t", type=str, default=os.environ.get("TARGET_DIR", "./output"),
                        help="目标目录路径，也可通过环境变量 TARGET_DIR 指定")
    args = parser.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir

    print("=" * 80)
    print("多进程文件拷贝脚本")
    print("=" * 80)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"进程数: {NUM_PROCESSES}")
    print("=" * 80)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 准备拷贝任务
    print("\n正在扫描文件...")
    tasks = prepare_copy_tasks(source_dir, target_dir)
    
    if not tasks:
        print("没有找到需要拷贝的文件")
        return
    
    # 多进程拷贝
    print(f"\n开始拷贝，使用 {NUM_PROCESSES} 个进程...")
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(copy_file, tasks),
            total=len(tasks),
            desc="拷贝进度",
            unit="文件"
        ))
    
    # 统计结果
    for success, file_path, error in results:
        if success:
            success_count += 1
        else:
            fail_count += 1
            failed_files.append((file_path, error))
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("拷贝完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print("=" * 80)
    
    # 输出失败的文件
    if failed_files:
        print("\n失败的文件列表:")
        for file_path, error in failed_files:
            print(f"  - {file_path}")
            print(f"    错误: {error}")


if __name__ == "__main__":
    main()
