#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Arrow文件读取COS视频URL并批量下载
支持多进程并行下载，输出jsonl元数据
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from multiprocessing import Pool, Lock
from tqdm import tqdm
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import Dict, Any, Tuple
import time
import traceback


# ==================== 配置参数 ====================
ARROW_DIR = "./video2audio/leo_21m/all_packed_arrow_leo"
OUTPUT_VIDEO_DIR = "./train_datasets/ovi/t2va_distill"
OUTPUT_JSONL = "./train_datasets/ovi/t2va_distill/metadata.jsonl"
FAILED_LOG = "./train_datasets/ovi/t2va_distill/failed_downloads.log"
MAX_DOWNLOADS = 100000  # 10万条
NUM_PROCESSES = 64
DOWNLOAD_TIMEOUT = 30  # 秒
MAX_RETRIES = 3
CHUNK_SIZE = 8192  # 下载块大小


# ==================== 辅助函数 ====================
def explore_arrow_schema(arrow_dir: str, num_samples: int = 5):
    """
    探索Arrow文件结构，打印schema和前几行数据
    
    Args:
        arrow_dir: Arrow文件目录
        num_samples: 打印的样本数量
    """
    print("=" * 80)
    print("探索Arrow文件结构")
    print("=" * 80)
    
    # 查找第一个arrow文件
    arrow_files = []
    for root, dirs, files in os.walk(arrow_dir):
        for file in files:
            if file.endswith('.arrow'):
                arrow_files.append(os.path.join(root, file))
                if len(arrow_files) >= 3:  # 只看前3个文件
                    break
        if len(arrow_files) >= 3:
            break
    
    if not arrow_files:
        print(f"错误: 在 {arrow_dir} 中未找到arrow文件")
        return None
    
    print(f"\n找到 {len(arrow_files)} 个arrow文件（显示前3个）:")
    for f in arrow_files[:3]:
        print(f"  - {f}")
    
    # 读取第一个文件
    first_file = arrow_files[0]
    print(f"\n正在读取文件: {first_file}")
    
    try:
        # 使用正确的方法读取arrow文件
        table = pa.memory_map(first_file, "r")
        table = pa.ipc.RecordBatchFileReader(table).read_all()
        
        print(f"\n文件类型: Arrow")
        print(f"总行数: {len(table)}")
        print(f"\nSchema:")
        print(table.schema)
        print(f"\n列名: {table.schema.names}")
        
        print(f"\n前 {min(num_samples, len(table))} 行数据:")
        # 直接从arrow table读取，不转换为pandas
        for i in range(min(num_samples, len(table))):
            print(f"\n--- 样本 {i+1} ---")
            for col_name in table.schema.names:
                try:
                    value = table[col_name][i].as_py()
                    if isinstance(value, str) and len(value) > 100:
                        print(f"{col_name}: {value[:100]}...")
                    elif isinstance(value, (list, dict)) and len(str(value)) > 100:
                        print(f"{col_name}: {str(value)[:100]}...")
                    else:
                        print(f"{col_name}: {value}")
                except Exception as e:
                    print(f"{col_name}: <读取错误: {e}>")
        
        return table
        
    except Exception as e:
        print(f"读取失败: {e}")
        traceback.print_exc()
        return None


def generate_video_filename(url: str, index: int, default_ext: str = ".mp4") -> str:
    """
    根据URL生成唯一的视频文件名
    
    Args:
        url: 视频URL
        index: 数据索引
        default_ext: 默认扩展名
    
    Returns:
        文件名
    """
    # 从URL提取扩展名
    ext = default_ext
    if url:
        url_parts = url.split('?')[0]  # 去除查询参数
        if '.' in url_parts:
            ext = '.' + url_parts.split('.')[-1].lower()
            # 确保是常见视频格式
            if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']:
                ext = default_ext
    
    # 使用URL的MD5作为文件名，确保唯一性
    if url:
        hash_str = hashlib.md5(url.encode()).hexdigest()
    else:
        hash_str = f"video_{index:08d}"
    
    return f"{hash_str}{ext}"


def download_single_video(args: Tuple) -> Dict[str, Any]:
    """
    下载单个视频文件
    
    Args:
        args: (data_dict, index, output_dir)
            data_dict: 包含video_cos_url等字段的字典
            index: 数据索引
            output_dir: 输出目录
    
    Returns:
        result字典，包含status, data, video_local_path等信息
    """
    data_dict, index, output_dir = args
    
    try:
        # 获取视频URL
        video_url = data_dict.get('video_cos_url', '')
        if not video_url:
            return {
                'status': 'fail',
                'index': index,
                'error': 'video_cos_url字段为空',
                'data': data_dict
            }
        
        # 生成本地文件名和路径
        filename = generate_video_filename(video_url, index)
        video_path = os.path.join(output_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            # 文件已存在，添加本地路径字段
            result_data = data_dict.copy()
            result_data['video_local_path'] = video_path
            return {
                'status': 'skip',
                'index': index,
                'video_path': video_path,
                'data': result_data
            }
        
        # 下载视频（带重试）
        for retry in range(MAX_RETRIES):
            try:
                response = requests.get(
                    video_url, 
                    stream=True, 
                    timeout=DOWNLOAD_TIMEOUT
                )
                response.raise_for_status()
                
                # 创建临时文件
                temp_path = video_path + '.tmp'
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                # 写入文件
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                
                # 重命名为正式文件
                os.rename(temp_path, video_path)
                
                # 下载成功，添加本地路径字段
                result_data = data_dict.copy()
                result_data['video_local_path'] = video_path
                
                return {
                    'status': 'success',
                    'index': index,
                    'video_path': video_path,
                    'data': result_data
                }
                
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    time.sleep(1)  # 重试前等待1秒
                    continue
                else:
                    # 所有重试都失败
                    # 清理临时文件
                    temp_path = video_path + '.tmp'
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    
                    return {
                        'status': 'fail',
                        'index': index,
                        'error': f'下载失败(重试{MAX_RETRIES}次): {str(e)}',
                        'url': video_url,
                        'data': data_dict
                    }
    
    except Exception as e:
        return {
            'status': 'fail',
            'index': index,
            'error': f'处理异常: {str(e)}',
            'data': data_dict
        }


def load_arrow_data(arrow_dir: str, max_items: int) -> list:
    """
    从Arrow目录加载数据
    
    Args:
        arrow_dir: Arrow文件目录
        max_items: 最大加载数量
    
    Returns:
        数据列表，每项为字典
    """
    print("\n" + "=" * 80)
    print("加载Arrow数据")
    print("=" * 80)
    
    # 查找所有arrow文件
    arrow_files = []
    for root, dirs, files in os.walk(arrow_dir):
        for file in files:
            if file.endswith('.arrow'):
                arrow_files.append(os.path.join(root, file))
    
    if not arrow_files:
        print(f"错误: 在 {arrow_dir} 中未找到arrow文件")
        return []
    
    # 排序确保顺序一致
    arrow_files.sort()
    print(f"找到 {len(arrow_files)} 个arrow文件")
    
    # 逐个读取文件并收集数据
    all_data = []
    for arrow_file in arrow_files:
        if len(all_data) >= max_items:
            break
        
        try:
            # 使用正确的方法读取arrow文件
            table = pa.memory_map(arrow_file, "r")
            table = pa.ipc.RecordBatchFileReader(table).read_all()
            
            # 获取列名
            column_names = table.schema.names
            
            # 直接从arrow table提取数据，不转换为pandas
            for i in range(len(table)):
                if len(all_data) >= max_items:
                    break
                
                row_dict = {}
                for col_name in column_names:
                    try:
                        value = table[col_name][i].as_py()
                        row_dict[col_name] = value
                    except Exception as e:
                        # 如果某个字段读取失败，设置为None
                        row_dict[col_name] = None
                        print(f"  警告: 读取字段 {col_name} 失败 (行{i}): {e}")
                
                all_data.append(row_dict)
            
            print(f"  已加载 {len(all_data)}/{max_items} 条数据 (文件: {os.path.basename(arrow_file)})")
            
        except Exception as e:
            print(f"  警告: 读取文件 {arrow_file} 失败: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n总共加载 {len(all_data)} 条数据")
    return all_data


def write_jsonl_line(filepath: str, data: dict, lock=None):
    """
    追加一行到jsonl文件（线程安全）
    
    Args:
        filepath: jsonl文件路径
        data: 数据字典
        lock: 多进程锁（可选）
    """
    json_line = json.dumps(data, ensure_ascii=False) + '\n'
    
    if lock:
        with lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json_line)
    else:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json_line)


def main():
    """主函数"""
    print("=" * 80)
    print("COS视频批量下载脚本")
    print("=" * 80)
    print(f"Arrow目录: {ARROW_DIR}")
    print(f"输出目录: {OUTPUT_VIDEO_DIR}")
    print(f"元数据文件: {OUTPUT_JSONL}")
    print(f"最大下载数: {MAX_DOWNLOADS}")
    print(f"进程数: {NUM_PROCESSES}")
    print("=" * 80)
    
    # 步骤1: 探索Arrow文件结构
    print("\n【步骤1】探索Arrow文件结构...")
    explore_arrow_schema(ARROW_DIR, num_samples=3)
    
    # 创建输出目录
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
    # 清空或创建输出文件
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        pass  # 清空文件
    
    # 步骤2: 加载Arrow数据
    print("\n【步骤2】加载Arrow数据...")
    data_list = load_arrow_data(ARROW_DIR, MAX_DOWNLOADS)
    
    if not data_list:
        print("错误: 没有加载到数据")
        return
    
    # 步骤3: 准备下载任务
    print("\n【步骤3】准备下载任务...")
    tasks = [
        (data_dict, idx, OUTPUT_VIDEO_DIR)
        for idx, data_dict in enumerate(data_list)
    ]
    print(f"准备了 {len(tasks)} 个下载任务")
    
    # 步骤4: 多进程下载
    print(f"\n【步骤4】开始多进程下载（{NUM_PROCESSES}进程）...")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_items = []
    
    with Pool(processes=NUM_PROCESSES) as pool:
        # 使用imap实现流式处理
        with tqdm(total=len(tasks), desc="下载进度", unit="个") as pbar:
            for result in pool.imap(download_single_video, tasks, chunksize=1):
                status = result['status']
                
                if status == 'success':
                    success_count += 1
                    # 写入jsonl
                    write_jsonl_line(OUTPUT_JSONL, result['data'])
                    
                elif status == 'skip':
                    skip_count += 1
                    # 写入jsonl
                    write_jsonl_line(OUTPUT_JSONL, result['data'])
                    
                elif status == 'fail':
                    fail_count += 1
                    failed_items.append(result)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '跳过': skip_count,
                    '失败': fail_count
                })
    
    # 步骤5: 输出统计信息
    print("\n" + "=" * 80)
    print("下载完成!")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个（文件已存在）")
    print(f"失败: {fail_count} 个")
    print(f"总计: {len(tasks)} 个")
    print("=" * 80)
    
    print(f"\n元数据文件: {OUTPUT_JSONL}")
    print(f"视频目录: {OUTPUT_VIDEO_DIR}")
    
    # 保存失败记录
    if failed_items:
        print(f"\n正在保存失败记录到: {FAILED_LOG}")
        with open(FAILED_LOG, 'w', encoding='utf-8') as f:
            for item in failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已保存 {len(failed_items)} 条失败记录")


if __name__ == "__main__":
    main()
