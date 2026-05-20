#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 safetensors 权重，使用方式: python check_s2v_ckpt.py [--ckpt_path xxx] [--log_path xxx]"""

from safetensors.torch import load_file
import os
import argparse
from datetime import datetime

def check_safetensors_weights(ckpt_path=None, log_path=None):
    """
    读取 safetensors 文件并打印所有权重的名字和 shape
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    # 默认路径使用相对路径
    if ckpt_path is None:
        ckpt_path = os.path.join(project_root, "models", "train", "Evan-Wan2.2-S2V-5B_full", "step-1000.safetensors")
    if log_path is None:
        log_path = os.path.join(script_dir, "step-1000_weights_info.log")
    
    print(f"开始读取 checkpoint: {ckpt_path}")
    
    # 加载 safetensors 文件
    state_dict = load_file(ckpt_path)
    
    # 准备日志内容
    log_lines = []
    log_lines.append("=" * 100)
    log_lines.append(f"Checkpoint 权重信息")
    log_lines.append(f"文件路径: {ckpt_path}")
    log_lines.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 100)
    log_lines.append(f"\n总共权重数量: {len(state_dict)}\n")
    log_lines.append("=" * 100)
    log_lines.append(f"{'序号':<8} {'权重名称':<80} {'Shape':<40}")
    log_lines.append("=" * 100)
    
    # 遍历所有权重
    total_params = 0
    for idx, (name, tensor) in enumerate(sorted(state_dict.items()), 1):
        shape_str = str(tuple(tensor.shape))
        num_params = tensor.numel()
        total_params += num_params
        
        log_line = f"{idx:<8} {name:<80} {shape_str:<40}"
        log_lines.append(log_line)
        print(log_line)
    
    # 添加统计信息
    log_lines.append("=" * 100)
    log_lines.append(f"\n统计信息:")
    log_lines.append(f"  - 总权重数量: {len(state_dict)}")
    log_lines.append(f"  - 总参数量: {total_params:,}")
    log_lines.append(f"  - 总参数量 (M): {total_params / 1e6:.2f}M")
    log_lines.append(f"  - 总参数量 (B): {total_params / 1e9:.4f}B")
    log_lines.append("=" * 100)
    
    # 写入日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    print(f"\n日志已保存到: {log_path}")
    print(f"总权重数量: {len(state_dict)}")
    print(f"总参数量: {total_params:,} ({total_params / 1e9:.4f}B)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 safetensors  checkpoint 权重")
    parser.add_argument("--ckpt_path", "-c", type=str, default=os.environ.get("CKPT_PATH"),
                        help="checkpoint 文件路径，也可通过环境变量 CKPT_PATH 指定")
    parser.add_argument("--log_path", "-l", type=str, default=os.environ.get("LOG_PATH"),
                        help="输出日志路径，也可通过环境变量 LOG_PATH 指定")
    args = parser.parse_args()
    check_safetensors_weights(ckpt_path=args.ckpt_path, log_path=args.log_path)
