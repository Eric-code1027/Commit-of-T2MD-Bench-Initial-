import os
import csv
from pathlib import Path

# 输入输出路径
input_csv = "./train_datasets/wan_s2v/example_video_dataset/evan_metadata_s2v.csv"
output_csv = "./train_datasets/wan_s2v/example_video_dataset/evan_metadata_s2v_with_prompt.csv"

# 统一的prompt内容
default_prompt = "a person is speaking"

def add_prompt_to_csv(input_path, output_path, prompt):
    """为CSV文件添加prompt字段"""
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    # 读取原始CSV
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        if 'prompt' in fieldnames:
            print("警告: CSV文件中已存在prompt字段")
            user_input = input("是否覆盖现有prompt? (y/n): ")
            if user_input.lower() != 'y':
                print("操作已取消")
                return
        
        for row in reader:
            rows.append(row)
    
    print(f"读取了 {len(rows)} 行数据")
    
    # 添加prompt字段
    new_fieldnames = list(fieldnames)
    if 'prompt' not in new_fieldnames:
        new_fieldnames.append('prompt')
    
    # 写入新的CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        
        for row in rows:
            row['prompt'] = prompt
            writer.writerow(row)
    
    print(f"成功写入 {len(rows)} 行数据到: {output_path}")
    print(f"添加的prompt内容: '{prompt}'")
    
    # 显示前几行示例
    print("\n前3行示例:")
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"  {i+1}. video={row['video']}, audio={row['input_audio']}, prompt={row['prompt']}")

def main():
    print("=" * 80)
    print("为S2V数据集CSV添加prompt字段")
    print("=" * 80)
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    print(f"Prompt内容: '{default_prompt}'")
    print("=" * 80)
    
    add_prompt_to_csv(input_csv, output_csv, default_prompt)
    
    print("\n处理完成！")
    print(f"新的CSV文件保存在: {output_csv}")

if __name__ == "__main__":
    main()
