import glob
from safetensors import safe_open

def load_and_print_model_info(model_path_pattern, model_name):
    """加载模型并打印权重信息"""
    print(f"\n{'='*80}")
    print(f"模型: {model_name}")
    print(f"路径模式: {model_path_pattern}")
    print(f"{'='*80}\n")
    
    # 找到所有匹配的safetensors文件
    files = sorted(glob.glob(model_path_pattern))
    
    if not files:
        print(f"❌ 未找到匹配的文件！")
        return
    
    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {f}")
    print()
    
    # 收集所有权重信息
    all_weights = {}
    
    for file_path in files:
        print(f"正在加载: {file_path}")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_weights[key] = tensor.shape
        print(f"  ✓ 加载完成")
    
    print(f"\n总共 {len(all_weights)} 个权重:\n")
    print(f"{'权重名称':<80} {'Shape'}")
    print(f"{'-'*80} {'-'*30}")
    
    for name, shape in sorted(all_weights.items()):
        shape_str = str(tuple(shape))
        print(f"{name:<80} {shape_str}")
    
    return all_weights


if __name__ == "__main__":
    # 模型1: Wan2.2-S2V-14B
    s2v_14b_path = "./github_projects/Wan2.2/Wan2.2-S2V-14B/diffusion_pytorch_model-*.safetensors"
    s2v_weights = load_and_print_model_info(s2v_14b_path, "Wan2.2-S2V-14B")
    
    # 模型2: Wan2.2-TI2V-5B
    ti2v_5b_path = "./github_projects/Wan2.2/Wan2.2-TI2V-5B/diffusion_pytorch_model-*.safetensors"
    ti2v_weights = load_and_print_model_info(ti2v_5b_path, "Wan2.2-TI2V-5B")
    
    # 比较两个模型
    print(f"\n{'='*80}")
    print("模型对比")
    print(f"{'='*80}\n")
    
    if s2v_weights and ti2v_weights:
        s2v_keys = set(s2v_weights.keys())
        ti2v_keys = set(ti2v_weights.keys())
        
        common_keys = s2v_keys & ti2v_keys
        only_s2v = s2v_keys - ti2v_keys
        only_ti2v = ti2v_keys - s2v_keys
        
        print(f"共同权重数量: {len(common_keys)}")
        print(f"仅在 S2V-14B 中的权重: {len(only_s2v)}")
        print(f"仅在 TI2V-5B 中的权重: {len(only_ti2v)}")
        
        if only_s2v:
            print(f"\n仅在 S2V-14B 中的权重:")
            for key in sorted(only_s2v):
                print(f"  - {key}: {s2v_weights[key]}")
        
        if only_ti2v:
            print(f"\n仅在 TI2V-5B 中的权重:")
            for key in sorted(only_ti2v):
                print(f"  - {key}: {ti2v_weights[key]}")
        
        # 检查shape不同的权重
        shape_diff = []
        for key in common_keys:
            if s2v_weights[key] != ti2v_weights[key]:
                shape_diff.append(key)
        
        if shape_diff:
            print(f"\nShape不同的权重 ({len(shape_diff)}):")
            for key in sorted(shape_diff):
                print(f"  - {key}")
                print(f"    S2V-14B:  {s2v_weights[key]}")
                print(f"    TI2V-5B: {ti2v_weights[key]}")
