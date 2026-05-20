import torch
from safetensors.torch import load_file

ckpt_a_path = "./released_models/wan_s2v/si2v_5b_stage2/step-18000.safetensors"
ckpt_b_path = "./github_projects/from_modelscope/DiffSynth-Studio/models/train/Evan-Wan2.2-S2V-DMD-5B_full/step-1000.safetensors"

print("Loading ckptA ...")
state_a = load_file(ckpt_a_path, device="cpu")
print(f"Loading ckptB ...")
state_b = load_file(ckpt_b_path, device="cpu")

keys_a = set(state_a.keys())
keys_b = set(state_b.keys())
common_keys = sorted(keys_a & keys_b)
only_a = sorted(keys_a - keys_b)
only_b = sorted(keys_b - keys_a)

print("\n" + "=" * 80)
print("基本信息")
print("=" * 80)
print(f"ckptA key 数量: {len(keys_a)}")
print(f"ckptB key 数量: {len(keys_b)}")
print(f"共有 key 数量:  {len(common_keys)}")
print(f"仅在 A 中的 key: {len(only_a)}")
print(f"仅在 B 中的 key: {len(only_b)}")

if only_a:
    print("\n--- 仅在 A 中存在的 key ---")
    for k in only_a:
        print(f"  {k}  shape={state_a[k].shape}")

if only_b:
    print("\n--- 仅在 B 中存在的 key ---")
    for k in only_b:
        print(f"  {k}  shape={state_b[k].shape}")

print("\n" + "=" * 80)
print("逐层权重差异分析 (共有 key)")
print("=" * 80)

unchanged_keys = []
changed_records = []

for key in common_keys:
    wa = state_a[key].float()
    wb = state_b[key].float()

    if wa.shape != wb.shape:
        print(f"[SHAPE MISMATCH] {key}: A={wa.shape} vs B={wb.shape}")
        continue

    diff = wb - wa
    l2_diff = torch.norm(diff).item()
    max_abs_diff = torch.max(torch.abs(diff)).item()
    mean_diff = torch.mean(diff).item()
    l2_a = torch.norm(wa).item()
    relative_change = l2_diff / l2_a if l2_a > 0 else float("inf") if l2_diff > 0 else 0.0

    if l2_diff == 0:
        unchanged_keys.append(key)
    else:
        changed_records.append({
            "key": key,
            "l2_diff": l2_diff,
            "max_abs_diff": max_abs_diff,
            "mean_diff": mean_diff,
            "l2_a": l2_a,
            "relative_change": relative_change,
            "shape": tuple(wa.shape),
            "numel": wa.numel(),
        })

print(f"\n未变化层数量: {len(unchanged_keys)}")
print(f"已变化层数量: {len(changed_records)}")
print(f"总共有 key:   {len(common_keys)}")

if unchanged_keys:
    print(f"\n--- 未变化的层 ({len(unchanged_keys)}) ---")
    for k in unchanged_keys:
        print(f"  {k}")

if changed_records:
    sorted_by_rel_desc = sorted(changed_records, key=lambda x: x["relative_change"], reverse=True)
    sorted_by_rel_asc = sorted(changed_records, key=lambda x: x["relative_change"])

    n_top = min(20, len(changed_records))

    print(f"\n--- Top-{n_top} 相对变化率最大的层 ---")
    print(f"{'Layer':<80s} {'RelChange':>12s} {'L2(diff)':>12s} {'MaxAbsDiff':>12s} {'MeanDiff':>12s} {'L2(A)':>12s} {'Shape':>20s}")
    print("-" * 160)
    for rec in sorted_by_rel_desc[:n_top]:
        print(f"{rec['key']:<80s} {rec['relative_change']:>12.6f} {rec['l2_diff']:>12.6f} {rec['max_abs_diff']:>12.6f} {rec['mean_diff']:>12.6f} {rec['l2_a']:>12.6f} {str(rec['shape']):>20s}")

    print(f"\n--- Top-{n_top} 相对变化率最小的层 (非零) ---")
    print(f"{'Layer':<80s} {'RelChange':>12s} {'L2(diff)':>12s} {'MaxAbsDiff':>12s} {'MeanDiff':>12s} {'L2(A)':>12s} {'Shape':>20s}")
    print("-" * 160)
    for rec in sorted_by_rel_asc[:n_top]:
        print(f"{rec['key']:<80s} {rec['relative_change']:>12.6f} {rec['l2_diff']:>12.6f} {rec['max_abs_diff']:>12.6f} {rec['mean_diff']:>12.6f} {rec['l2_a']:>12.6f} {str(rec['shape']):>20s}")

    avg_rel = sum(r["relative_change"] for r in changed_records) / len(changed_records)
    print(f"\n--- 总体统计 ---")
    print(f"已变化层的平均相对变化率: {avg_rel:.6f}")
    print(f"已变化层的最大相对变化率: {sorted_by_rel_desc[0]['relative_change']:.6f} ({sorted_by_rel_desc[0]['key']})")
    print(f"已变化层的最小相对变化率: {sorted_by_rel_asc[0]['relative_change']:.6f} ({sorted_by_rel_asc[0]['key']})")
