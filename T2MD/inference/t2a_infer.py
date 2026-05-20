# torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
import os
import sys

# 添加项目根目录到 Python 路径，以便导入 ovi 模块
# 执行地址是 ./AudioSummary/DiffSynth-Studio
# 代码文件在 inference/t2a_infer.py，需要将上级目录（DiffSynth-Studio）添加到路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录（inference/）
project_root = os.path.dirname(script_dir)  # 获取项目根目录（DiffSynth-Studio/）
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 将项目根目录添加到 Python 路径的最前面

import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import copy
from ovi.utils.io_utils import save_audio
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_audio_engine import OviAudioEngine



def _init_logging(rank):
    # 初始化日志
    if rank == 0:
        # 设置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args): 

    world_size = get_world_size()  # 获取总GPU数量
    global_rank = get_global_rank()  # 获取当前GPU的全局排名
    local_rank = get_local_rank()  # 获取当前GPU的本地排名
    device = local_rank
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的CUDA设备
    sp_size = config.get("sp_size", 1)  # 从配置获取序列并行大小，默认值为1
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)  # 初始化日志系统
    if global_rank == 0:
        ovi_ckpt = config.get("ovi_ckpt", None)  # 从配置获取检查点路径
        logging.info(f"OVI fusion checkpoint (config.ovi_ckpt): {ovi_ckpt}")
        if ovi_ckpt:
            assert os.path.exists(ovi_ckpt), f"ovi_ckpt does not exist: {ovi_ckpt}"  # 检查检查点文件是否存在

    if world_size > 1:
        torch.distributed.init_process_group(  # 初始化分布式进程组，用于多GPU通信
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: 断言不共享 t5 等...


    initialize_sequence_parallel_state(sp_size)  # 初始化序列并行状态和通信组
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")  # 获取序列并行状态标志
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # 在加载模型之前验证输入，如果输入无效则不会浪费时间
    # OviAudioEngine 只支持 text-to-audio，不需要图像输入
    text_prompt = config.get("text_prompt")  # 从配置获取文本提示词（可以是字符串或CSV文件路径）
    # 使用 t2v 模式来验证和处理文本提示词（因为 t2a 本质上也是 text-to-x）
    text_prompts, _ = validate_and_process_user_prompt(text_prompt, image_path=None, mode="t2v")  # 验证并处理用户输入，支持单个文本或CSV批量输入

    # 支持在一次运行中比较多个融合检查点
    compare_ckpts = config.get("compare_ovi_ckpts", None)  # 获取要比较的检查点列表
    if compare_ckpts is None:
        ckpt_list = [config.get("ovi_ckpt", None)]  # 如果没有指定，使用单个检查点
    else:
        # 接受列表/元组或逗号分隔的字符串
        if isinstance(compare_ckpts, str):  # 如果是字符串，按逗号分割
            ckpt_list = [s.strip() for s in compare_ckpts.split(",") if s.strip()]  # 分割字符串并去除空白
        else:
            ckpt_list = list(compare_ckpts)  # 转换为列表
    ckpt_list = [c for c in ckpt_list if c]  # 过滤空值
    if not ckpt_list:
        ckpt_list = [config.get("ovi_ckpt", None)]  # 如果列表为空，使用默认检查点
    
    output_dir = config.get("output_dir", "./outputs")  # 获取输出目录路径
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

    # 加载 CSV 数据（仅文本提示词，无图像）
    all_eval_data = text_prompts  # OviAudioEngine 只需要文本提示词列表
    max_examples = config.get("max_examples", None)  # 获取最大样本数限制
    if max_examples is not None:
        try:
            max_examples = int(max_examples)  # 转换为整数
            if max_examples > 0:
                all_eval_data = all_eval_data[:max_examples]  # 限制数据量
        except Exception:
            pass

    # 获取序列并行（SP）配置
    use_sp = get_sequence_parallel_state()  # 检查是否启用序列并行
    if use_sp:
        sp_size = nccl_info.sp_size  # 获取序列并行组大小
        sp_rank = nccl_info.rank_within_group  # 获取组内排名
        sp_group_id = global_rank // sp_size  # 计算当前GPU所属的SP组ID
        num_sp_groups = world_size // sp_size  # 计算SP组的总数
    else:
        # 无 SP：将每个 GPU 视为自己的组
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # 数据分发 - 按 SP 组
    total_files = len(all_eval_data)  # 获取总数据文件数

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # 填充以匹配 SP 组数量
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # 在 SP 组之间分发
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    # 仅音频模式：始终保存音频
    save_audio_enabled = bool(config.get("save_audio", True))  # 获取是否启用音频保存标志
    for ckpt_path in ckpt_list:
        this_config = copy.deepcopy(config)  # 深拷贝配置，避免修改原始配置
        this_config.ovi_ckpt = ckpt_path  # 设置当前检查点路径

        # 为文件名添加简短标签
        try:
            ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]  # 从检查点路径提取文件名（不含扩展名），例如：/path/to/checkpoint-15000.safetensors -> checkpoint-15000
        except Exception:
            ckpt_tag = "ckpt"  # 如果提取失败，使用默认标签

        # Pass ckpt tag down for any visualizers that want it
        if this_config.get("attn_vis", None) is not None and hasattr(this_config.attn_vis, "get"):  # 检查是否有注意力可视化配置
            try:
                this_config.attn_vis.ckpt_tag = ckpt_tag  # 设置检查点标签
            except Exception:
                pass
        if this_config.get("sim_vis", None) is not None and hasattr(this_config.sim_vis, "get"):  # 检查是否有相似度可视化配置
            try:
                this_config.sim_vis.ckpt_tag = ckpt_tag  # 设置检查点标签
            except Exception:
                pass

        logging.info("Loading OVI Audio Engine...")
        ovi_engine = OviAudioEngine(config=this_config, device=device, target_dtype=target_dtype)  # 创建OVI音频引擎实例，加载模型（仅音频，无视频）
        logging.info("OVI Audio Engine loaded!")

        for _, text_prompt in tqdm(enumerate(this_rank_eval_data)):  # 遍历当前GPU分配的数据，显示进度条
            # Audio-only generation parameters
            seed = this_config.get("seed", 100)  # 获取随机种子，默认100
            solver_name = this_config.get("solver_name", "unipc")  # 获取求解器名称，默认unipc
            sample_steps = this_config.get("sample_steps", 50)  # 获取采样步数，默认50
            shift = this_config.get("shift", 5.0)  # 获取shift参数，默认5.0
            audio_guidance_scale = this_config.get("audio_guidance_scale", 3.0)  # 获取音频引导强度，默认3.0
            slg_layer = this_config.get("slg_layer", 11)  # 获取SLG层数，默认11
            audio_negative_prompt = this_config.get("audio_negative_prompt", "")  # 获取音频负提示词
            audio_duration = this_config.get("audio_duration", 5.04)  # 获取音频时长（秒），默认5.04秒

            for idx in range(this_config.get("each_example_n_times", 1)):  # 每个样本生成多次（用于测试不同随机种子）
                # Generate: OviAudioEngine.generate() 直接返回音频（numpy数组），不返回视频和图像
                generated_audio = ovi_engine.generate(  # 调用音频引擎生成音频
                    text_prompt=text_prompt,
                    seed=seed + idx,
                    solver_name=solver_name,
                    sample_steps=sample_steps,
                    shift=shift,
                    audio_guidance_scale=audio_guidance_scale,
                    slg_layer=slg_layer,
                    audio_negative_prompt=audio_negative_prompt,
                    audio_duration=audio_duration,
                )

                # Save audio
                if sp_rank == 0 and save_audio_enabled and generated_audio is not None:  # 只有组内rank 0的GPU保存结果，避免重复保存
                    formatted_prompt = format_prompt_for_filename(text_prompt)  # 格式化提示词为安全的文件名
                    # Use .wav extension for audio-only output
                    output_path = os.path.join(  # 拼接输出文件路径
                        output_dir,
                        f"{ckpt_tag}__{formatted_prompt}_{seed+idx}_{global_rank}.wav",
                    )
                    sample_rate = this_config.get("sample_rate", 16000)  # 获取采样率，默认16000Hz
                    save_audio(output_path, generated_audio, sample_rate=sample_rate)  # 保存音频文件为WAV格式
                    logging.info(f"Saved audio to: {output_path}")
        


if __name__ == "__main__":
    args = get_arguments()  # 解析命令行参数

    # Use the specific configuration file by default if not provided
    config_file = args.config_file if args.config_file else "./ovi/configs/inference/inference_audio.yaml"
    config = OmegaConf.load(config_file)  # 加载YAML配置文件
    
    # Optional checkpoint override:
    # - Prefer explicit `ovi_ckpt` from YAML.
    # - If not provided, fall back to a local trained checkpoint if present.
    if not config.get("ovi_ckpt", None):  # 如果配置中没有指定检查点
        trained_ckpt_path = "./outputs/t2a_train_V0129/step-22000.safetensors"
        if os.path.exists(trained_ckpt_path):  # 检查默认检查点是否存在
            config.ovi_ckpt = trained_ckpt_path  # 使用默认检查点
            logging.info(f"Setting ovi_ckpt to trained model (fallback): {trained_ckpt_path}")

    try:
        main(config=config, args=args)  # 执行主函数
    finally:
        # Avoid NCCL resource leak warnings on exit.
        if torch.distributed.is_available() and torch.distributed.is_initialized():  # 检查分布式是否已初始化
            torch.distributed.destroy_process_group()  # 清理分布式进程组，避免资源泄漏