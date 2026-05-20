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
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_fusion_engine import OviFusionEngine



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)
    if global_rank == 0:
        ovi_ckpt = config.get("ovi_ckpt", None)
        logging.info(f"OVI fusion checkpoint (config.ovi_ckpt): {ovi_ckpt}")
        if ovi_ckpt:
            assert os.path.exists(ovi_ckpt), f"ovi_ckpt does not exist: {ovi_ckpt}"

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # validate inputs before loading model to not waste time if input is not valid
    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path", None)
    assert config.get("mode") in ["t2v", "i2v", "t2i2v"], f"Invalid mode {config.get('mode')}, must be one of ['t2v', 'i2v', 't2i2v']"
    text_prompts, image_paths = validate_and_process_user_prompt(text_prompt, image_path, mode=config.get("mode"))
    if config.get("mode") != "i2v":
        logging.info(f"mode: {config.get('mode')}, setting all image_paths to None")
        image_paths = [None] * len(text_prompts)
    else:
        assert all(p is not None and os.path.isfile(p) for p in image_paths), f"In i2v mode, all image paths must be provided.{image_paths}"

    # Support comparing multiple fusion checkpoints in one run.
    compare_ckpts = config.get("compare_ovi_ckpts", None)
    if compare_ckpts is None:
        ckpt_list = [config.get("ovi_ckpt", None)]
    else:
        # Accept list/tuple or comma-separated string
        if isinstance(compare_ckpts, str):
            ckpt_list = [s.strip() for s in compare_ckpts.split(",") if s.strip()]
        else:
            ckpt_list = list(compare_ckpts)
    ckpt_list = [c for c in ckpt_list if c]
    if not ckpt_list:
        ckpt_list = [config.get("ovi_ckpt", None)]
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths))
    max_examples = config.get("max_examples", None)
    if max_examples is not None:
        try:
            max_examples = int(max_examples)
            if max_examples > 0:
                all_eval_data = all_eval_data[:max_examples]
        except Exception:
            pass

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    save_video_enabled = bool(config.get("save_video", True))
    for ckpt_path in ckpt_list:
        this_config = copy.deepcopy(config)
        this_config.ovi_ckpt = ckpt_path

        # Add a short tag for filenames
        try:
            ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        except Exception:
            ckpt_tag = "ckpt"

        # Pass ckpt tag down for any visualizers that want it
        if this_config.get("attn_vis", None) is not None and hasattr(this_config.attn_vis, "get"):
            try:
                this_config.attn_vis.ckpt_tag = ckpt_tag
            except Exception:
                pass
        if this_config.get("sim_vis", None) is not None and hasattr(this_config.sim_vis, "get"):
            try:
                this_config.sim_vis.ckpt_tag = ckpt_tag
            except Exception:
                pass

        logging.info("Loading OVI Fusion Engine...")
        ovi_engine = OviFusionEngine(config=this_config, device=device, target_dtype=target_dtype)
        logging.info("OVI Fusion Engine loaded!")

        for _, (text_prompt, image_path) in tqdm(enumerate(this_rank_eval_data)):
            video_frame_height_width = this_config.get("video_frame_height_width", None)
            seed = this_config.get("seed", 100)
            solver_name = this_config.get("solver_name", "unipc")
            sample_steps = this_config.get("sample_steps", 50)
            shift = this_config.get("shift", 5.0)
            video_guidance_scale = this_config.get("video_guidance_scale", 4.0)
            audio_guidance_scale = this_config.get("audio_guidance_scale", 3.0)
            slg_layer = this_config.get("slg_layer", 11)
            video_negative_prompt = this_config.get("video_negative_prompt", "")
            audio_negative_prompt = this_config.get("audio_negative_prompt", "")

            for idx in range(this_config.get("each_example_n_times", 1)):
                generated_video, generated_audio, generated_image = ovi_engine.generate(
                    text_prompt=text_prompt,
                    image_path=image_path,
                    video_frame_height_width=video_frame_height_width,
                    seed=seed + idx,
                    solver_name=solver_name,
                    sample_steps=sample_steps,
                    shift=shift,
                    video_guidance_scale=video_guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    slg_layer=slg_layer,
                    video_negative_prompt=video_negative_prompt,
                    audio_negative_prompt=audio_negative_prompt,
                )

                if sp_rank == 0 and save_video_enabled:
                    formatted_prompt = format_prompt_for_filename(text_prompt)
                    output_path = os.path.join(
                        output_dir,
                        f"{ckpt_tag}__{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}.mp4",
                    )
                    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                    if generated_image is not None:
                        generated_image.save(output_path.replace(".mp4", ".png"))
        


if __name__ == "__main__":
    args = get_arguments()

    # Use the specific configuration file by default if not provided
    config_file = args.config_file if args.config_file else "./AudioSummary/DiffSynth-Studio/ovi/configs/inference/inference_fusion.yaml"
    config = OmegaConf.load(config_file)
    
    # Optional checkpoint override:
    # - Prefer explicit `ovi_ckpt` from YAML.
    # - If not provided, fall back to a local trained checkpoint if present.
    if not config.get("ovi_ckpt", None):
        trained_ckpt_path = "models/train/Ovi_sft/epoch-3.safetensors"
        if os.path.exists(trained_ckpt_path):
            config.ovi_ckpt = trained_ckpt_path
            logging.info(f"Setting ovi_ckpt to trained model (fallback): {trained_ckpt_path}")

    try:
        main(config=config, args=args)
    finally:
        # Avoid NCCL resource leak warnings on exit.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()