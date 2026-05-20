import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video_s2v import WanVideoS2VPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.loss import TrajectoryImitationS2VSelfForcingLoss
from diffsynth.diffusion.ema import FastEmaModelUpdater, compute_power_ema_exp
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.environ['LIBROSA_CACHE_LEVEL'] = '0'  # 禁用缓存


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        resume_from_checkpoint=None,
        ema_enabled=False,
        ema_rate=0.1,
        ema_device="cpu",
        ema_dtype="bf16",
        inject_motion_as_prefix=False,
        motion_prefix_frames=2,
        motion_dropout_rate=0.5,
        chunk_frames=25,
        motion_latent_frames=1,
        teacher_steps=40,
        student_steps=2,
        teacher_cfg_scale=5.0,
        use_regularization=False,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device, resume_from_checkpoint=resume_from_checkpoint)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoS2VPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.ema_enabled = False
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchS2VSFTLoss_WithIdentity(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchS2VSFTLoss_WithIdentity(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillS2VLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillS2VLoss(pipe, **inputs_shared, **inputs_posi),
        }
        if task == "ti_self_forcing":
            self.loss_fn = TrajectoryImitationS2VSelfForcingLoss(
                chunk_frames=chunk_frames,
                motion_latent_frames=motion_latent_frames,
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                teacher_cfg_scale=teacher_cfg_scale,
                use_regularization=use_regularization,
            )
            self.task_to_loss["ti_self_forcing"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)
            if ema_enabled:
                ema_device_obj = torch.device(ema_device)
                self._ema_model_ref = [copy.deepcopy(self.pipe.dit).float().to(ema_device_obj).eval().requires_grad_(False)]
                self.ema_updater = FastEmaModelUpdater()
                self.ema_exp = compute_power_ema_exp(ema_rate)
                self.ema_enabled = True
                self.ema_device = ema_device
                self.ema_dtype = ema_dtype
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.inject_motion_as_prefix = inject_motion_as_prefix
        self.motion_prefix_frames = motion_prefix_frames
        self.motion_dropout_rate = motion_dropout_rate
        
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        return super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            # FlashHead mode parameters
            "inject_motion_as_prefix": self.inject_motion_as_prefix,
            "motion_prefix_frames": self.motion_prefix_frames,
            "motion_dropout_rate": self.motion_dropout_rate,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from.")
    parser.add_argument("--inject_motion_as_prefix", default=False, action="store_true", help="Use FlashHead-style motion injection (motion frames as prefix instead of FramePack).")
    parser.add_argument("--motion_prefix_frames", type=int, default=2, help="Number of motion frames (in latent domain) to use as prefix when inject_motion_as_prefix is enabled.")
    parser.add_argument("--motion_dropout_rate", type=float, default=0.5, help="Probability of dropping motion frames during training (0.0-1.0). When dropped, motion frames are also noised.")
    parser.add_argument("--chunk_frames", type=int, default=25, help="Chunk frames for ti_self_forcing.")
    parser.add_argument("--motion_latent_frames", type=int, default=1, help="Motion latent frames for ti_self_forcing.")
    parser.add_argument("--teacher_steps", type=int, default=40, help="Teacher steps for ti_self_forcing.")
    parser.add_argument("--student_steps", type=int, default=2, help="Student steps for ti_self_forcing.")
    parser.add_argument("--teacher_cfg_scale", type=float, default=5.0, help="Teacher CFG scale for ti_self_forcing.")
    parser.add_argument("--ema_device", type=str, default="cpu", help="EMA device.")
    parser.add_argument("--ema_dtype", type=str, default="bf16", help="EMA dtype.")
    parser.add_argument("--use_regularization", default=False, action="store_true", help="Use regularization for ti_self_forcing.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        resume_from_checkpoint=args.resume_from_checkpoint,
        ema_enabled=getattr(args, 'ema_enabled', False),
        ema_rate=getattr(args, 'ema_rate', 0.1),
        ema_device=getattr(args, 'ema_device', 'cpu'),
        ema_dtype=getattr(args, 'ema_dtype', 'bf16'),
        inject_motion_as_prefix=args.inject_motion_as_prefix,
        motion_prefix_frames=args.motion_prefix_frames,
        motion_dropout_rate=args.motion_dropout_rate,
        chunk_frames=getattr(args, 'chunk_frames', 25),
        motion_latent_frames=getattr(args, 'motion_latent_frames', 1),
        teacher_steps=getattr(args, 'teacher_steps', 40),
        student_steps=getattr(args, 'student_steps', 2),
        teacher_cfg_scale=getattr(args, 'teacher_cfg_scale', 5.0),
        use_regularization=getattr(args, 'use_regularization', False),
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        start_step=args.start_step,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_s2v_task,
        "direct_distill": launch_direct_distill_s2v_task,
        "ti_self_forcing": launch_ti_self_forcing_s2v_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
