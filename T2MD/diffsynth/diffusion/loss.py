from .base_pipeline import BasePipeline
from .flow_match import FlowMatchScheduler
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def FlowMatchS2VSFTLoss_WithIdentity(pipe: BasePipeline, **inputs):
    # FlashHead mode parameters
    inject_motion_as_prefix = inputs.get("inject_motion_as_prefix", False)
    motion_prefix_frames = inputs.get("motion_prefix_frames", 2)
    motion_dropout_rate = inputs.get("motion_dropout_rate", 0.5)
    
    # 原始损失准备
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    
    # 随机采样时间步
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    
    # 用于损失计算的起始帧（根据是否丢弃 motion 决定）
    loss_start_frame = 1  # 默认从第1帧开始
    
    if inject_motion_as_prefix:
        # FlashHead mode: decide whether to drop motion frames
        drop_motion = torch.rand(1, device=pipe.device).item() < motion_dropout_rate
        
        if drop_motion:
            # Motion Dropout: motion frames are also noised
            # Layout: [ref_frame(clean) | motion_frames(noised) | rest_frames(noised)]
            noisy_latents = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
            # Only ref_frame stays clean
            noisy_latents[:, :, 0:1] = inputs["input_latents"][:, :, 0:1]
            # Loss starts from frame 1 (motion frames are also predicted)
            loss_start_frame = 1
        else:
            # Provide Motion: motion frames are kept as GT (clean latents)
            # Layout: [ref_frame(clean) | motion_frames(clean) | rest_frames(noised)]
            motion_gt = inputs["input_latents"][:, :, 1:1+motion_prefix_frames]
            # Replace motion frames in noise with GT
            noise[:, :, 1:1+motion_prefix_frames] = motion_gt
            
            noisy_latents = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
            # Ensure ref_frame (frame 0) stays clean
            noisy_latents[:, :, 0:1] = inputs["input_latents"][:, :, 0:1]
            # Ensure motion frames stay clean
            noisy_latents[:, :, 1:1+motion_prefix_frames] = motion_gt
            # Loss starts from frame (1 + motion_prefix_frames) (skip ref and motion frames)
            loss_start_frame = 1 + motion_prefix_frames
    else:
        # Original mode: only ref_frame is kept clean
        noisy_latents = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        
        # 强制将第0帧替换为干净的 latents (Clean Reference)
        # 这一点对 S2V 身份保持至关重要！
        noisy_latents[:, :, 0:1] = inputs["input_latents"][:, :, 0:1]
        # Loss starts from frame 1
        loss_start_frame = 1
    
    inputs["latents"] = noisy_latents

    # 计算 Target (Velocity)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # 模型前向
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    # 注意：Wan S2V 模型输出会自动拼接 ref_frame 在第0位
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    # === 计算损失 ===
    # Use loss_start_frame determined above
    v_loss = torch.nn.functional.mse_loss(
        noise_pred[:, :, loss_start_frame:].float(), 
        training_target[:, :, loss_start_frame:].float()
    )
    coeff = pipe.scheduler.training_weight(timestep)
    
    # === 计算身份损失 (在 x0 空间对比) ===
    # 提取参考帧 (Clean x0)
    ref_frame = inputs["input_latents"][:, :, 0:1]
    
    # Use the same loss_start_frame for identity loss calculation
    pred_velocity = noise_pred[:, :, loss_start_frame:]
    current_xt = inputs["latents"][:, :, loss_start_frame:]
    
    # 获取当前的 sigma (用于反解 x0)
    # Flow Matching 公式: xt = x0 + sigma * v_pred  ==>  pred_x0 = xt - sigma * v_pred
    # 注意：这里假设 simplified flow matching，sigma 约为 t/T。
    # 更严谨的做法是从 scheduler 获取 sigma，但对于 Wan 模型，timesteps 已经是 sigma * 1000 的形式
    current_sigma = timestep / 1000.0
    pred_x0 = current_xt - current_sigma * pred_velocity
    
    # 在早期时间步增加身份约束 (t 越小，图像越清晰，Identity 约束越有效)
    identity_weight = 0.15
    
    # 计算 Identity Loss
    # 策略：生成的帧在时间维度平均后，应该接近参考帧（嘴部运动被平均掉，静态特征保留）
    identity_loss = torch.nn.functional.mse_loss(
        pred_x0.mean(dim=2),             # 预测视频的平均帧 (Reconstructed x0)
        ref_frame.squeeze(2).detach()    # 参考帧
    ) * identity_weight
    
    loss = v_loss * coeff.to(identity_loss.device) + identity_loss
    
    return {
        "loss": loss, 
        "v_loss": v_loss, 
        "identity_loss": identity_loss,
        "coeff": coeff,
        "timestep": timestep
    }


def FlowMatchS2VSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    v_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    coeff = pipe.scheduler.training_weight(timestep)
    loss = v_loss * coeff
    return {"loss": loss, "v_loss": v_loss, "coeff": coeff}


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillS2VLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs.get("num_inference_steps", 2))
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # === First Frame Conditioning ===
    if "input_latents" in inputs:
        # Force the first frame of the noisy input to be the clean reference frame
        inputs["latents"][:, :, 0:1] = inputs["input_latents"][:, :, 0:1]

    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
        
        # Enforce first frame consistency
        if "input_latents" in inputs:
             inputs["latents"][:, :, 0:1] = inputs["input_latents"][:, :, 0:1]

    # === Loss Calculation ===
    # 1. MSE Loss on generated frames (excluding first frame)
    v_loss = torch.nn.functional.mse_loss(
        inputs["latents"][:, :, 1:].float(), 
        inputs["input_latents"][:, :, 1:].float()
    )
    
    loss = v_loss
    
    return {
        "loss": loss
    }


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss


class TrajectoryImitationS2VLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            # S2V: Force first frame to be clean reference
            if "input_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["input_latents"][:, :, 0:1]

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            # S2V: Force first frame to be clean reference
            if "input_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["input_latents"][:, :, 0:1]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            # S2V: Force first frame to be clean reference
            if "input_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["input_latents"][:, :, 0:1]

        # 使用 decode_for_training: 在当前设备上单次解码，梯度自然流过 VAE (VAE 参数 requires_grad=False，不会累积梯度)
        device = next(pipe.dit.parameters()).device
        image_pred = pipe.vae.decode_for_training(inputs_shared["latents"], device=device)
        image_real = pipe.vae.decode_for_training(trajectory_teacher[-1].to(device), device=device)
        loss = self.loss_fn(image_pred.transpose(1, 2).flatten(0, 1).float(), image_real.transpose(1, 2).flatten(0, 1).float()).mean()
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(2)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 40, 5)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 2, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 2, 1)
        loss = loss_1 + loss_2
        return {"loss": loss, "traj_loss": loss_1, "regu_loss": loss_2}



def backward_simulation_s2v(pipe, inputs, num_steps: int = 4, with_grad: bool = True, last_step_grad_only: bool = True, dit_model=None, enable_tqdm: bool = False, use_fsdp: bool = False):
    """
    针对 Flow Matching S2V 的反向模拟 (Backward Simulation)
    使用 Wan 的 set_timesteps_wan(shift=5) 生成非线性时间步调度，与推理保持一致。
    
    Args:
        pipe: Pipeline 实例
        inputs: 输入数据
        num_steps: 模拟步数
        with_grad: 是否计算梯度
        last_step_grad_only: 是否只在最后一步计算梯度
        dit_model: 自定义 dit 模型（用于 FSDP 场景）
        enable_tqdm: 是否显示进度条
        use_fsdp: 是否使用 FSDP 兼容的 model_fn
    """
    
    # 1. 使用 Wan 的 shifted schedule 生成时间步
    sigmas_sched, timesteps_sched = FlowMatchScheduler.set_timesteps_wan(num_steps, shift=5.0)
    sigmas_full = torch.cat([sigmas_sched, torch.tensor([0.0])]).to(pipe.device)
    timesteps_full = torch.cat([timesteps_sched, torch.tensor([0.0])]).to(pipe.device)
    
    # 2. 初始化噪声
    clean_latents = inputs["input_latents"]
    latents = torch.randn_like(clean_latents)
    
    # [关键] 强制第一帧为 clean reference (Condition)
    latents[:, :, 0:1] = clean_latents[:, :, 0:1]
    
    # 获取模型引用
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if dit_model is not None:
        models['dit'] = dit_model
    
    sim_inputs = inputs.copy()
    
    # 选择使用哪个 model_fn
    if use_fsdp:
        from ..pipelines.wan_video_s2v import model_fn_wans2v_fsdp
        model_fn = model_fn_wans2v_fsdp
    else:
        model_fn = pipe.model_fn
    
    # 3. Euler 积分循环（在 sigma 空间计算步长）
    if enable_tqdm:
        pbar = tqdm(range(num_steps))
    else:
        pbar = range(num_steps)
    for i in pbar:
        sigma_curr = sigmas_full[i]
        sigma_next = sigmas_full[i + 1]
        t_curr = timesteps_full[i]
        
        is_last_step = (i == num_steps - 1)
        if not with_grad:
            context = torch.no_grad()
        elif last_step_grad_only and not is_last_step:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()
            
        with context:
            sim_inputs["latents"] = latents
            
            batch_size = latents.shape[0]
            t_tensor = torch.full((batch_size,), t_curr.item(), device=pipe.device, dtype=pipe.torch_dtype)
            
            step_inputs = sim_inputs.copy()
            step_inputs["timestep"] = t_tensor
            
            noise_pred = model_fn(**models, **step_inputs)
            
            dt = sigma_next - sigma_curr
            latents = latents + noise_pred * dt
            
            latents[:, :, 0:1] = clean_latents[:, :, 0:1]

    return latents


def DMDDistillS2VLoss(pipe: BasePipeline, **inputs):
    """
    DMD Student Loss (Generator Loss)
    """
    # 1. Backward Simulation (Student 生成 x0)
    # 随机选择模拟步数 (1-4步)
    num_steps = torch.randint(1, 5, (1,)).item()
    
    # 关键：Student 需要梯度
    G_x0 = backward_simulation_s2v(
        pipe, 
        inputs, 
        num_steps=num_steps, 
        with_grad=True, 
        last_step_grad_only=True # 节省显存
    )
    
    # 2. 构造判别器的时间步 t (D_time)
    # 随机采样一个时间步 t ~ U[0, 1000]
    batch_size = G_x0.shape[0]
    D_time = torch.randint(0, 1000, (batch_size,), device=pipe.device).long()
    
    # 3. 对生成的 x0 加噪得到 x_t
    # Flow Matching 加噪公式: x_t = (1-sigma)*x0 + sigma*noise
    # 注意：WanVideo 的 scheduler.add_noise 已经封装了这个逻辑
    noise = torch.randn_like(G_x0)
    # S2V: 保持第一帧不加噪 (条件帧)
    noise[:, :, 0:1] = 0 
    
    # 使用 scheduler 加噪
    # 注意：这里我们是对"生成的 x0"加噪，而不是对 GT 加噪
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)
    
    # 4. 获取 Teacher 和 Fake Score Network
    # 假设 pipe 中已经挂载了 teacher 和 fake_score
    # 可以在 training_module 中设置 pipe.teacher_model 和 pipe.fake_score_model
    if not hasattr(pipe, "teacher_model") or not hasattr(pipe, "fake_score_model"):
        raise ValueError("DMD Loss requires 'teacher_model' and 'fake_score_model' attached to the pipe.")
    
    # 5. Teacher 预测 (Ground Truth for Distribution)
    with torch.no_grad():
        # Teacher 预测 x0 (或 v -> x0)
        # 构造 teacher 的 models 字典
        # 除了 dit 换成 teacher_model，其他组件（如 vae, text_encoder）复用 pipe 的
        models_teacher = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        if 'dit' in models_teacher:
            models_teacher['dit'] = pipe.teacher_model
        
        # 构造 input 字典，避免 latents 参数冲突
        teacher_inputs = inputs.copy()
        teacher_inputs["latents"] = D_xt
        teacher_inputs["timestep"] = D_time
        
        v_teacher = pipe.model_fn(**models_teacher, **teacher_inputs)
        
        # 转换 v -> x0
        # Flow Matching: x0 = x_t - sigma * v
        sigmas = (D_time.float() / 1000.0).view(-1, 1, 1, 1, 1).to(pipe.device)
        x0_teacher = D_xt - sigmas * v_teacher
        
        # 可选：Teacher Guidance (CFG)
        if inputs.get("teacher_guidance_scale", 1.0) > 1.0:
            # 计算 uncond
            # ... (略去细节，如果有 uncond 输入的话)
            pass

    # 6. Fake Score Network 预测 (Discriminator)
    with torch.no_grad():
        # 构造 fake score 的 models 字典
        models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        if 'dit' in models_fake:
            models_fake['dit'] = pipe.fake_score_model
            
        # Fake Score 预测 v
        # 构造 input 字典，避免 latents 参数冲突
        fake_inputs_student_step = inputs.copy()
        fake_inputs_student_step["latents"] = D_xt
        fake_inputs_student_step["timestep"] = D_time
        
        v_fake = pipe.model_fn(**models_fake, **fake_inputs_student_step)
        
        # 转换 v -> x0
        x0_fake = D_xt - sigmas * v_fake

    # 7. 计算梯度方向 (DMD Gradient)
    # grad = (fake - teacher) / weight
    # weight 通常取 |fake - teacher| 的均值，用于稳定梯度
    
    # 忽略第一帧 (因为它是条件，不应该产生梯度)
    diff = (x0_fake[:, :, 1:] - x0_teacher[:, :, 1:]).float()
    
    # 计算自适应权重 (避免梯度爆炸)
    weight_factor = torch.abs(diff).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-5)
    
    # 归一化梯度方向
    grad_direction = diff / weight_factor
    
    # 补全第一帧的梯度 (0)
    grad_full = torch.zeros_like(G_x0)
    grad_full[:, :, 1:] = grad_direction

    # 8. 构造 DMD Loss
    # 我们希望 G_x0 移动方向是 -grad_direction
    # Loss = || G_x0 - (G_x0 - grad_direction).detach() ||^2
    target = (G_x0 - grad_full).detach()
    
    # 只计算非第一帧的 Loss
    loss_dmd = torch.nn.functional.mse_loss(
        G_x0[:, :, 1:].float(), 
        target[:, :, 1:].float()
    )
    
    return {"loss": loss_dmd}


def DMDCriticS2VLoss(pipe: BasePipeline, **inputs):
    """
    DMD Critic Loss (Fake Score Network Loss)
    """
    # 1. Backward Simulation (Student 生成 x0)
    # Critic 更新时不需要 Student 的梯度
    num_steps = torch.randint(1, 5, (1,)).item()
    with torch.no_grad():
        G_x0 = backward_simulation_s2v(
            pipe, 
            inputs, 
            num_steps=num_steps, 
            with_grad=False # 关键：这里不需要 Student 的梯度
        )
    
    # 2. 构造判别器的时间步 t
    batch_size = G_x0.shape[0]
    D_time = torch.randint(0, 1000, (batch_size,), device=pipe.device).long()
    
    # 3. 加噪
    noise = torch.randn_like(G_x0)
    noise[:, :, 0:1] = 0
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)
    
    # 4. Fake Score Network 预测
    # 这里需要 Fake Score 的梯度
    models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if 'dit' in models_fake:
        models_fake['dit'] = pipe.fake_score_model
    
    # Fake Score 预测 v
    # 移除 'latents' 关键字参数，因为 inputs 中已经包含了 'latents' (即 D_xt)
    # 并且如果 inputs 中有 'latents'，解包 **inputs 会导致与 latents=D_xt 冲突
    # 我们构造一个新的 input 字典
    fake_inputs = inputs.copy()
    fake_inputs["latents"] = D_xt
    fake_inputs["timestep"] = D_time
    
    v_fake = pipe.model_fn(**models_fake, **fake_inputs)
    
    # 转换 v -> x0
    sigmas = (D_time.float() / 1000.0).view(-1, 1, 1, 1, 1).to(pipe.device)
    x0_fake = D_xt - sigmas * v_fake
    
    # 5. Critic Loss
    # 目标：Fake Score Network 应该能够准确还原 Student 生成的样本
    # Loss = || x0_fake - G_x0.detach() ||^2
    # 同样忽略第一帧
    loss_critic = torch.nn.functional.mse_loss(
        x0_fake[:, :, 1:].float(),
        G_x0[:, :, 1:].detach().float() # G_x0 是 ground truth
    )
    
    # 加权 (可选)：根据 TurboDiffusion，可以除以 sigma^2 或者 sin^2
    # 这里简单起见使用 MSE，或者加权 loss_critic / (sigmas**2 + 1e-5)
    
    return {"loss": loss_critic}


def DMDDistillS2VLoss_v2(pipe: BasePipeline, fake_score_model, teacher_model, **inputs):
    """
    DMD Student Loss v2: fake_score_model 和 teacher_model 通过参数显式传入，不从 pipe 上读取。
    """
    num_steps = torch.randint(1, 5, (1,)).item()

    G_x0 = backward_simulation_s2v(
        pipe,
        inputs,
        num_steps=num_steps,
        with_grad=True,
        last_step_grad_only=True
    )

    batch_size = G_x0.shape[0]
    D_time = torch.randint(0, 1000, (batch_size,), device=pipe.device).long()

    noise = torch.randn_like(G_x0)
    noise[:, :, 0:1] = 0
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)

    # Teacher 预测
    with torch.no_grad():
        models_teacher = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        if 'dit' in models_teacher:
            models_teacher['dit'] = teacher_model

        teacher_inputs = inputs.copy()
        teacher_inputs["latents"] = D_xt
        teacher_inputs["timestep"] = D_time

        v_teacher = pipe.model_fn(**models_teacher, **teacher_inputs)

        sigmas = (D_time.float() / 1000.0).view(-1, 1, 1, 1, 1).to(pipe.device)
        x0_teacher = D_xt - sigmas * v_teacher

    # Fake Score 预测
    with torch.no_grad():
        models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        if 'dit' in models_fake:
            models_fake['dit'] = fake_score_model

        fake_inputs = inputs.copy()
        fake_inputs["latents"] = D_xt
        fake_inputs["timestep"] = D_time

        v_fake = pipe.model_fn(**models_fake, **fake_inputs)
        x0_fake = D_xt - sigmas * v_fake

    diff = (x0_fake[:, :, 1:] - x0_teacher[:, :, 1:]).float()
    weight_factor = torch.abs(diff).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-5)
    grad_direction = diff / weight_factor

    grad_full = torch.zeros_like(G_x0)
    grad_full[:, :, 1:] = grad_direction

    target = (G_x0 - grad_full).detach()

    loss_dmd = torch.nn.functional.mse_loss(
        G_x0[:, :, 1:].float(),
        target[:, :, 1:].float()
    )

    return {"loss": loss_dmd}


def DMDCriticS2VLoss_v2(pipe: BasePipeline, fake_score_model, **inputs):
    """
    DMD Critic Loss v2: fake_score_model 通过参数显式传入，不从 pipe 上读取。
    """
    num_steps = torch.randint(1, 5, (1,)).item()
    with torch.no_grad():
        G_x0 = backward_simulation_s2v(
            pipe,
            inputs,
            num_steps=num_steps,
            with_grad=False
        )

    batch_size = G_x0.shape[0]
    D_time = torch.randint(0, 1000, (batch_size,), device=pipe.device).long()

    noise = torch.randn_like(G_x0)
    noise[:, :, 0:1] = 0
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)

    models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if 'dit' in models_fake:
        models_fake['dit'] = fake_score_model

    fake_inputs = inputs.copy()
    fake_inputs["latents"] = D_xt
    fake_inputs["timestep"] = D_time

    v_fake = pipe.model_fn(**models_fake, **fake_inputs)

    sigmas = (D_time.float() / 1000.0).view(-1, 1, 1, 1, 1).to(pipe.device)
    x0_fake = D_xt - sigmas * v_fake

    loss_critic = torch.nn.functional.mse_loss(
        x0_fake[:, :, 1:].float(),
        G_x0[:, :, 1:].detach().float()
    )

    return {"loss": loss_critic}


def _lookup_sigmas(scheduler, timesteps_tensor):
    """
    从 scheduler 查表获取每个 timestep 对应的 shifted sigma。
    
    Wan 模型使用 shifted schedule: sigma_shifted = shift * sigma / (1 + (shift-1) * sigma)
    scheduler.sigmas 中存储的就是 shifted sigma。
    
    Flow Matching 加噪公式: x_t = (1 - sigma_shifted) * x0 + sigma_shifted * noise
    速度定义: v = noise - x0
    x0 转换公式: x0 = x_t - sigma_shifted * v
    
    Args:
        scheduler: FlowMatchScheduler 实例
        timesteps_tensor: 时间步张量
    
    Returns:
        sigma_shifted: shifted sigma（与 scheduler.sigmas 一致）
    """
    t_cpu = timesteps_tensor.cpu()
    sigma_shifted_list = []
    for t in t_cpu:
        tid = torch.argmin((scheduler.timesteps - t).abs())
        sigma_shifted_list.append(scheduler.sigmas[tid])
    sigma_shifted = torch.stack(sigma_shifted_list)
    
    return sigma_shifted


def _align_v_prediction_shape(v_pred, reference_latents):
    """
    对齐 v 预测与参考 latent 的时序长度，兼容 FSDP/非FSDP 的返回差异。
    """
    if v_pred.shape == reference_latents.shape:
        return v_pred

    if (
        v_pred.ndim == 5
        and reference_latents.ndim == 5
        and v_pred.shape[:2] == reference_latents.shape[:2]
        and v_pred.shape[3:] == reference_latents.shape[3:]
    ):
        if v_pred.shape[2] + 1 == reference_latents.shape[2]:
            # 缺失 reference frame 时补零，保持 reference 帧不被更新。
            return torch.cat([torch.zeros_like(reference_latents[:, :, :1]), v_pred], dim=2)
        if v_pred.shape[2] == reference_latents.shape[2] + 1:
            return v_pred[:, :, : reference_latents.shape[2]]

    raise RuntimeError(
        f"v_pred shape {tuple(v_pred.shape)} is incompatible with latents shape {tuple(reference_latents.shape)}"
    )


def DMDDistillS2VLoss_v3(pipe: BasePipeline, student, fake_score_model, teacher_model, inputs_nega=None, teacher_guidance_scale=5.0, **inputs):
    """
    DMD Student Loss v3: 修复时间步采样和权重问题，添加 CFG 支持。

    核心修复:
    1. 限制 D_time 范围到 20~980，避开极端时间步（低时间步 sigma 极小导致 1/sigma^2 爆炸）
    2. 移除 1/sigma^2 权重，使用原始 MSE loss
    3. 使用 shifted sigma 进行 x0 转换（与加噪公式一致）
    4. 使用 double 精度计算梯度，避免数值不稳定
    5. 添加 NaN 处理
    6. 添加 CFG (Classifier-Free Guidance) 支持
    7. 支持 FSDP 包装的模型，使用 model_fn_wans2v_fsdp 兼容函数
    """
    from ..pipelines.wan_video_s2v import model_fn_wans2v_fsdp

    num_steps = torch.randint(1, 5, (1,)).item()

    is_fsdp_student = isinstance(student, FSDP)
    G_x0 = backward_simulation_s2v(
        pipe,
        inputs,
        num_steps=num_steps,
        with_grad=True,
        last_step_grad_only=False,
        dit_model=student,
        use_fsdp=is_fsdp_student,
    )

    batch_size = G_x0.shape[0]
    D_time = torch.randint(int(0.02 * 1000), int(0.98 * 1000), (batch_size,), device=pipe.device).long()
    noise = torch.randn_like(G_x0)
    noise[:, :, 0:1] = 0
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)
    D_xt[:, :, 0:1] = G_x0[:, :, 0:1]
    sigma_shifted = pipe.scheduler.get_sigma_from_timestep(D_time).view(-1, 1, 1, 1, 1).to(pipe.device, dtype=G_x0.dtype)

    with torch.no_grad():
        models_teacher = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        models_teacher['dit'] = teacher_model

        teacher_inputs_cond = inputs.copy()
        teacher_inputs_cond["latents"] = D_xt
        teacher_inputs_cond["timestep"] = D_time
        v_teacher_cond = _align_v_prediction_shape(pipe.model_fn(**models_teacher, **teacher_inputs_cond), D_xt)

        if teacher_guidance_scale > 1.0 and inputs_nega is not None:
            teacher_inputs_uncond = inputs.copy()
            for key in inputs_nega:
                if inputs_nega[key] is not None:
                    teacher_inputs_uncond[key] = inputs_nega[key]
            teacher_inputs_uncond["latents"] = D_xt
            teacher_inputs_uncond["timestep"] = D_time
            v_teacher_uncond = _align_v_prediction_shape(pipe.model_fn(**models_teacher, **teacher_inputs_uncond), D_xt)
            v_teacher = v_teacher_uncond + teacher_guidance_scale * (v_teacher_cond - v_teacher_uncond)
        else:
            v_teacher = v_teacher_cond

        x0_teacher = D_xt - sigma_shifted * v_teacher

    with torch.no_grad():
        models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        models_fake['dit'] = fake_score_model

        fake_inputs = inputs.copy()
        fake_inputs["latents"] = D_xt
        fake_inputs["timestep"] = D_time

        is_fsdp_fake = isinstance(fake_score_model, FSDP)
        if is_fsdp_fake:
            v_fake = _align_v_prediction_shape(model_fn_wans2v_fsdp(**models_fake, **fake_inputs), D_xt)
        else:
            v_fake = _align_v_prediction_shape(pipe.model_fn(**models_fake, **fake_inputs), D_xt)
        x0_fake = D_xt - sigma_shifted * v_fake

    diff = (x0_fake[:, :, 1:].double() - x0_teacher[:, :, 1:].double())
    weight_factor = torch.abs(G_x0[:, :, 1:].double() - x0_teacher[:, :, 1:].double()).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-5)
    grad_direction = diff / weight_factor

    grad_full = torch.zeros_like(G_x0)
    grad_full[:, :, 1:] = grad_direction

    target = (G_x0.double() - grad_full.double()).detach()
    loss_map = (G_x0[:, :, 1:].double() - target[:, :, 1:].double()) ** 2
    nan_sample_mask = torch.isnan(loss_map).flatten(start_dim=1).any(dim=1)
    if nan_sample_mask.any():
        loss_map[nan_sample_mask] = 0

    loss_dmd = loss_map.mean()

    reg_scale = float(inputs.get("dmd_reg_scale", 0.0))
    loss_reg = torch.zeros((), device=pipe.device, dtype=torch.float64)
    if reg_scale > 0:
        clean_x0 = inputs["input_latents"]
        loss_reg = torch.nn.functional.mse_loss(
            G_x0[:, :, 1:].double(),
            clean_x0[:, :, 1:].double()
        )

    loss_total = loss_dmd + reg_scale * loss_reg
    sigma_stats = sigma_shifted[:, 0, 0, 0, 0].float()
    return {
        "loss": loss_total,
        "loss_dmd": loss_dmd,
        "loss_reg": loss_reg,
        "sigma_mean": sigma_stats.mean(),
        "sigma_min": sigma_stats.min(),
        "sigma_max": sigma_stats.max(),
        "x0_fake_teacher_gap": torch.abs(x0_fake[:, :, 1:].float() - x0_teacher[:, :, 1:].float()).mean(),
        "nan_sample_ratio": nan_sample_mask.float().mean(),
    }


def DMDCriticS2VLoss_v3(pipe: BasePipeline, student, fake_score_model, **inputs):
    """
    DMD Critic Loss v3:
    让 fake_score 在噪声样本上预测 velocity，与 student 的 G_x0 对应的真实 velocity 一致（忽略 reference 帧）。
    训练目标使用 flow 空间 MSE（velocity 预测与 noise - G_x0 的 MSE），与 CausVid 一致，以稳定 loss 量级。
    """
    from ..pipelines.wan_video_s2v import model_fn_wans2v_fsdp

    num_steps = torch.randint(1, 5, (1,)).item()

    is_fsdp_student = isinstance(student, FSDP)
    with torch.no_grad():
        G_x0 = backward_simulation_s2v(
            pipe,
            inputs,
            num_steps=num_steps,
            with_grad=False,
            dit_model=student,
            use_fsdp=is_fsdp_student,
        )

    batch_size = G_x0.shape[0]
    D_time = torch.randint(int(0.02 * 1000), int(0.98 * 1000), (batch_size,), device=pipe.device).long()
    noise = torch.randn_like(G_x0)
    noise[:, :, 0:1] = 0
    D_xt = pipe.scheduler.add_noise(G_x0, noise, D_time)
    D_xt[:, :, 0:1] = G_x0[:, :, 0:1]
    sigma_shifted = pipe.scheduler.get_sigma_from_timestep(D_time).view(-1, 1, 1, 1, 1).to(pipe.device, dtype=G_x0.dtype)

    models_fake = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    models_fake['dit'] = fake_score_model

    fake_inputs = inputs.copy()
    fake_inputs["latents"] = D_xt
    fake_inputs["timestep"] = D_time

    is_fsdp_fake = isinstance(fake_score_model, FSDP)
    if is_fsdp_fake:
        v_fake = _align_v_prediction_shape(model_fn_wans2v_fsdp(**models_fake, **fake_inputs), D_xt)
    else:
        v_fake = _align_v_prediction_shape(pipe.model_fn(**models_fake, **fake_inputs), D_xt)

    v_target = (noise - G_x0)[:, :, 1:].detach()
    loss_map = (v_fake[:, :, 1:].float() - v_target.float()) ** 2
    nan_sample_mask = torch.isnan(loss_map).flatten(start_dim=1).any(dim=1)
    if nan_sample_mask.any():
        loss_map[nan_sample_mask] = 0
    loss_critic = loss_map.mean()

    sigma_stats = sigma_shifted[:, 0, 0, 0, 0].float()
    return {
        "loss": loss_critic,
        "sigma_mean": sigma_stats.mean(),
        "sigma_min": sigma_stats.min(),
        "sigma_max": sigma_stats.max(),
        "nan_sample_ratio": nan_sample_mask.float().mean(),
    }


def backward_simulation_s2v_streaming(
    pipe, inputs, num_steps: int = 4,
    motion_latents=None,
    with_grad: bool = True, last_step_grad_only: bool = True,
    dit_model=None, enable_tqdm: bool = False, use_fsdp: bool = False,
):
    """
    Backward simulation for streaming self-forcing S2V training.
    Extends backward_simulation_s2v with motion latent injection:
    positions 1:1+K are frozen to motion_latents during the entire denoising.

    Args:
        motion_latents: [B, C, K, H, W] motion conditioning from previous chunk.
                        If None, behaves identically to backward_simulation_s2v.
    """
    sigmas_sched, timesteps_sched = FlowMatchScheduler.set_timesteps_wan(num_steps, shift=5.0)
    sigmas_full = torch.cat([sigmas_sched, torch.tensor([0.0])]).to(pipe.device)
    timesteps_full = torch.cat([timesteps_sched, torch.tensor([0.0])]).to(pipe.device)

    clean_latents = inputs["input_latents"]
    latents = torch.randn_like(clean_latents)

    latents[:, :, 0:1] = clean_latents[:, :, 0:1]
    if motion_latents is not None:
        K = motion_latents.shape[2]
        latents[:, :, 1:1 + K] = motion_latents

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if dit_model is not None:
        models['dit'] = dit_model

    sim_inputs = inputs.copy()

    if use_fsdp:
        from ..pipelines.wan_video_s2v import model_fn_wans2v_fsdp
        model_fn = model_fn_wans2v_fsdp
    else:
        model_fn = pipe.model_fn

    if enable_tqdm:
        pbar = tqdm(range(num_steps))
    else:
        pbar = range(num_steps)
    for i in pbar:
        sigma_curr = sigmas_full[i]
        sigma_next = sigmas_full[i + 1]
        t_curr = timesteps_full[i]

        is_last_step = (i == num_steps - 1)
        if not with_grad:
            context = torch.no_grad()
        elif last_step_grad_only and not is_last_step:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            sim_inputs["latents"] = latents

            batch_size = latents.shape[0]
            t_tensor = torch.full((batch_size,), t_curr.item(), device=pipe.device, dtype=pipe.torch_dtype)

            step_inputs = sim_inputs.copy()
            step_inputs["timestep"] = t_tensor

            noise_pred = model_fn(**models, **step_inputs)

            dt = sigma_next - sigma_curr
            latents = latents + noise_pred * dt

            latents[:, :, 0:1] = clean_latents[:, :, 0:1]
            if motion_latents is not None:
                latents[:, :, 1:1 + K] = motion_latents

    return latents


class TrajectoryImitationS2VSelfForcingLoss(torch.nn.Module):
    """
    Multi-chunk Trajectory Imitation loss with self-forcing for streaming S2V.

    During training the long video is split into overlapping chunks.
    Chunk 0 is trained with standard TI loss (no motion conditioning).
    For subsequent chunks the student's own generation from the previous chunk
    provides the motion latent, bridging the train-test gap.

    Key latent layout per chunk (chunk_latent_frames = 7 for chunk_frames=25):
        position 0        : reference image (frozen)
        position 1        : motion latent from previous chunk (frozen, absent for chunk 0)
        positions 1/2 .. 6: new content (loss computed here)
    """

    def __init__(
        self,
        chunk_frames: int = 25,
        motion_latent_frames: int = 1,
        teacher_steps: int = 40,
        student_steps: int = 2,
        teacher_cfg_scale: float = 5.0,
        use_regularization: bool = False,
    ):
        super().__init__()
        self.chunk_frames = chunk_frames
        self.motion_latent_frames = motion_latent_frames
        self.teacher_steps = teacher_steps
        self.student_steps = student_steps
        self.teacher_cfg_scale = teacher_cfg_scale
        self.use_regularization = use_regularization

        self.chunk_latent_frames = (chunk_frames - 1) // 4 + 1           # 7
        self.content_latent_frames = self.chunk_latent_frames - 1         # 6
        self.slice_latent_frames = self.content_latent_frames - motion_latent_frames  # 5
        self.chunk_audio_frames = chunk_frames - 1                        # 24
        self.slice_pixel_frames = chunk_frames - 1 - motion_latent_frames * 4  # 20

        self.initialized = False

    def initialize(self, device):
        if self.use_regularization:
            import lpips
            self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_num_chunks(self, total_content_latent_frames: int) -> int:
        if total_content_latent_frames <= self.content_latent_frames:
            return 1
        remaining = total_content_latent_frames - self.content_latent_frames
        import math
        return 1 + math.ceil(remaining / self.slice_latent_frames)

    def _build_chunk_gt(self, full_latents, ref_latent, chunk_idx):
        """Assemble the ground-truth latent tensor for *chunk_idx*."""
        if chunk_idx == 0:
            return full_latents[:, :, :self.chunk_latent_frames]

        offset = self.content_latent_frames + (chunk_idx - 1) * self.slice_latent_frames
        motion_gt = full_latents[:, :, offset:offset + self.motion_latent_frames]
        new_start = offset + self.motion_latent_frames
        new_end = new_start + self.slice_latent_frames
        new_content = full_latents[:, :, new_start:new_end]

        if new_content.shape[2] < self.slice_latent_frames:
            pad = self.slice_latent_frames - new_content.shape[2]
            new_content = torch.nn.functional.pad(new_content, (0, 0, 0, 0, 0, pad))

        return torch.cat([ref_latent, motion_gt, new_content], dim=2)

    def _get_audio_slice(self, full_audio, chunk_idx):
        start = chunk_idx * self.slice_pixel_frames
        end = start + self.chunk_audio_frames
        audio = full_audio[..., start:end]
        if audio.shape[-1] < self.chunk_audio_frames:
            pad = self.chunk_audio_frames - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        return audio

    def _prepare_chunk_io(
        self,
        chunk_gt, chunk_audio, motion_latents,
        inputs_shared_base, inputs_posi_base, inputs_nega_base,
    ):
        """Create per-chunk copies of the three input dicts."""
        s = inputs_shared_base.copy()
        p = inputs_posi_base.copy()
        n = inputs_nega_base.copy()

        s["input_latents"] = chunk_gt
        s["latents"] = torch.randn_like(chunk_gt)
        s["latents"][:, :, 0:1] = chunk_gt[:, :, 0:1]
        if motion_latents is not None:
            K = motion_latents.shape[2]
            s["latents"][:, :, 1:1 + K] = motion_latents

        p["audio_embeds"] = chunk_audio
        n["audio_embeds"] = 0.0 * chunk_audio
        return s, p, n

    # ------------------------------------------------------------------
    # per-chunk TI stages (adapted from TrajectoryImitationS2VLoss)
    # ------------------------------------------------------------------

    def _freeze_conditioning(self, latents, input_latents, motion_latents):
        """Re-impose ref frame and (optionally) motion latent conditioning."""
        latents[:, :, 0:1] = input_latents[:, :, 0:1]
        if motion_latents is not None:
            K = motion_latents.shape[2]
            latents[:, :, 1:1 + K] = motion_latents
        return latents

    def fetch_trajectory_for_chunk(
        self, pipe_teacher, timesteps_student,
        inputs_shared, inputs_posi, inputs_nega,
        num_inference_steps, cfg_scale, motion_latents=None,
    ):
        trajectory = [inputs_shared["latents"].clone()]

        pipe_teacher.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe_teacher, name) for name in pipe_teacher.in_iteration_models}
        for progress_id, timestep in enumerate(pipe_teacher.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe_teacher.torch_dtype, device=pipe_teacher.device)
            noise_pred = pipe_teacher.cfg_guided_model_fn(
                pipe_teacher.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = pipe_teacher.step(
                pipe_teacher.scheduler, progress_id=progress_id,
                noise_pred=noise_pred.detach(), **inputs_shared,
            )
            inputs_shared["latents"] = self._freeze_conditioning(
                inputs_shared["latents"], inputs_shared["input_latents"], motion_latents,
            )
            trajectory.append(inputs_shared["latents"].clone())

        return pipe_teacher.scheduler.timesteps, trajectory

    def align_trajectory_for_chunk(
        self, pipe, timesteps_teacher, trajectory_teacher,
        inputs_shared, inputs_posi, inputs_nega,
        num_inference_steps, cfg_scale,
        motion_latents=None, loss_start_pos: int = 1,
    ):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]
            inputs_shared["latents"] = self._freeze_conditioning(
                inputs_shared["latents"], inputs_shared["input_latents"], motion_latents,
            )

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                pid_t = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[pid_t]

            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            step_loss = torch.nn.functional.mse_loss(
                noise_pred[:, :, loss_start_pos:].float(),
                target[:, :, loss_start_pos:].float(),
            )
            weight = pipe.scheduler.training_weight(timestep)
            if weight.dim() > 0:
                weight = weight.squeeze()
            loss = loss + step_loss * weight.to(step_loss.device)
        return loss

    def generate_student_x0_for_chunk(
        self, pipe, inputs_shared, inputs_posi, inputs_nega,
        num_inference_steps, cfg_scale, motion_latents=None,
    ):
        """Run student inference (no grad) to obtain x0 for motion extraction."""
        inputs_shared["latents"] = torch.randn_like(inputs_shared["input_latents"])
        inputs_shared["latents"] = self._freeze_conditioning(
            inputs_shared["latents"], inputs_shared["input_latents"], motion_latents,
        )

        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = pipe.step(
                pipe.scheduler, progress_id=progress_id,
                noise_pred=noise_pred.detach(), **inputs_shared,
            )
            inputs_shared["latents"] = self._freeze_conditioning(
                inputs_shared["latents"], inputs_shared["input_latents"], motion_latents,
            )
        return inputs_shared["latents"]

    # ------------------------------------------------------------------
    # main forward
    # ------------------------------------------------------------------

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)

        full_latents = inputs_shared["input_latents"]
        full_audio = inputs_posi["audio_embeds"]
        ref_latent = full_latents[:, :, 0:1]
        total_content = full_latents.shape[2] - 1
        num_chunks = self._get_num_chunks(total_content)

        pipe_teacher = inputs_shared["teacher"]

        total_traj_loss = 0
        total_regu_loss = 0
        motion_latents = None

        for chunk_idx in range(num_chunks):
            chunk_gt = self._build_chunk_gt(full_latents, ref_latent, chunk_idx)
            chunk_audio = self._get_audio_slice(full_audio, chunk_idx)

            is_first = (chunk_idx == 0)
            cur_motion = None if is_first else motion_latents
            loss_start = 1 if is_first else (1 + self.motion_latent_frames)

            # -- teacher trajectory (GT motion for teacher) --
            gt_motion_for_teacher = None
            if not is_first:
                offset = self.content_latent_frames + (chunk_idx - 1) * self.slice_latent_frames
                gt_motion_for_teacher = full_latents[:, :, offset:offset + self.motion_latent_frames]

            chunk_shared_t, chunk_posi_t, chunk_nega_t = self._prepare_chunk_io(
                chunk_gt, chunk_audio, gt_motion_for_teacher,
                inputs_shared, inputs_posi, inputs_nega,
            )
            with torch.no_grad():
                pipe_teacher.scheduler.set_timesteps(self.student_steps)
                ts_teacher, traj_teacher = self.fetch_trajectory_for_chunk(
                    pipe_teacher, pipe_teacher.scheduler.timesteps,
                    chunk_shared_t, chunk_posi_t, chunk_nega_t,
                    self.teacher_steps, self.teacher_cfg_scale,
                    motion_latents=gt_motion_for_teacher,
                )
                ts_teacher = ts_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)

            # -- student alignment (self-forced motion) --
            chunk_shared_s, chunk_posi_s, chunk_nega_s = self._prepare_chunk_io(
                chunk_gt, chunk_audio, cur_motion,
                inputs_shared, inputs_posi, inputs_nega,
            )
            traj_loss = self.align_trajectory_for_chunk(
                pipe, ts_teacher, traj_teacher,
                chunk_shared_s, chunk_posi_s, chunk_nega_s,
                self.student_steps, 1,
                motion_latents=cur_motion, loss_start_pos=loss_start,
            )
            total_traj_loss = total_traj_loss + traj_loss

            # -- optional LPIPS regularization --
            regu_loss = torch.tensor(0.0, device=pipe.device)
            if self.use_regularization:
                chunk_shared_r, chunk_posi_r, chunk_nega_r = self._prepare_chunk_io(
                    chunk_gt, chunk_audio, cur_motion,
                    inputs_shared, inputs_posi, inputs_nega,
                )
                chunk_shared_r["latents"] = traj_teacher[0]
                chunk_shared_r["latents"] = self._freeze_conditioning(
                    chunk_shared_r["latents"], chunk_shared_r["input_latents"], cur_motion,
                )
                pipe.scheduler.set_timesteps(self.student_steps)
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                for pid, ts in enumerate(pipe.scheduler.timesteps):
                    ts = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    np_ = pipe.cfg_guided_model_fn(
                        pipe.model_fn, 1,
                        chunk_shared_r, chunk_posi_r, chunk_nega_r,
                        **models, timestep=ts, progress_id=pid,
                    )
                    chunk_shared_r["latents"] = pipe.step(
                        pipe.scheduler, progress_id=pid, noise_pred=np_.detach(), **chunk_shared_r,
                    )
                    chunk_shared_r["latents"] = self._freeze_conditioning(
                        chunk_shared_r["latents"], chunk_shared_r["input_latents"], cur_motion,
                    )
                device = next(pipe.dit.parameters()).device
                img_pred = pipe.vae.decode_for_training(chunk_shared_r["latents"], device=device)
                img_real = pipe.vae.decode_for_training(traj_teacher[-1].to(device), device=device)
                regu_loss = self.loss_fn(
                    img_pred.transpose(1, 2).flatten(0, 1).float(),
                    img_real.transpose(1, 2).flatten(0, 1).float(),
                ).mean()
                total_regu_loss = total_regu_loss + regu_loss

            # -- extract motion for next chunk (self-forcing) --
            if chunk_idx < num_chunks - 1:
                with torch.no_grad():
                    chunk_shared_m, chunk_posi_m, chunk_nega_m = self._prepare_chunk_io(
                        chunk_gt, chunk_audio, cur_motion,
                        inputs_shared, inputs_posi, inputs_nega,
                    )
                    student_x0 = self.generate_student_x0_for_chunk(
                        pipe, chunk_shared_m, chunk_posi_m, chunk_nega_m,
                        self.student_steps, 1,
                        motion_latents=cur_motion,
                    )
                    motion_latents = student_x0[:, :, -self.motion_latent_frames:].detach()

        loss = total_traj_loss + total_regu_loss
        return {
            "loss": loss,
            "traj_loss": total_traj_loss,
            "regu_loss": total_regu_loss,
            "num_chunks": num_chunks,
        }
