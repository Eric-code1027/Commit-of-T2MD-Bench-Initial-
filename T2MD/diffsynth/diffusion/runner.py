import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .ema import power_ema_beta
from torch.nn.utils import clip_grad_norm_


class SkipFirstNSampler(torch.utils.data.Sampler):
    """用于在第一个epoch跳过前N个样本的Sampler，后续epoch使用所有样本"""
    def __init__(self, dataset, skip_n=0, shuffle=True, seed=0):
        self.dataset_size = len(dataset)
        self.skip_n = min(skip_n, self.dataset_size)
        self.shuffle = shuffle
        self.seed = seed
        self.is_first_epoch = True
    
    def __iter__(self):
        if self.is_first_epoch and self.skip_n > 0:
            # 第一个epoch，跳过前N个样本
            indices = list(range(self.skip_n, self.dataset_size))
        else:
            # 后续epoch，使用所有样本
            indices = list(range(self.dataset_size))
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            # 先打乱indices
            shuffled_indices = torch.randperm(len(indices), generator=g).tolist()
            # 根据打乱的结果重新映射到实际的索引
            indices = [indices[i] for i in shuffled_indices]
        
        return iter(indices)
    
    def __len__(self):
        if self.is_first_epoch and self.skip_n > 0:
            return self.dataset_size - self.skip_n
        return self.dataset_size
    
    def reset_for_next_epoch(self):
        """重置为后续epoch模式，使用所有样本"""
        self.is_first_epoch = False

def launch_training_s2v_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        skip_samples = getattr(args, 'skip_samples', 0)
    else:
        skip_samples = 0
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 准备优化器和调度器
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                loss = loss_dict["loss"]
                v_loss = loss_dict["v_loss"]
                coeff = loss_dict["coeff"]
                timestep = loss_dict["timestep"]
                identity_loss = loss_dict["identity_loss"]
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), v_loss=v_loss.item(), coeff=coeff.item(), identity_loss=identity_loss.item(), timestep=timestep.item(), lr=current_lr)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_direct_distill_s2v_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        skip_samples = getattr(args, 'skip_samples', 0)
    else:
        skip_samples = 0
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 准备优化器和调度器
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=current_lr)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_trajectory_imitation_s2v_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        skip_samples = getattr(args, 'skip_samples', 0)
    else:
        skip_samples = 0
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 准备优化器和调度器
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    step_count = 0
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                loss = loss_dict["loss"]
                traj_loss = loss_dict["traj_loss"]
                regu_loss = loss_dict["regu_loss"]
                accelerator.backward(loss)
                optimizer.step()
                
                # EMA update
                unwrapped_model = accelerator.unwrap_model(model)
                if getattr(unwrapped_model, 'ema_enabled', False):
                    step_count += 1
                    ema_beta = power_ema_beta(step_count, unwrapped_model.ema_exp)
                    unwrapped_model.ema_updater.update_average(
                        unwrapped_model.pipe.dit,
                        unwrapped_model._ema_model_ref[0],
                        beta=ema_beta
                    )
                
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), traj_loss=traj_loss.item(), regu_loss=regu_loss.item(), lr=current_lr)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_ti_self_forcing_s2v_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    """Training loop for Trajectory-Imitation with self-forcing streaming."""
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    step_count = 0
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                loss = loss_dict["loss"]
                traj_loss = loss_dict["traj_loss"]
                regu_loss = loss_dict["regu_loss"]
                num_chunks = loss_dict["num_chunks"]
                accelerator.backward(loss)
                optimizer.step()
                
                unwrapped_model = accelerator.unwrap_model(model)
                if getattr(unwrapped_model, 'ema_enabled', False):
                    step_count += 1
                    ema_beta = power_ema_beta(step_count, unwrapped_model.ema_exp)
                    unwrapped_model.ema_updater.update_average(
                        unwrapped_model.pipe.dit,
                        unwrapped_model._ema_model_ref[0],
                        beta=ema_beta,
                    )
                
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(
                    loss=loss.item(),
                    traj=traj_loss.item() if hasattr(traj_loss, 'item') else traj_loss,
                    regu=regu_loss.item() if hasattr(regu_loss, 'item') else regu_loss,
                    chunks=num_chunks,
                    lr=current_lr,
                )
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_dmd_distill_s2v_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        skip_samples = getattr(args, 'skip_samples', 0)
        critic_update_freq = getattr(args, 'critic_update_freq', 5)
    else:
        skip_samples = 0
        critic_update_freq = 5
    
    # 使用单个 Optimizer 管理两个参数组
    # Group 0: Student (dit)
    # Group 1: Critic (fake_score_model)
    optimizer = torch.optim.AdamW([
        {"params": model.pipe.dit.parameters(), "lr": learning_rate, "weight_decay": weight_decay},
        {"params": model.pipe.fake_score_model.parameters(), "lr": learning_rate, "weight_decay": weight_decay}
    ])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 只需要 prepare 一次
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
        
    global_step = 0
    student_step_count = 0
    
    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            global_step += 1
            
            # 策略：每 (freq+1) 步中，前 freq 步 Critic，最后 1 步 Student
            # 例如 freq=5: C, C, C, C, C, S, C, C...
            # 这里我们简化逻辑：如果 step % (freq+1) != 0，则为 Critic
            is_student_step = (global_step % (critic_update_freq + 1) == 0)
            
            unwrapped_model = accelerator.unwrap_model(model)
            original_task = getattr(unwrapped_model, "task", "dmd_distill")
            
            if is_student_step:
                phase = "student"
                if original_task == "dmd_distill":
                    unwrapped_model.task = "dmd_distill:student"
                else:
                    unwrapped_model.task = original_task
            else:
                phase = "critic"
                unwrapped_model.task = "dmd_distill:critic"

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                
                # 保存当前 LR
                prev_lrs = [group['lr'] for group in optimizer.param_groups]

                # 动态调整 LR：交替训练时将非当前任务的 LR 设为 0
                # Group 0: Student, Group 1: Critic
                if is_student_step:
                    optimizer.param_groups[1]['lr'] = 0.0
                else:
                    optimizer.param_groups[0]['lr'] = 0.0
                    
                optimizer.step()

                # 恢复 LR
                for group, lr in zip(optimizer.param_groups, prev_lrs):
                    group['lr'] = lr
                
                # EMA update (only on student steps)
                if is_student_step and getattr(unwrapped_model, 'ema_enabled', False):
                    student_step_count += 1
                    ema_beta = power_ema_beta(student_step_count, unwrapped_model.ema_exp)
                    unwrapped_model.ema_updater.update_average(
                        unwrapped_model.pipe.dit,
                        unwrapped_model._ema_model_ref[0],
                        beta=ema_beta
                    )
                
                if save_steps is not None:
                     model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                     
                scheduler.step()
                
                # 这里的 current_lr 显示可能需要调整，显示当前活跃 group 的 lr
                active_lr = optimizer.param_groups[0]['lr'] if is_student_step else optimizer.param_groups[1]['lr']
                pbar.set_postfix(phase=phase, loss=loss.item(), lr=f"{active_lr:.2e}")
            
            # 恢复 Task
            unwrapped_model.task = original_task

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_dmd_distill_s2v_task_v2(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    """
    DMD Distill V2: fake_score_model 完全独立于 model 模块树（用 list 包裹），
    各自拥有独立的 Optimizer 和 DeepSpeed Engine，交替训练。
    """
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        critic_update_freq = getattr(args, 'critic_update_freq', 5)
    else:
        critic_update_freq = 5

    # 提取独立的 fake_score_model（存储在 list 中，不是 model 的子模块）
    fake_score_model = model._fake_score_model_ref[0]

    optimizer_student = torch.optim.AdamW(model.pipe.dit.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_critic = torch.optim.AdamW(fake_score_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_student = torch.optim.lr_scheduler.LinearLR(optimizer_student, start_factor=0.01, end_factor=1.0, total_iters=1000)
    scheduler_critic = torch.optim.lr_scheduler.LinearLR(optimizer_critic, start_factor=0.01, end_factor=1.0, total_iters=1000)

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    # 两次独立 prepare，各自成为独立的 Engine
    model, optimizer_student, scheduler_student, dataloader = accelerator.prepare(
        model, optimizer_student, scheduler_student, dataloader
    )
    fake_score_model, optimizer_critic, scheduler_critic = accelerator.prepare(
        fake_score_model, optimizer_critic, scheduler_critic
    )

    # 更新引用，让 loss 函数通过闭包能拿到 prepared 版本
    accelerator.unwrap_model(model)._fake_score_model_ref[0] = fake_score_model

    global_step = 0
    student_step_count = 0

    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            global_step += 1

            is_student_step = (global_step % (critic_update_freq + 1) == 0)

            unwrapped_model = accelerator.unwrap_model(model)
            original_task = getattr(unwrapped_model, "task", "dmd_distill_v2")

            if is_student_step:
                phase = "student"
                current_optimizer = optimizer_student
                current_scheduler = scheduler_student
                accumulate_target = model
                unwrapped_model.task = "dmd_distill_v2:student"
            else:
                phase = "critic"
                current_optimizer = optimizer_critic
                current_scheduler = scheduler_critic
                accumulate_target = fake_score_model
                unwrapped_model.task = "dmd_distill_v2:critic"

            with accelerator.accumulate(accumulate_target):
                current_optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)

                loss = loss_dict["loss"]
                accelerator.backward(loss)
                current_optimizer.step()

                if is_student_step and getattr(unwrapped_model, 'ema_enabled', False):
                    student_step_count += 1
                    ema_beta = power_ema_beta(student_step_count, unwrapped_model.ema_exp)
                    unwrapped_model.ema_updater.update_average(
                        unwrapped_model.pipe.dit,
                        unwrapped_model._ema_model_ref[0],
                        beta=ema_beta
                    )

                if save_steps is not None:
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

                current_scheduler.step()

                current_lr = current_optimizer.param_groups[0]['lr']
                pbar.set_postfix(phase=phase, loss=loss.item(), lr=f"{current_lr:.2e}")

            unwrapped_model.task = original_task

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_dmd_distill_s2v_task_v3(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    dataloader,
    student,
    teacher,
    fake_score_model,
    opt_student,
    sched_student,
    opt_critic,
    sched_critic,
    model_logger: ModelLogger,
    args=None,
):
    """
    DMD Distill V3: student/teacher/fake_score_model 完全独立于 model 模块树，
    各自由外部 accelerator.prepare 管理，交替训练。
    """
    if args is not None:
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        critic_update_freq = getattr(args, 'critic_update_freq', 5)
    else:
        save_steps = None
        num_epochs = 1
        critic_update_freq = 5

    global_step = 0
    student_step_count = 0
    if args is not None and hasattr(args, "gradient_accumulation_steps"):
        gradient_accumulation_steps = args.gradient_accumulation_steps
    else:
        gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    student_accum_step = 0
    critic_accum_step = 0

    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            global_step += 1

            is_student_step = (global_step % (critic_update_freq + 1) == 0)

            unwrapped_model = accelerator.unwrap_model(model)

            if is_student_step:
                phase = "student"
                student_accum_step += 1
                
                # 只在累积周期开始时 zero_grad
                if student_accum_step == 1:
                    opt_student.zero_grad()
                
                unwrapped_model.task = "dmd_distill_v3:student"
                if dataset.load_from_cache:
                    loss_dict = model({}, inputs=data)
                else:
                    loss_dict = model(data)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                
                # 只在累积周期结束时 step
                grad_norm_display = f"accum/{student_accum_step}"
                if student_accum_step == gradient_accumulation_steps:
                    grad_norm = accelerator.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    if grad_norm is not None:
                        grad_norm_display = f"{grad_norm:.4f}"
                    else:
                        grad_norm_display = "N/A"
                    opt_student.step()
                    sched_student.step()
                    student_accum_step = 0

                    if getattr(unwrapped_model, 'ema_enabled', False):
                        student_step_count += 1
                        ema_beta = power_ema_beta(student_step_count, unwrapped_model.ema_exp)
                        unwrapped_model.ema_updater.update_average(
                            unwrapped_model.pipe.dit,
                            unwrapped_model._ema_model_ref[0],
                            beta=ema_beta
                        )
            else:
                phase = "critic"
                critic_accum_step += 1
                
                # 只在累积周期开始时 zero_grad
                if critic_accum_step == 1:
                    opt_critic.zero_grad()
                
                with torch.no_grad():
                    if dataset.load_from_cache:
                        inputs = data
                    else:
                        inputs = unwrapped_model.get_pipeline_inputs(data)
                    inputs = unwrapped_model.transfer_data_to_device(
                        inputs, unwrapped_model.pipe.device, unwrapped_model.pipe.torch_dtype)
                    for unit in unwrapped_model.pipe.units:
                        inputs = unwrapped_model.pipe.unit_runner(unit, unwrapped_model.pipe, *inputs)
                loss_dict = unwrapped_model.task_to_loss["dmd_distill_v3:critic"](
                    unwrapped_model.pipe, *inputs)
                loss = loss_dict["loss"]
                fake_score_model.backward(loss)
                
                # 只在累积周期结束时 step
                grad_norm_display = f"accum/{critic_accum_step}"
                if critic_accum_step == gradient_accumulation_steps:
                    grad_norm = accelerator.clip_grad_norm_(fake_score_model.parameters(), max_norm=1.0)
                    if grad_norm is not None:
                        grad_norm_display = f"{grad_norm:.4f}"
                    else:
                        grad_norm_display = "N/A"
                    opt_critic.step()
                    sched_critic.step()
                    critic_accum_step = 0

            if save_steps is not None:
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

            current_lr = opt_student.param_groups[0]['lr'] if is_student_step else opt_critic.param_groups[0]['lr']
            sigma_mean = loss_dict["sigma_mean"].item() if "sigma_mean" in loss_dict else None
            nan_ratio = loss_dict["nan_sample_ratio"].item() if "nan_sample_ratio" in loss_dict else None
            x0_gap = loss_dict["x0_fake_teacher_gap"].item() if "x0_fake_teacher_gap" in loss_dict else None
            pbar.set_postfix(
                phase=phase,
                loss=loss.item(),
                loss_reg=loss_dict["loss_reg"].item() if "loss_reg" in loss_dict else "N/A",
                sigma=f"{sigma_mean:.4f}" if sigma_mean is not None else "N/A",
                nan=f"{nan_ratio:.4f}" if nan_ratio is not None else "N/A",
                gap=f"{x0_gap:.4f}" if x0_gap is not None else "N/A",
                grad_norm=grad_norm_display,
                lr=f"{current_lr:.2e}",
            )

            unwrapped_model.task = "dmd_distill_v3"

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        skip_samples = getattr(args, 'skip_samples', 0)
    else:
        skip_samples = 0
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 准备优化器和调度器
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    
    for epoch_id in range(num_epochs):
        # 为每个epoch创建DataLoader
        # if epoch_id == 0 and skip_samples > 0:
        #     # 第一个epoch，使用自定义sampler跳过前N个样本
        #     sampler = SkipFirstNSampler(dataset, skip_n=skip_samples, shuffle=True, seed=42)
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=lambda x: x[0], num_workers=num_workers)
        #     if accelerator.is_main_process:
        #         print(f"第一个epoch: 跳过前 {skip_samples} 个样本，剩余 {len(sampler)} 个样本")
        # else:
        #     # 后续epoch，使用所有样本，启用shuffle
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
        #     if accelerator.is_main_process and epoch_id > 0:
        #         print(f"Epoch {epoch_id}: 使用所有 {len(dataset)} 个样本")
        
        # 使用accelerator准备dataloader
        # dataloader = accelerator.prepare(dataloader)
        
        pbar = tqdm(dataloader)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=current_lr)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
