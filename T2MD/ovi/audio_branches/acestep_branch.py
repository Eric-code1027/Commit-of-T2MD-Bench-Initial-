"""
ACE-Step 1.5 wrapper as a frozen audio branch.

基于 ACE-Step-1.5-main 真实 API:
- acestep.handler.AceStepHandler.initialize_service(...)
- acestep.inference.generate_music(dit_handler, llm_handler, params, config, save_dir)

Input: music caption string (5~10 seconds duration)
Output: 1D numpy float32 waveform @ 16kHz mono (对齐 Ovi save_video)
"""
import os
import sys
import numpy as np
import torch


class ACEStepAudioBranch:
    def __init__(
        self,
        acestep_project_root: str,
        config_path: str = "acestep-v15-xl-turbo",
        device: str = "auto",
        offload_to_cpu: bool = False,
    ):
        """
        acestep_project_root: ACE-Step-1.5-main 项目根目录(包含 checkpoints/ 子目录)
        config_path:          模型变体名,xl-turbo 是 4B 蒸馏版,4090 上跑得动
        device:               'auto' 让 ACE-Step 自动选(cuda),也可以 'cuda:1' 指定
        """
        # 把 ACE-Step 加进 sys.path(防止环境没 editable 装)
        if acestep_project_root not in sys.path:
            sys.path.insert(0, acestep_project_root)

        from acestep.handler import AceStepHandler

        self.acestep_root = acestep_project_root
        self._dit = AceStepHandler()

        status_msg, ok = self._dit.initialize_service(
            project_root=acestep_project_root,
            config_path=config_path,
            device=device,
            offload_to_cpu=offload_to_cpu,
            use_flash_attention=False,  # 与 Ovi 的 flash-attn 隔离,避免版本冲突
        )
        if not ok:
            raise RuntimeError(f"ACE-Step init failed: {status_msg}")

        self.native_sr = 48000   # ACE-Step 输出固定 48k stereo
        self.target_sr = 16000   # Ovi save_video 期望 16k mono

    @torch.no_grad()
    def generate(
        self,
        music_prompt: str,
        duration: float = 5.04,
        seed: int = 42,
        infer_steps: int = 8,
        lyrics: str = "[Instrumental]",
    ) -> np.ndarray:
        """
        return: 1D numpy float32 in [-1, 1], 采样率 16k mono,长度 = int(duration * 16000)
        """
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            task_type="text2music",
            thinking=False,                    # 不用 LLM,直接 DiT
            caption=music_prompt,
            lyrics=lyrics,
            duration=duration,
            inference_steps=infer_steps,
            guidance_scale=1.0,                # turbo 不用 CFG
            seed=seed,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=False,
            seeds=[seed],
        )

        result = generate_music(
            dit_handler=self._dit,
            llm_handler=None,                  # thinking=False 时允许 None
            params=params,
            config=config,
            save_dir=None,                     # 不存盘,直接拿 tensor
        )
        if not result.success:
            raise RuntimeError(f"ACE-Step generate failed: {result.status_message}")

        # result.audios[0] = {"path", "tensor", "key", "sample_rate", "params"}
        audio_dict = result.audios[0]
        wav = audio_dict["tensor"]   # torch.Tensor [channels=2, samples],float32,CPU
        sr = audio_dict["sample_rate"]

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().float().numpy()

        # stereo -> mono
        if wav.ndim == 2:
            wav = wav.mean(axis=0)

        # 48k -> 16k(Ovi save_video 期望)
        if sr != self.target_sr:
            import librosa
            wav = librosa.resample(
                wav.astype(np.float32),
                orig_sr=sr,
                target_sr=self.target_sr,
            )

        return np.clip(wav, -1.0, 1.0).astype(np.float32)