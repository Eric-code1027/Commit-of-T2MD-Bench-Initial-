#!/usr/bin/env python3
"""
Smoke test: 跑通 ACE-Step xl-turbo,不用 LLM(thinking=False),
直接出一段纯乐器音乐。验证 API 能跑通,作为后面 wrapper 的参照。
"""
import os
import sys
import time

# 清掉代理
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)

# 让 acestep 包能被 import
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "smoke_test")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ===== 1. 初始化 DiT handler (xl-turbo) =====
    logger.info("Init DiT handler (xl-turbo)...")
    t0 = time.time()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-xl-turbo",   # 用 xl-turbo
        device="auto",                         # auto -> cuda
        offload_to_cpu=False,
        use_flash_attention=False,             # 4090 不要打开 flash-attn(Ovi 占用了,可能冲突)
    )
    if not ok:
        logger.error(f"init failed: {status}")
        sys.exit(1)
    logger.info(f"DiT loaded in {time.time()-t0:.1f}s")

    # ===== 2. 准备参数 =====
    params = GenerationParams(
        task_type="text2music",
        thinking=False,        # 不用 LLM,直接 DiT
        caption="Up-tempo electronic hip-hop instrumental, around 110 BPM, heavy bass drops, crisp percussion, energetic and danceable mood.",
        lyrics="[Instrumental]",
        duration=5.04,         # 对齐 Ovi 默认 5.04s 视频时长
        inference_steps=8,     # turbo 8 步够
        guidance_scale=1.0,    # turbo 不用 CFG(代码里会强制改成 1.0)
        seed=42,
    )

    config = GenerationConfig(
        batch_size=1,
        audio_format="wav",
        use_random_seed=False,
        seeds=[42],
    )

    # ===== 3. 生成 =====
    logger.info("Generating...")
    t0 = time.time()
    result = generate_music(
        dit_handler=dit,
        llm_handler=None,     # thinking=False 时可以传 None
        params=params,
        config=config,
        save_dir=SAVE_DIR,
    )
    elapsed = time.time() - t0

    if not result.success:
        logger.error(f"FAILED in {elapsed:.1f}s: {result.status_message}")
        sys.exit(1)

    logger.info(f"OK in {elapsed:.1f}s")
    for a in result.audios:
        logger.info(f"  path: {a.get('path')}")
        logger.info(f"  tensor shape: {a.get('tensor').shape}, sample_rate: {a.get('sample_rate')}")


if __name__ == "__main__":
    main()