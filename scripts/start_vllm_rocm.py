#!/usr/bin/env python3
import os
import sys

PROJECT_ROOT = "/public/home/xdzs2026_a07/Medical-RAG-DCU"
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "Qwen2.5-3B-Instruct",
)
SHIM_DIR = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "vllm_rocm_shim",
)

def main():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"模型目录不存在：{MODEL_PATH}")

    env = os.environ.copy()

    # 当前进程和所有子进程均加载 ROCm 平台修复
    env["VLLM_FORCE_ROCM_SHIM"] = "1"
    env["PYTHONPATH"] = (
        SHIM_DIR
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    # 使用兼容性更稳的 vLLM V0，并关闭前端多进程
    env["VLLM_USE_V1"] = "0"

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", "qwen-local",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--dtype", "float16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.60",
        "--max-num-seqs", "8",
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-frontend-multiprocessing",
    ]

    print("正在以海光 ROCm 模式启动 vLLM……", flush=True)
    print("模型目录：", MODEL_PATH, flush=True)

    # 用真正的 vLLM 进程替换当前包装进程，
    # 避免 multiprocessing 重复执行本启动脚本
    os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
