import os

# 该文件会在每个新 Python 进程启动时自动加载
if os.getenv("VLLM_FORCE_ROCM_SHIM") == "1":
    import vllm.platforms as platforms
    platforms.builtin_platform_plugins.pop("cuda", None)
