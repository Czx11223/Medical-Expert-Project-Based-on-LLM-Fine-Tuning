# -*- coding: utf-8 -*-
"""
============================================================
环境自检脚本（训练前必跑）
============================================================
检查项：
  1. Python 版本（需 >= 3.10，bitsandbytes Windows 版要求）
  2. 核心依赖库是否已安装及版本（torch/transformers/peft 等）
  3. CUDA 是否可用、GPU 型号与显存
  4. bitsandbytes 4bit 量化是否可用（Windows 兼容性）
  5. 数据集文件是否就位
  6. 磁盘剩余空间（模型缓存 + 训练输出需要约 20GB+）

用法：python scripts/00_check_env.py
"""

import importlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASS, WARN, FAIL = "[通过]", "[警告]", "[缺失]"


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"  {tag} {name:24s} {detail}")
    return ok


def main():
    print("=" * 62)
    print(" 基于大模型微调的医疗专家 - 环境自检")
    print("=" * 62)

    # ---------- 1. Python 版本 ----------
    print("\n[1] Python 解释器")
    ver = sys.version_info
    check("Python 版本", ver >= (3, 10),
          f"当前 {ver.major}.{ver.minor}.{ver.micro}（需 >= 3.10）")
    print(f"       解释器路径：{sys.executable}")

    # ---------- 2. 依赖库 ----------
    print("\n[2] 依赖库安装情况")
    libs = [
        ("torch", "PyTorch 训练框架"),
        ("transformers", "模型加载/训练"),
        ("peft", "LoRA 适配器"),
        ("accelerate", "设备调度加速"),
        ("bitsandbytes", "4bit 量化（Windows 需 >= 0.43.0）"),
        ("datasets", "数据集处理"),
        ("gradio", "Web 界面"),
        ("yaml", "配置文件解析（PyYAML）"),
    ]
    missing = []
    for mod, desc in libs:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "未知版本")
            check(f"{mod} ({desc})", True, version)
        except ImportError:
            check(f"{mod} ({desc})", False)
            missing.append(mod)
    if missing:
        print(f"\n  >>> 缺少依赖，请执行：pip install -r requirements.txt")
        print(  "      torch 请先按 CUDA 版本单独安装，见 README.md 第 2 节")

    # ---------- 3. CUDA / GPU ----------
    print("\n[3] CUDA 与 GPU")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            check("CUDA 可用", True, torch.version.cuda)
            check("GPU 型号", True, name)
            check("显存容量", vram >= 8, f"{vram:.1f} GB（4bit 微调建议 >= 8GB）",
                  warn_only=True)
            check("bf16 支持", torch.cuda.is_bf16_supported(),
                  "RTX30 系及以上支持，否则自动降级 fp16", warn_only=True)
        else:
            check("CUDA 可用", False,
                  "未检测到可用 GPU，训练/推理将回退 CPU（极慢）", warn_only=True)
    except ImportError:
        print("  [跳过] torch 未安装，无法检查 CUDA")

    # ---------- 4. bitsandbytes 量化功能 ----------
    print("\n[4] 4bit 量化能力")
    try:
        import torch
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4")
        check("BitsAndBytesConfig", True, f"bnb {bnb.__version__}")
        check("量化算子", True, "可创建（实际算力需 CUDA 环境验证）")
    except Exception as e:
        check("bitsandbytes", False, f"异常：{e}", warn_only=True)

    # ---------- 5. 数据集 ----------
    print("\n[5] 数据与配置文件")
    data_file = PROJECT_ROOT / "data" / "medical_cot_zh.jsonl"
    check("原始数据集", data_file.exists(),
          f"{data_file}（{data_file.stat().st_size/1e6:.1f} MB）"
          if data_file.exists() else "缺失")
    check("训练配置", (PROJECT_ROOT / "configs" / "train_config.yaml").exists())

    processed = PROJECT_ROOT / "data" / "processed" / "train.json"
    check("预处理结果", processed.exists(),
          "已生成，可直接训练" if processed.exists()
          else "未生成，请先运行 data/preprocess.py", warn_only=True)

    # ---------- 6. 磁盘空间 ----------
    print("\n[6] 磁盘剩余空间")
    for label, path in [("项目盘", PROJECT_ROOT), ("系统盘", Path.home())]:
        try:
            free = shutil.disk_usage(path).free / 1024 ** 3
            check(f"{label} 剩余空间", free >= 20, f"{free:.1f} GB"
                  f"（基座模型缓存+训练产物约需 20GB+）", warn_only=True)
        except Exception:
            pass

    print("\n" + "=" * 62)
    print("自检结束。若存在 [缺失] 项，请按 README.md 安装后再运行。")
    print("=" * 62)


if __name__ == "__main__":
    main()
