# -*- coding: utf-8 -*-
"""
============================================================
LoRA 权重合并脚本：基座模型 + LoRA 适配器 -> 完整独立模型
============================================================
为什么需要合并？
  - LoRA 适配器只是"增量权重"（几十~几百MB），推理时需要基座在场；
  - 合并后得到一个独立的完整模型（约 14GB, fp16），可单独部署、
    直接传给 vLLM/Ollama 等推理框架，也方便毕设打包演示。
  - 注意：合并必须在"非量化"的基座上进行（fp16/bf16），否则无法
    将低秩矩阵运算固化回原始权重。

用法（在项目根目录执行）：
  python inference/merge_lora.py                          # 使用默认路径
  python inference/merge_lora.py --lora_path outputs/lora_medical_qwen/final_adapter ^
      --base_model Qwen/Qwen2.5-7B-Instruct --output_dir outputs/medical_expert_merged

合并产物（约 15GB 磁盘空间）：
  outputs/medical_expert_merged/
      config.json / model-*.safetensors / tokenizer.json 等
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE = str(PROJECT_ROOT / "models" / "qwen2.5-7b-instruct")
DEFAULT_LORA = PROJECT_ROOT / "outputs" / "lora_medical_qwen" / "final_adapter"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "medical_expert_merged"


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE,
                        help="基座模型名称或本地路径")
    parser.add_argument("--lora_path", type=str, default=str(DEFAULT_LORA),
                        help="LoRA 适配器目录")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT),
                        help="合并后完整模型的保存目录")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="合并保存精度（默认 float16，兼容性最好）")
    args = parser.parse_args()

    lora_path = Path(args.lora_path)
    output_dir = Path(args.output_dir)

    # ---- 前置检查 ----
    if not lora_path.exists():
        raise FileNotFoundError(
            f"未找到 LoRA 适配器：{lora_path}\n"
            f"请先完成训练（python train/train_lora.py）")
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[警告] 输出目录 {output_dir} 已存在且非空，文件将被覆盖")

    dtype = (torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)

    # ---- 1. 以全精度在 CPU 上加载基座（保证合并数值正确）----
    # 不做 4bit 量化：量化权重无法与 LoRA 增量做代数合并
    print(f"[1/3] 以 {args.dtype} 精度在 CPU 加载基座：{args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="cpu",          # CPU 上合并，避免显存不足
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)

    # ---- 2. 挂载 LoRA 并合并权重 ----
    print(f"[2/3] 挂载并合并 LoRA 适配器：{lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    # merge_and_unload：W' = W + (alpha/r) * B @ A，随后移除适配器包装
    model = model.merge_and_unload()
    model.config.use_cache = True

    # ---- 3. 保存完整模型 ----
    print(f"[3/3] 保存合并后的完整模型：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "=" * 60)
    print("合并完成！")
    print(f"  完整模型目录：{output_dir}")
    print("验证方式：")
    print(f"  python inference/predict.py --merged_model {output_dir}")
    print(f"  python app/app.py                 # Web 界面会自动优先加载该模型")
    print("=" * 60)


if __name__ == "__main__":
    main()
