# -*- coding: utf-8 -*-
"""诊断脚本：查看 4bit 量化加载后各模块的设备分布与显存占用"""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "models/qwen2.5-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)

print("\n=== hf_device_map ===")
for k, v in model.hf_device_map.items():
    print(f"  {k:40s} -> {v}")

print("\n=== 参数分布 ===")
device_totals = {}
for name, p in model.named_parameters():
    mb = p.numel() * p.element_size() / 1024 / 1024
    key = str(p.device)
    device_totals[key] = device_totals.get(key, 0) + mb
    if "embed" in name or "lm_head" in name or "layers.0." in name or name.endswith(".norm"):
        print(f"  {name:50s} {str(p.device):8s} {str(p.dtype):15s} {mb:8.0f} MB")

print("\n=== 每设备合计 ===")
for k, v in device_totals.items():
    print(f"  {k}: {v:.2f} MB")

print("\n=== GPU 显存 ===")
print(f"  allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"  reserved : {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
free, total = torch.cuda.mem_get_info(0)
print(f"  free/total: {free/1024**3:.2f} / {total/1024**3:.2f} GB")
