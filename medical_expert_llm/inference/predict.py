# -*- coding: utf-8 -*-
"""
============================================================
模型推理测试脚本（命令行交互）
============================================================
支持两种加载方式：
  方式 A：基座模型 + LoRA 适配器（推荐日常测试，省磁盘、可随时切换）
  方式 B：加载 merge_lora.py 合并后的完整模型

用法示例（在项目根目录执行）：
  # 方式 A：加载基座 + LoRA 适配器（4bit 量化推理）
  python inference/predict.py ^
      --base_model Qwen/Qwen2.5-7B-Instruct ^
      --lora_path outputs/lora_medical_qwen/final_adapter

  # 方式 B：加载合并后的完整模型
  python inference/predict.py --merged_model outputs/medical_expert_merged

  # 单条问题测试（非交互模式）
  python inference/predict.py --lora_path ... --question "最近总是失眠怎么办？"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 系统提示词：与训练数据（data/preprocess.py）保持完全一致，
# 保证推理时的输入分布与训练分布对齐
SYSTEM_PROMPT = (
    "你是一位经验丰富的医疗AI助手，具备中西医结合的临床知识。"
    "请针对用户提出的健康问题进行解答。回答时请先进行逐步的医学推理分析，"
    "再给出专业、严谨、有条理的结论。"
    "注意：你的回答仅供健康参考，不能替代执业医师的现场诊断与治疗建议。"
)

# 默认模型路径（若存在则免传参）
DEFAULT_BASE = str(PROJECT_ROOT / "models" / "qwen2.5-7b-instruct")
DEFAULT_LORA = PROJECT_ROOT / "outputs" / "lora_medical_qwen" / "final_adapter"
DEFAULT_MERGED = PROJECT_ROOT / "outputs" / "medical_expert_merged"


def load_model(args):
    """
    按优先级加载模型：
      merged_model > (base_model + lora_path) > base_model 裸模型
    推理默认使用 4bit 量化（显存友好，质量损失极小）。
    """
    # 自动发现默认产物，减少手动传参
    if not args.merged_model and DEFAULT_MERGED.exists():
        args.merged_model = str(DEFAULT_MERGED)
    if not args.lora_path and DEFAULT_LORA.exists():
        args.lora_path = str(DEFAULT_LORA)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    ) if args.load_in_4bit else None

    if args.merged_model:
        # ---- 方式 B：加载合并后的完整模型 ----
        print(f"[加载] 合并模型：{args.merged_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.merged_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # ---- 方式 A：基座 + LoRA 适配器 ----
        print(f"[加载] 基座模型：{args.base_model}")
        print(f"[加载] LoRA 适配器：{args.lora_path or '（未指定，加载原始基座）'}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.lora_path:
            # 将 LoRA 适配器挂载到基座上（基座权重不变）
            model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()  # 推理模式，关闭 dropout
    model.config.use_cache = True  # 推理时启用 KV Cache 加速
    return model, tokenizer


def build_prompt(tokenizer, question: str) -> str:
    """将用户问题 + 系统提示词套用 Qwen ChatML 模板，拼成模型输入。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()  # 推理不需要梯度，省显存加速
def generate_answer(model, tokenizer, question: str,
                    max_new_tokens: int = 1024,
                    temperature: float = 0.7) -> str:
    """生成回答：采样参数采用 Qwen 官方推荐组合。"""
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.8,                # 核采样：限制累计概率前 80% 的词
        top_k=20,                 # 只在概率前 20 的词中采样
        repetition_penalty=1.05,  # 抑制重复（医疗长文本必备）
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
        or tokenizer.eos_token_id,
    )
    # 只解码新生成的部分（去掉输入 prompt）
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="医疗专家模型推理测试")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE,
                        help="基座模型（方式 A）")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 适配器目录（方式 A）")
    parser.add_argument("--merged_model", type=str, default=None,
                        help="合并后的完整模型目录（方式 B，优先级最高）")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="4bit 量化推理（默认开启，省显存）")
    parser.add_argument("--no_4bit", dest="load_in_4bit",
                        action="store_false", help="关闭量化")
    parser.add_argument("--question", type=str, default=None,
                        help="单条提问（不传则进入交互模式）")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print("=" * 60)
    print(" 医疗专家模型 - 命令行推理测试")
    print("=" * 60)
    model, tokenizer = load_model(args)

    if args.question:
        # 单条问答模式
        answer = generate_answer(model, tokenizer, args.question,
                                  args.max_new_tokens, args.temperature)
        print("\n[问题]", args.question)
        print("\n[回答]\n", answer)
        return

    # 交互模式：循环问答，输入 exit/quit 退出
    print("\n进入交互模式（输入 exit 或 quit 退出）\n")
    while True:
        try:
            question = input("患者> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break
        print("专家> 正在思考 ...")
        answer = generate_answer(model, tokenizer, question,
                                 args.max_new_tokens, args.temperature)
        print("专家>", answer, "\n")


if __name__ == "__main__":
    main()
