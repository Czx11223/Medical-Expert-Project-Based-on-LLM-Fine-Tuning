# -*- coding: utf-8 -*-
"""
============================================================
数据预处理脚本：medical_cot_zh.jsonl -> Qwen ChatML 训练格式
============================================================
功能：
  1. 读取原始数据集 data/medical_cot_zh.jsonl（约 2 万条中文医疗思维链样本）
  2. 清洗数据：过滤缺失字段 / 过长样本，规范化空白字符
  3. 将每条 (Question, Complex_CoT, Response) 三元组转换为 Qwen
     的多轮对话 messages 格式：
        system    : 医疗专家角色设定（与推理端保持一致，训练/推理必须同源）
        user      : Question（患者提问）
        assistant : 【医学推理】CoT 思维链 + 【专业解答】最终回答
  4. 随机打乱后按比例划分训练集 / 验证集（默认 95% / 5%）
  5. 输出 data/processed/train.json 和 data/processed/val.json

用法（在项目根目录执行）：
  python data/preprocess.py                         # 全量处理
  python data/preprocess.py --val_ratio 0.05        # 指定验证集比例
  python data/preprocess.py --max_samples 500       # 只取前 500 条（快速调试）
"""

import argparse
import json
import random
import re
from pathlib import Path

# ------------------------- 全局常量 -------------------------

# 项目根目录（本文件位于 data/ 下，故取上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据集路径
RAW_DATA_PATH = PROJECT_ROOT / "data" / "medical_cot_zh.jsonl"

# 预处理输出目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 系统提示词：角色设定 + 输出格式约束 + 安全提醒
# 【重要】该提示词与 train/train_lora.py 训练数据、app/app.py 推理端
#          必须保持完全一致，否则会造成训练/推理分布不一致（train-serving skew）
SYSTEM_PROMPT = (
    "你是一位经验丰富的医疗AI助手，具备中西医结合的临床知识。"
    "请针对用户提出的健康问题进行解答。回答时请先进行逐步的医学推理分析，"
    "再给出专业、严谨、有条理的结论。"
    "注意：你的回答仅供健康参考，不能替代执业医师的现场诊断与治疗建议。"
)

# assistant 回复模板：显式区分"思维链"与"最终回答"两段，
# 让模型在推理时同样先输出推理过程再输出结论（CoT 思维链外显）
ASSISTANT_TEMPLATE = "【医学推理】\n{cot}\n\n【专业解答】\n{response}"

# 按字符数粗过滤的阈值：中文 1 token ≈ 1.5 字符，
# 1600 字符 ≈ 1100 token + 模板开销 < max_length(1536)，训练时无需截断
MAX_TOTAL_CHARS = 1600


# ------------------------- 工具函数 -------------------------

def normalize_text(text: str) -> str:
    """规范化文本：去除首尾空白、压缩连续 3 个以上的换行为 2 个。"""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_raw_data(path: Path):
    """逐行读取 JSONL 原始数据，跳过损坏行。"""
    records = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                bad_lines += 1
                print(f"  [警告] 第 {line_no} 行 JSON 解析失败，已跳过")
    return records, bad_lines


def convert_sample(rec: dict):
    """
    将一条原始记录转换为 Qwen messages 格式。
    返回 None 表示该样本不合法（缺字段）或过长，将被过滤。
    """
    question = rec.get("Question") or rec.get("question")
    cot = rec.get("Complex_CoT") or rec.get("complex_cot") or rec.get("cot")
    response = rec.get("Response") or rec.get("response")

    # 三字段任一缺失即丢弃
    if not (question and cot and response):
        return None

    question = normalize_text(question)
    cot = normalize_text(cot)
    response = normalize_text(response)

    # 按总字符数过滤超长样本（避免训练时截断破坏回答完整性）
    if len(question) + len(cot) + len(response) > MAX_TOTAL_CHARS:
        return None

    assistant = ASSISTANT_TEMPLATE.format(cot=cot, response=response)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
    }


def save_jsonl(samples, path: Path):
    """将样本列表写为 JSONL 文件（每行一个 JSON 对象）。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  已保存 {len(samples):>6} 条 -> {path}")


# ------------------------- 主流程 -------------------------

def main():
    parser = argparse.ArgumentParser(description="医疗CoT数据集预处理")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH),
                        help="原始 JSONL 数据路径")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录（生成 train.json / val.json）")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="验证集比例，默认 0.05（5%%）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多处理条数（调试用），默认全量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证划分可复现")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # 1. 读取原始数据
    print("=" * 60)
    print("步骤 1/4：读取原始数据集")
    print("=" * 60)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到数据集文件：{input_path}")
    records, bad_lines = load_raw_data(input_path)
    print(f"  读取完成：{len(records)} 条（损坏行 {bad_lines} 条）")

    # 调试模式：截取前 N 条
    if args.max_samples is not None:
        records = records[: args.max_samples]
        print(f"  [调试] 仅使用前 {len(records)} 条")

    # 2. 清洗并转换为 messages 格式
    print("=" * 60)
    print("步骤 2/4：清洗数据并转换为 Qwen 对话格式")
    print("=" * 60)
    samples, filtered = [], 0
    total_chars = 0
    for rec in records:
        s = convert_sample(rec)
        if s is None:
            filtered += 1
            continue
        samples.append(s)
        # 统计 assistant 长度用于打印均值
        total_chars += len(s["messages"][2]["content"])
    print(f"  转换成功 {len(samples)} 条，过滤 {filtered} 条（缺字段或超长）")
    if samples:
        avg = total_chars / len(samples)
        print(f"  assistant 平均长度：{avg:.0f} 字符（CoT+解答）")

    # 3. 打乱并划分训练/验证集
    print("=" * 60)
    print("步骤 3/4：打乱并划分数据集")
    print("=" * 60)
    random.seed(args.seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  训练集 {len(train_samples)} 条 / 验证集 {len(val_samples)} 条 "
          f"(验证比例 {args.val_ratio:.0%})")

    # 4. 保存结果
    print("=" * 60)
    print("步骤 4/4：保存处理结果")
    print("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_samples, output_dir / "train.json")
    save_jsonl(val_samples, output_dir / "val.json")

    # 打印一条样例供人工核对格式
    print("\n[样例预览] 第 1 条训练样本：")
    preview = json.dumps(train_samples[0], ensure_ascii=False, indent=2)
    print(preview[:1200] + ("\n...(截断)" if len(preview) > 1200 else ""))
    print("\n预处理全部完成！下一步：python train/train_lora.py 启动微调训练")


if __name__ == "__main__":
    main()
