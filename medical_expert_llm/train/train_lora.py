# -*- coding: utf-8 -*-
"""
============================================================
LoRA 4bit 量化微调训练脚本（QLoRA）
============================================================
项目：基于大模型微调的医疗专家
基座：Qwen2.5-7B-Instruct
方案：PEFT + LoRA + BitsAndBytes 4bit 量化（不改动基座权重）

核心原理：
  1. QLoRA：将 7B 基座模型量化为 4bit 加载（显存占用约 5~6GB），
     训练时仅 LoRA 适配器参数以 bf16/fp16 参与梯度计算，
     使 8~16GB 显存的家用显卡也能完成 7B 模型微调。
  2. Loss 屏蔽：只对 assistant（模型应回答的部分）计算交叉熵损失，
     system / user 部分的 label 置为 -100（CrossEntropyLoss 的忽略标记），
     避免模型"学习如何提问"而稀释"学习如何回答"。

用法（在项目根目录执行）：
  python train/train_lora.py                          # 按 configs/train_config.yaml 全量训练
  python train/train_lora.py --max_train_samples 200  # 快速跑通流程（调试）
  python train/train_lora.py --num_train_epochs 1     # 命令行覆盖配置文件
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

# ------------------------- 路径常量 -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"

# 系统提示词：必须与 data/preprocess.py 中写入训练数据的提示词完全一致，
# 保证模型学到"在该角色设定下作答"
SYSTEM_PROMPT = (
    "你是一位经验丰富的医疗AI助手，具备中西医结合的临床知识。"
    "请针对用户提出的健康问题进行解答。回答时请先进行逐步的医学推理分析，"
    "再给出专业、严谨、有条理的结论。"
    "注意：你的回答仅供健康参考，不能替代执业医师的现场诊断与治疗建议。"
)


# ------------------------- 配置加载 -------------------------

def load_config() -> dict:
    """读取 YAML 配置文件，返回配置字典。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_argparser(config: dict) -> argparse.ArgumentParser:
    """
    构建命令行参数：default=None 表示"未指定则使用 yaml 中的值"，
    指定后覆盖 yaml 配置（方便实验调参而不修改配置文件）。
    """
    parser = argparse.ArgumentParser(description="医疗专家模型 LoRA 微调训练")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型名称或本地路径（覆盖配置文件）")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="预处理数据目录（含 train.json / val.json）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true", default=None,
                        help="强制 4bit 量化训练")
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false",
                        help="关闭量化（仅显存充裕时使用）")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="仅训练前 N 条（快速调试）")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="仅验证前 N 条")
    return parser


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    """用命令行参数覆盖 yaml 配置，得到最终生效配置。"""
    final = dict(config)
    for key, value in vars(args).items():
        if value is not None and key in final:
            final[key] = value
    return final


# ------------------------- 数据处理 -------------------------

def tokenize_with_loss_mask(tokenizer, sample: dict, max_length: int):
    """
    将一条 messages 样本编码为 (input_ids, labels)，并做 loss 屏蔽。

    关键技巧：
      full   = chat_template( system + user + assistant )   # 完整序列
      prefix = chat_template( system + user, add_generation_prompt=True )
               # 只到 "<|im_start|>assistant\n" 为止
      则 assistant 部分在序列中的位置 = full[len(prefix):]
      将该位置之前的 labels 全部置为 -100，即可实现"只学回答不学提问"。
    """
    messages = sample["messages"]

    # 编码完整对话（带 <|im_end|> 等特殊标记）
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    # 编码前缀（system + user + 生成提示符，即 assistant 开头之前）
    prefix_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )

    # 超长保护：理论上预处理已过滤，此处兜底截断（保留末尾回答）
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    # 构造 labels：前缀部分（提问）不计损失，仅 assistant 回答计损失
    labels = [-100] * min(len(prefix_ids), len(full_ids)) + \
             full_ids[min(len(prefix_ids), len(full_ids)):]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ------------------------- 数据整理器 -------------------------

class MedicalDataCollator:
    """
    动态 Padding 数据整理器：
      同一 batch 内按最长序列补齐（而非全局定长），显著减少无效计算。
      input_ids 补 pad_token_id，attention_mask 补 0，labels 补 -100（不计损失）。
      补齐长度取 8 的倍数，提高 GPU 张量核心利用率。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        # 向上取整到 8 的倍数
        max_len = (max_len + 7) // 8 * 8

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(
                list(f["input_ids"]) + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad_len)
            labels.append(list(f["labels"]) + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------- 低显存 Trainer -------------------------

class LowMemoryTrainer(Trainer):
    """
    8GB 显存优化版 Trainer：
      Qwen2.5 词表高达 15 万，官方 ForCausalLMLoss 会把整条序列的
      logits [B, L, 151k] 全部转成 fp32 再计算交叉熵，仅此一项峰值
      就超过 1GB。本类只对 labels != -100 的有效回答 token 计算损失
      （提问部分本来就不计损失），显存峰值大幅下降且数值结果等价。
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 自回归移位：第 i 个位置的 logits 预测第 i+1 个 token
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        def _masked_ce(sl, sy, n_items):
            # 仅取 assistant 有效 token（labels != -100）参与损失计算
            mask = sy != -100
            valid_logits = sl[mask]
            valid_labels = sy[mask]
            if valid_labels.numel() == 0:
                return sl.sum() * 0.0  # 空 batch 保护（正常数据不会出现）
            loss_sum = F.cross_entropy(
                valid_logits.float(), valid_labels, reduction="sum")
            if n_items is not None and n_items > 0:
                # 梯度累积下按全局有效 token 数平均（与官方实现口径一致）
                return loss_sum / n_items
            return loss_sum / valid_labels.numel()

        # 训练时用梯度检查点包裹 CE：前向不保留 gather/fp32 中间量
        # （省约 0.7GB 峰值显存），反向时重算一次，代价可忽略
        if self.model.training and shift_logits.requires_grad:
            loss = torch.utils.checkpoint.checkpoint(
                _masked_ce, shift_logits, shift_labels, num_items_in_batch,
                use_reentrant=False)
        else:
            loss = _masked_ce(shift_logits, shift_labels, num_items_in_batch)

        return (loss, outputs) if return_outputs else loss


# ------------------------- 工具函数 -------------------------

def offload_embeddings_to_cpu(model):
    """
    8GB 显存优化：把 embed_tokens（约 1GB，且 LoRA 不挂载它）移到 CPU，
    并挂 forward hook 自动完成 GPU->CPU->GPU 的数据搬运。
    每步仅传输约 7MB 的隐向量，性能损耗可忽略，却能为训练腾出 1GB 显存。
    注：不能在 from_pretrained 的 device_map 里直接指定 CPU，
    否则 bitsandbytes 量化加载会报 "Some modules are dispatched on the CPU"。
    """
    emb = model.get_input_embeddings()
    if emb is None or not emb.weight.is_cuda:
        return
    emb.to("cpu")

    def _pre_hook(module, args, kwargs):
        # 输入 token id 移到 CPU 完成 embedding 查表
        return (args[0].to("cpu"),), kwargs

    def _post_hook(module, args, output):
        # 查表结果移回 GPU 供后续层使用
        return output.to("cuda:0")

    emb.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    emb.register_forward_hook(_post_hook)
    print("  [显存优化] embed_tokens 已 offload 到 CPU（省约 1GB 显存）")


def offload_lm_head_to_cpu(model):
    """
    8GB 显存显存优化（补充）：把 lm_head（约 1GB，未被量化且 LoRA 不挂载）
    也移到 CPU。logits 矩阵乘在 CPU 计算，每步增加数秒耗时，
    但换来 1GB 显存余量，使 8GB 显卡上 7B QLoRA 训练成为可能。
    hook 机制下 .to() 属于可导操作，反向梯度会自动跨设备回传。
    """
    lm = model.get_output_embeddings()
    if lm is None or not lm.weight.is_cuda:
        return
    # fp16 在 CPU 上矩阵乘极慢，转为 fp32 后由 MKL 多线程加速
    lm.to(device="cpu", dtype=torch.float32)

    def _pre_hook(module, args, kwargs):
        # hidden 移到 CPU 并对齐 lm_head 权重精度（bf16 autocast 输出
        # 需转 fp32 才能与之相乘）
        h = args[0].to("cpu")
        if h.dtype != module.weight.dtype:
            h = h.to(module.weight.dtype)
        return (h,), kwargs

    def _post_hook(module, args, output):
        return output.to("cuda:0")

    lm.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    lm.register_forward_hook(_post_hook)
    print("  [显存优化] lm_head 已 offload 到 CPU（再省约 1GB 显存，"
          "每步增加数秒 CPU 计算耗时）")


# ------------------------- 主流程 -------------------------

def main():
    config = load_config()
    args = build_argparser(config).parse_args()
    cfg = merge_config(config, args)

    # ---- 打印最终生效配置 ----
    print("=" * 62)
    print("最终训练配置：")
    for k in ("base_model", "load_in_4bit", "data_dir", "output_dir",
              "num_train_epochs", "learning_rate",
              "per_device_train_batch_size", "gradient_accumulation_steps",
              "max_length", "lora_r", "lora_alpha"):
        print(f"  {k:32s} = {cfg[k]}")
    print("=" * 62)

    # ---- 环境自检 ----
    if not torch.cuda.is_available():
        print("[警告] 未检测到可用 CUDA，训练将在 CPU 上进行（极慢，仅限调试）")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[GPU] {gpu_name}，显存 {vram_gb:.1f} GB")

    # 是否支持 bf16（RTX30 系及以上；GTX16/Ti 系需降级 fp16）
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"[精度] 计算精度：{'bfloat16' if bf16_supported else 'float16'}")

    # ---- 1. 加载分词器 ----
    print("\n[1/6] 加载分词器 ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    # Qwen 已有 eos/pad 定义；若缺失则补齐，防止 collator 报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- 2. 4bit 量化加载基座模型 ----
    print("\n[2/6] 加载基座模型（量化方式：4bit NF4）...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],          # nf4 精度更高
        bnb_4bit_compute_dtype=dtype,                             # 计算精度
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],  # 双重量化省显存
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quant_config,
        device_map={"": 0} if cfg["load_in_4bit"] else "auto",
        trust_remote_code=True,
    )
    # 训练阶段禁用 KV Cache（与梯度检查点冲突）
    model.config.use_cache = False

    # embed_tokens / lm_head 移到 CPU 共省约 2GB 显存（LoRA 不挂载这两层）
    if cfg["load_in_4bit"]:
        offload_embeddings_to_cpu(model)
        offload_lm_head_to_cpu(model)

    # 4bit 模型训练前准备：冻结参数 + 梯度检查点。
    # 注意：不调用 peft.prepare_model_for_kbit_training——它会把未被量化的
    # embed_tokens / lm_head 升为 fp32（约 +5GB 显存），8GB 显卡必然 OOM；
    # LoRA 训练只需冻结基座并由 get_peft_model 注入 fp32 的低秩分支即可。
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ---- 3. 注入 LoRA 适配器 ----
    print("\n[3/6] 注入 LoRA 适配器 ...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,       # 因果语言模型任务
        r=cfg["lora_r"],                    # 低秩维度
        lora_alpha=cfg["lora_alpha"],       # 缩放系数
        lora_dropout=cfg["lora_dropout"],   # dropout
        target_modules=cfg["lora_target_modules"],  # 挂载到全部线性投影层
        bias="none",                        # 不训练偏置项
    )
    model = get_peft_model(model, lora_config)
    # 打印可训练参数量对比：LoRA 通常仅占基座 0.5%~1%
    model.print_trainable_parameters()

    # ---- 4. 加载并编码数据集 ----
    print("\n[4/6] 加载并编码数据集 ...")
    data_dir = PROJECT_ROOT / cfg["data_dir"]
    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"
    if not train_file.exists():
        raise FileNotFoundError(
            f"未找到 {train_file}，请先运行: python data/preprocess.py")

    # 注意：文件名虽为 .json，实际是 JSONL 格式（每行一条）
    raw_datasets = load_dataset("json", data_files={
        "train": str(train_file), "validation": str(val_file),
    })

    def tokenize_fn(sample):
        return tokenize_with_loss_mask(tokenizer, sample, cfg["max_length"])

    train_ds = raw_datasets["train"].map(
        tokenize_fn, remove_columns=raw_datasets["train"].column_names,
        desc="编码训练集",
    )
    val_ds = raw_datasets["validation"].map(
        tokenize_fn, remove_columns=raw_datasets["validation"].column_names,
        desc="编码验证集",
    )
    # 只保留 Trainer 需要的三列
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(columns=cols)
    val_ds.set_format(columns=cols)

    # 快速调试：截取子集
    if cfg.get("max_train_samples"):
        train_ds = train_ds.select(range(min(cfg["max_train_samples"],
                                              len(train_ds))))
    if cfg.get("max_eval_samples"):
        val_ds = val_ds.select(range(min(cfg["max_eval_samples"], len(val_ds))))
    print(f"  训练集 {len(train_ds)} 条 / 验证集 {len(val_ds)} 条")

    # ---- 5. 配置训练参数并启动训练 ----
    print("\n[5/6] 启动训练 ...")
    output_dir = PROJECT_ROOT / cfg["output_dir"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,

        # 训练超参
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        lr_scheduler_type=cfg["lr_scheduler_type"],

        # 混合精度（按显卡能力自动选择，CPU 上自动关闭）
        bf16=bf16_supported,
        fp16=(not bf16_supported) and torch.cuda.is_available(),

        # 梯度检查点：用时间换显存，8GB 显卡跑 7B 的关键开关
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # 8bit 分页优化器：进一步降低优化器状态显存占用
        optim="paged_adamw_8bit",

        # 日志 / 评估 / 保存策略
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        metric_for_best_model="eval_loss",   # 以验证损失挑选最优模型
        greater_is_better=False,
        load_best_model_at_end=True,

        # Windows 兼容：禁用多进程数据加载，避免子进程崩溃/卡死
        dataloader_num_workers=0,
        # 关闭外部实验追踪（wandb 等），避免弹窗要求登录
        report_to="none",
        seed=cfg["seed"],
    )

    callbacks = []
    if cfg.get("use_early_stopping", True):
        # 早停：连续 patience 次评估 loss 未下降则终止，防过拟合
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping_patience"]))

    trainer = LowMemoryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=MedicalDataCollator(tokenizer),
        callbacks=callbacks,
    )

    train_result = trainer.train()

    # ---- 6. 保存最终 LoRA 适配器 ----
    print("\n[6/6] 保存最终 LoRA 适配器 ...")
    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # 记录训练摘要，便于撰写毕设论文/开发文档
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    metrics["eval_samples"] = len(val_ds)
    with open(output_dir / "train_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print("训练完成！关键产物：")
    print(f"  LoRA 适配器权重 : {final_dir}")
    print(f"  训练指标        : {output_dir / 'train_results.json'}")
    print("下一步：")
    print("  1) 命令行测试推理: python inference/predict.py "
          "--lora_path outputs/lora_medical_qwen/final_adapter")
    print("  2) 合并权重     : python inference/merge_lora.py")
    print("  3) 启动问诊网页 : python app/app.py")
    print("=" * 62)
    print(f"\n最终 train_loss = {metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    main()
