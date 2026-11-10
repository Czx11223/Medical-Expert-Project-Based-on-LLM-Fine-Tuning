# -*- coding: utf-8 -*-
"""
============================================================
ModelScope 数据集上传脚本（云端迁移 · 第 1 步）
============================================================
功能：
  1. 数据准备：校验 data/processed/ 下 train.json / val.json 的每一行
     JSON 合法性，转换为 ModelScope 数据集标准格式（jsonl，每行一条
     {"messages": [...]} 的 ChatML 样本，ms-swift 可直接加载）；
  2. 生成数据集卡片 README.md（字段结构、规模、用途说明）；
  3. 上传：HubApi Token 认证 -> 创建数据集仓库 -> 上传数据文件。

Token 认证（按优先级读取）：
  1. 环境变量 MODELSCOPE_API_TOKEN（推荐：Token 不写入任何文件）
  2. 命令行 --token 参数

Token 获取方式：modelscope.cn -> 右上角头像 ->「访问令牌」-> 创建

用法（在项目根目录 medical_expert_llm/ 下执行）：
  # 仅本地准备+校验数据（不联网、无需 Token）
  python modelscope_migration/01_upload_dataset.py --namespace 用户名 --prepare-only

  # 完整流程：准备 + 创建仓库 + 上传
  python modelscope_migration/01_upload_dataset.py --namespace 用户名
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "processed"
UPLOAD_DIR = Path(__file__).resolve().parent / "dataset_upload"

DATASET_CARD_TMPL = """# 中文医疗思维链（CoT）问答数据集

本数据集用于大模型医疗领域 LoRA 微调（毕业设计项目：基于大模型微调的医疗专家）。

## 数据规模

| 文件 | 条数 | 用途 |
|---|---|---|
| train.jsonl | {n_train} | 训练集 |
| val.jsonl | {n_val} | 验证集 |

## 数据格式

每行一条 JSON（jsonl），为标准 ChatML messages 结构，
可直接被 ms-swift / swift sft 加载：

```json
{{
  "messages": [
    {{"role": "system",    "content": "你是一位经验丰富的医疗AI助手……"}},
    {{"role": "user",      "content": "患者提问……"}},
    {{"role": "assistant", "content": "【医学推理】\\n思维链……\\n\\n【专业解答】\\n结论……"}}
  ]
}}
```

- 来源：中文医疗思维链数据集（20171 条原始样本，剔除超长样本后划分）
- 划分：随机打乱（seed=42）后按 95% / 5% 划分训练/验证集
- assistant 回复为「医学推理 + 专业解答」双段 CoT 外显结构

## 用途声明

数据仅供学习研究用途，不构成任何医疗建议。
"""


def parse_args():
    parser = argparse.ArgumentParser(description="上传医疗CoT数据集到 ModelScope")
    parser.add_argument("--namespace", type=str, required=True,
                        help="你的 ModelScope 用户名（数据集将创建在该账号下）")
    parser.add_argument("--dataset_name", type=str, default="medical-cot-zh",
                        help="数据集仓库名（默认 medical-cot-zh）")
    parser.add_argument("--token", type=str, default=None,
                        help="ModelScope SDK Token（优先读环境变量 MODELSCOPE_API_TOKEN）")
    parser.add_argument("--prepare_only", action="store_true",
                        help="仅本地准备数据，不执行上传")
    parser.add_argument("--private", action="store_true",
                        help="创建私有数据集（默认公开）")
    return parser.parse_args()


def prepare_data() -> tuple:
    """
    校验 + 转换：train.json / val.json -> dataset_upload/train.jsonl / val.jsonl
    返回 (训练条数, 验证条数)。任何一行 JSON 非法都会立即报错中止。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val"):
        src = SRC_DIR / f"{split}.json"
        dst = UPLOAD_DIR / f"{split}.jsonl"
        if not src.exists():
            sys.exit(f"[错误] 未找到 {src}，请先运行 data/preprocess.py")
        n, bad = 0, 0
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8", newline="\n") as fout:
            for i, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    # 格式硬校验：必须是 messages 列表且含 assistant 回复
                    roles = [m.get("role") for m in sample["messages"]]
                    assert "assistant" in roles, "缺少 assistant 消息"
                except Exception as e:
                    bad += 1
                    sys.exit(f"[错误] {src.name} 第 {i} 行校验失败: {e}")
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
        counts[split] = n
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"[准备] {split}.jsonl：{n} 条，{size_mb:.1f} MB")
        if bad:
            sys.exit(f"[错误] {split} 有 {bad} 行非法，已中止")
    return counts["train"], counts["val"]


def write_readme(n_train: int, n_val: int):
    card = DATASET_CARD_TMPL.format(n_train=n_train, n_val=n_val)
    path = UPLOAD_DIR / "README.md"
    path.write_text(card, encoding="utf-8")
    print(f"[准备] 数据集卡片：{path}")


def upload(args, token: str):
    """HubApi 认证 -> 创建数据集仓库 -> 上传 train/val/README 三个文件。"""
    from modelscope.hub.api import HubApi

    repo_id = f"{args.namespace}/{args.dataset_name}"

    # ---- 1. Token 认证 ----
    print(f"[上传] 正在登录 ModelScope ...")
    api = HubApi()
    api.login(token)

    # ---- 2. 创建数据集仓库（已存在则跳过，幂等）----
    visibility = 5 if args.private else 1  # 1=公开 5=私有
    print(f"[上传] 创建数据集仓库 {repo_id}（visibility={'私有' if args.private else '公开'}）...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        visibility=visibility,
        exist_ok=True,   # 已存在不报错，便于重复执行
        token=token,
    )

    # ---- 3. 逐个上传文件 ----
    files = ["README.md", "train.jsonl", "val.jsonl"]
    for name in files:
        local = UPLOAD_DIR / name
        print(f"[上传] {name}（{local.stat().st_size / 1024 / 1024:.1f} MB）...")
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local),
            path_in_repo=name,
            token=token,
        )

    print("\n" + "=" * 62)
    print("数据集上传完成！")
    print(f"  数据集 ID : {repo_id}")
    print(f"  页面地址  : https://modelscope.cn/datasets/{repo_id}")
    print(f"  云端训练引用（swift sft 的 --dataset 参数）:")
    print(f"    --dataset {repo_id}")
    print("=" * 62)


def main():
    args = parse_args()

    # ---- 数据准备与校验（本地操作，无需网络）----
    print("=" * 62)
    print(" 第 1 步：数据准备与格式校验")
    print("=" * 62)
    n_train, n_val = prepare_data()
    write_readme(n_train, n_val)

    if args.prepare_only:
        print("\n[完成] --prepare_only 模式：数据已就绪于 "
              f"{UPLOAD_DIR}，未执行上传。")
        return

    # ---- Token 解析：环境变量优先 ----
    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        sys.exit(
            "[错误] 未找到 ModelScope Token。二选一：\n"
            "  1. 设置环境变量（推荐）：$env:MODELSCOPE_API_TOKEN='ms-xxxxxxxx'\n"
            "  2. 命令行传入：--token ms-xxxxxxxx\n"
            "Token 获取：modelscope.cn -> 头像 -> 访问令牌 -> 创建令牌")

    print("=" * 62)
    print(" 第 2 步：上传到 ModelScope Hub")
    print("=" * 62)
    upload(args, token)


if __name__ == "__main__":
    main()
