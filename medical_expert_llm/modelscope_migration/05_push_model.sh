#!/bin/bash
# ============================================================
# LoRA 合并 + 推送 ModelScope Hub 脚本（云端迁移 · 第 5 步）
# ============================================================
# 执行位置：ModelScope Notebook 终端（需已完成 03_train_cloud.sh 训练）
# 用法：bash 05_push_model.sh
#
# 流程：
#   1. 定位训练输出目录中最新的 checkpoint-*（最终 LoRA 适配器）
#   2. swift export 合并 LoRA 到原始精度基座 -> 完整独立模型
#   3. 生成模型卡片 README.md（模型描述/训练数据/超参/用法示例）
#   4. 推送到你的 ModelScope 模型仓库
#      （create_repo 幂等 + push_model，可重复执行覆盖更新）
#
# 推送成功后的本地回迁方案（见脚本末尾打印）：
#   A. 下载完整合并模型（约 15GB）-> inference/predict.py --merged_model
#   B. 仅下载 LoRA 适配器（约 100MB）-> 配本地已有基座 --lora_path 加载
# ============================================================

set -o pipefail

# ---------- 0. 配置 ----------
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="$HOME/output/medical_expert_swift"        # 训练输出目录（03 脚本产物）
MERGED_DIR="$HOME/output/medical_expert_merged"       # 合并后完整模型目录
MODEL_REPO="Huahai123654/medical-expert-qwen2.5-7b"   # 推送目标模型仓库
DATASET_ID="Huahai123654/medical-cot-zh"              # 训练数据集（写入模型卡片）
TOKEN="${MODELSCOPE_API_TOKEN:-}"

if [[ -z "$TOKEN" || "$TOKEN" == "ms-请替换成你的令牌" ]]; then
    echo "[错误] 未检测到 Token，请先执行 02_setup_env.sh 或"
    echo "       export MODELSCOPE_API_TOKEN='ms-你的令牌'"
    exit 1
fi
export MODELSCOPE_API_TOKEN="$TOKEN"

# ---------- 1. 定位最优 LoRA 适配器 ----------
# 03 脚本设了 --add_version false：产物直接位于输出目录根部（无 v*-* 版本层）
# 优先级：训练结束保存到根部的最终适配器（多跑完最后 96 步）> 最新 checkpoint-*
if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" ]]; then
    CKPT="$OUTPUT_DIR"
    echo "[1/4] 使用训练结束时保存的最终 LoRA 适配器：$CKPT"
elif [[ -n "$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null)" ]]; then
    CKPT=$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n 1)
    echo "[1/4] 使用最新 checkpoint 的 LoRA 适配器：$CKPT"
else
    echo "[错误] $OUTPUT_DIR 下未找到任何适配器"
    echo "       （根部无 adapter_model.safetensors，也无 checkpoint-* 目录）"
    echo "       请确认训练已完成：bash 03_train_cloud.sh"
    exit 1
fi

# ---------- 2. 合并 LoRA（swift 以原始精度加载基座做代数合并） ----------
# 与 03 脚本同理：swift 命令不在 PATH 时改用 python -m 模块方式调用
SWIFT_EXPORT=(swift export)
if ! command -v swift >/dev/null 2>&1; then
    SWIFT_EXPORT=(python -m swift.cli.export)
fi

echo "[2/4] 合并 LoRA 权重到基座（需要几分钟，CPU/显存占用正常）..."
"${SWIFT_EXPORT[@]}" \
    --model "$BASE_MODEL" \
    --adapters "$CKPT" \
    --merge_lora true \
    --output_dir "$MERGED_DIR" \
    --exist_ok true \
    2>&1 | tee -a "$HOME/export_swift.log"

if [[ ! -f "$MERGED_DIR/config.json" ]]; then
    echo "[错误] 合并产物不完整（缺少 config.json），查看 $HOME/export_swift.log"
    exit 1
fi
echo "合并产物："
ls -lh "$MERGED_DIR" | head -n 20

# ---------- 3. 生成模型卡片 ----------
cat > "$MERGED_DIR/README.md" <<'EOF'
# Medical Expert Qwen2.5-7B（医疗专家模型 · LoRA 微调版）

基于 **Qwen2.5-7B-Instruct** 的中文医疗领域 LoRA 微调模型（QLoRA 4bit 训练）。
针对用户健康提问，模型按「医学推理 → 专业解答」思维链（CoT）结构输出
专业、有条理的医疗参考回答。

## 训练信息

| 项目 | 内容 |
|---|---|
| 基座模型 | Qwen/Qwen2.5-7B-Instruct |
| 训练数据集 | [Huahai123654/medical-cot-zh](https://modelscope.cn/datasets/Huahai123654/medical-cot-zh)（中文医疗 CoT，19134 条训练 / 1007 条验证） |
| 训练框架 | ms-swift（swift sft） |
| 微调方式 | LoRA + QLoRA（bitsandbytes 4bit NF4 + double quant） |
| 计算精度 | BF16 混合精度 |

## 训练超参

| 超参 | 取值 |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| target_modules | all-linear |
| epoch | 1 |
| learning rate | 2e-4（cosine 衰减，warmup 5%） |
| batch size x grad accum | 2 x 8（有效批大小 16） |
| max_length | 1536 |
| save / eval steps | 100 |

## 用法示例

### swift 命令行推理（推荐）

```bash
pip install -U ms-swift
swift infer --model Huahai123654/medical-expert-qwen2.5-7b \
    --infer_backend pt \
    --stream true
```

### transformers + peft 加载（基座 + LoRA 适配器方式）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "Huahai123654/medical-expert-qwen2.5-7b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

## 声明

本模型仅供学习研究与健康教育参考，输出不构成医疗建议，
不能替代执业医师的现场诊断与治疗。
EOF
echo "[3/4] 模型卡片已生成：$MERGED_DIR/README.md"

# ---------- 4. 推送到 ModelScope Hub ----------
echo "[4/4] 推送模型仓库 $MODEL_REPO（约 15GB，视网速几分钟到十几分钟）..."
python - <<PYEOF
import os
from modelscope.hub.api import HubApi

repo_id = "$MODEL_REPO"
token = os.environ["MODELSCOPE_API_TOKEN"]
merged_dir = "$MERGED_DIR"

api = HubApi()
api.login(token)
# 仓库不存在则创建（已存在则跳过），幂等可重复执行
api.create_repo(repo_id=repo_id, repo_type="model", visibility=1,
                exist_ok=True, token=token)
# 推送整个合并模型目录（含权重与模型卡片）
api.push_model(model_id=repo_id, model_dir=merged_dir)
print("[OK] 模型推送完成")
PYEOF

echo ""
echo "============================================================"
echo " 模型已推送到 ModelScope Hub！"
echo "============================================================"
echo " 模型 ID   : $MODEL_REPO"
echo " 页面地址  : https://modelscope.cn/models/$MODEL_REPO"
echo ""
echo " 回迁本地（二选一）："
echo "  A. 完整合并模型（约 15GB）："
echo "     modelscope download --model $MODEL_REPO --local_dir models/medical-expert-qwen"
echo "     -> 本地推理：python inference/predict.py --merged_model models/medical-expert-qwen"
echo "     -> WebUI   ：app/app.py 自动优先加载该目录"
echo ""
echo "  B. 仅 LoRA 适配器（约 100MB，复用本地已有的 15GB 基座）："
echo "     从模型页下载 adapter_model.safetensors / adapter_config.json"
echo "     -> python inference/predict.py --lora_path <适配器目录>"
echo "============================================================"
