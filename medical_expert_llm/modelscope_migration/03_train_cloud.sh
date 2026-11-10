#!/bin/bash
# ============================================================
# ModelScope 云端 LoRA 微调训练脚本（云端迁移 · 第 3/4 步）
# ============================================================
# 对应本地 train/train_lora.py 的 ms-swift 改造版
#   - 基座模型：ModelScope ID 云端拉取（--model Qwen/Qwen2.5-7B-Instruct）
#   - 数据集  ：第 1 步上传的 Huahai123654/medical-cot-zh（02 脚本已下载到本地）
#   - 训练方式：swift sft + QLoRA（bnb 4bit NF4）+ LoRA
#   - 精度    ：BF16 混合精度
#   - 注意力  ：自动检测 flash-attn，装了就启用，没装回退 sdpa
#
# 参数已按 ms-swift 4.5.3 源码逐项核对（tuner_type / report_to /
# resume_from_checkpoint 均为 4.x 写法，与网上大量 v3 教程不同）
#
# 执行位置：ModelScope Notebook 终端（需先完成 02_setup_env.sh）
# 用法：
#   bash 03_train_cloud.sh            # 从头开始全量训练
#   bash 03_train_cloud.sh resume     # 断点续训（从最近 checkpoint 恢复）
#
# 监控方式：
#   - 终端实时输出 loss（每 10 步打印一次）
#   - TensorBoard：训练结束后另开终端执行
#     tensorboard --logdir ~/output/medical_expert_swift/runs --host 0.0.0.0
#     然后在 Notebook 端口面板打开 TensorBoard 入口
#
# 显存不足自动降级：
#   检测到 CUDA OOM 时自动 batch 减半、累积步数翻倍（有效批大小不变），
#   最低降到 batch=1+accum=16+max_length=1024，仍失败则提示更换实例。
# ============================================================

set -o pipefail

# ---------- 0. 训练配置（按需修改） ----------
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
DATA_DIR="$HOME/medical_data"
OUTPUT_DIR="$HOME/output/medical_expert_swift"
LOG_FILE="$HOME/train_swift.log"

# 训练超参（有效批大小 = BATCH x ACCUM，调整时保持乘积 16 不变）
EPOCHS=1
LR=2e-4
BATCH=2
ACCUM=8
MAX_LEN=1536

# LoRA 超参（与本地 train/train_lora.py 完全一致）
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 监控与存档（全量 19134 条 ≈ 1196 个优化步，约每 1.7 小时存档一次）
LOG_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=100
SAVE_LIMIT=3

# 可选：训练后端。peft（默认，稳定）/ unsloth（更快，需先 pip install unsloth）
TUNER_BACKEND="${TUNER_BACKEND:-peft}"

MODE="${1:-new}"   # new: 从头训练 | resume: 断点续训
CUDA_VISIBLE_DEVICES=0

# ---------- 1. 前置检查 ----------
if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/val.jsonl" ]]; then
    echo "[错误] 未找到数据集 $DATA_DIR/train.jsonl，请先执行：bash 02_setup_env.sh"
    exit 1
fi

# ---------- 2. 确定 swift 命令入口 ----------
# 部分镜像的 console_scripts 不在 PATH（如 modelscope CLI），做双重保险
SWIFT_RUN=(swift sft)
if ! command -v swift >/dev/null 2>&1; then
    SWIFT_RUN=(python -m swift.cli.sft)
    echo "[提示] swift 命令不在 PATH，改用 python -m swift.cli.sft"
fi

# ---------- 3. 检测 Flash Attention ----------
ATTN_IMPL="sdpa"
if python -c "import flash_attn" 2>/dev/null; then
    ATTN_IMPL="flash_attn"
    echo "[OK] 检测到 flash-attn，启用 FlashAttention 加速"
else
    echo "[提示] 未安装 flash-attn，使用 sdpa 注意力（稳定）。"
    echo "        如需加速，可先执行：pip install flash-attn --no-build-isolation"
fi

# ---------- 3. 组装断点续训参数 ----------
# ms-swift 的 --resume_from_checkpoint 只接受真实存在的 checkpoint 路径，
# 因此这里自动定位输出目录下最新的 checkpoint-* 再传入
RESUME_ARGS=()
if [[ "$MODE" == "resume" ]]; then
    LATEST_CKPT=$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n 1)
    if [[ -n "$LATEST_CKPT" ]]; then
        RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CKPT")
        echo "[续训] 从最近 checkpoint 恢复：$LATEST_CKPT"
    else
        echo "[提示] $OUTPUT_DIR 下没有 checkpoint，将从头开始训练"
    fi
fi

# ---------- 4. 训练函数 ----------
run_train() {
    "${SWIFT_RUN[@]}" \
        --model "$MODEL_ID" \
        --tuner_type lora \
        --tuner_backend "$TUNER_BACKEND" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --target_modules all-linear \
        --quant_method bnb \
        --quant_bits 4 \
        --bnb_4bit_quant_type nf4 \
        --bnb_4bit_use_double_quant true \
        --bnb_4bit_compute_dtype bfloat16 \
        --torch_dtype bfloat16 \
        --bf16 true \
        --attn_impl "$ATTN_IMPL" \
        --dataset "$DATA_DIR/train.jsonl" \
        --val_dataset "$DATA_DIR/val.jsonl" \
        --max_length "$MAX_LEN" \
        --truncation_strategy delete \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size "$BATCH" \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps "$ACCUM" \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --gradient_checkpointing true \
        --logging_steps "$LOG_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_LIMIT" \
        --output_dir "$OUTPUT_DIR" \
        --add_version false \
        --report_to tensorboard \
        --dataloader_num_workers 4 \
        "${RESUME_ARGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
}

# ---------- 5. 启动训练（含 OOM 自动降级重试） ----------
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo ""
    echo "============================================================"
    echo " 第 $attempt 次启动 | batch=$BATCH accum=$ACCUM max_len=$MAX_LEN"
    echo " attn=$ATTN_IMPL backend=$TUNER_BACKEND 模式=$MODE"
    echo "============================================================"

    if run_train; then
        echo ""
        echo "============================================================"
        echo " 训练完成！"
        echo "============================================================"
        echo "输出目录   : $OUTPUT_DIR（LoRA 适配器在各 checkpoint-* 子目录）"
        echo "训练日志   : $LOG_FILE"
        echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR/runs --host 0.0.0.0"
        echo "下一步     : 上传 05_push_model.sh 并执行 bash 05_push_model.sh"
        echo "             （合并 LoRA 并推送到你的 ModelScope 模型仓库）"
        echo "============================================================"
        exit 0
    fi

    # ---------- 失败原因分析与自动重试 ----------
    if grep -qiE "out of memory|OutOfMemoryError|CUDA out of memory" "$LOG_FILE"; then
        echo "[OOM] 检测到显存不足，自动降级配置后重试..."
        if [[ "$BATCH" -gt 1 ]]; then
            BATCH=$((BATCH / 2))
            ACCUM=$((ACCUM * 2))
        elif [[ "$MAX_LEN" -gt 1024 ]]; then
            MAX_LEN=1024
        else
            echo "[失败] 已降至最低配置（batch=1, accum=16, max_len=1024）仍 OOM，"
            echo "       请停止实例并更换更大显存的 GPU 规格后重试。"
            exit 1
        fi
        # 降级后从已有 checkpoint 续训，避免浪费已完成的进度
        RESUME_ARGS=()
        LATEST_CKPT=$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -n 1)
        if [[ -n "$LATEST_CKPT" ]]; then
            RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CKPT")
            echo "[OOM] 降级后将从 checkpoint 续训：$LATEST_CKPT"
        fi
    elif [[ "$ATTN_IMPL" == "flash_attn" ]]; then
        echo "[提示] 非 OOM 失败且当前使用 flash_attn，回退到 sdpa 重试一次..."
        ATTN_IMPL="sdpa"
    else
        echo "[失败] 训练异常退出（非显存问题），完整日志见 $LOG_FILE"
        exit 1
    fi
done
