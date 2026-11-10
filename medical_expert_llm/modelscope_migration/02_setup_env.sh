#!/bin/bash
# ============================================================
# ModelScope 云端训练环境初始化脚本（云端迁移 · 第 2 步）
# ============================================================
# 执行位置：ModelScope Notebook 的终端（Linux）
# 执行方式：bash 02_setup_env.sh
#
# 功能（对应迁移计划第二步）：
#   1. 安装训练依赖 ms-swift（自动带 transformers/peft/datasets）
#   2. 配置 ModelScope SDK Token（当前会话 + 持久化 ~/.bashrc）
#   3. 环境自检：GPU 可见性、依赖版本
#   4. 试拉私有数据集，验证 Token 与仓库访问正常
#
# 说明：Notebook 镜像自带 PyTorch（GPU 版），本脚本不重装 torch，
#       基座模型 Qwen2.5-7B-Instruct 由 swift 训练命令首次运行时
#       自动从 ModelScope Hub 拉取（约 15GB，云端网速快，几分钟）。
# ============================================================

# ---------- 0. 基本配置（按需修改） ----------
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"          # 基座模型（云端 Hub 直接拉取）
DATASET_ID="Huahai123654/medical-cot-zh"     # 第 1 步已上传的数据集 ID（私有）
DATA_DIR="$HOME/medical_data"                # 数据集下载目录

# Token 优先读环境变量；未设置时使用下方占位值（务必替换！）
MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-ms-请替换成你的令牌}"

echo "============================================================"
echo " 第 1 节：安装训练依赖（ms-swift）"
echo "============================================================"
# 锁定 4.5.3：训练脚本 03 的参数已按此版本源码逐项核对；
# 不锁定的话最新版参数可能变动（如 v3 的 train_type 在 4.x 已改名）
pip install -U "ms-swift==4.5.3"
echo ""
echo "关键依赖版本："
pip list 2>/dev/null | grep -iE "^(ms-swift|transformers|peft|torch|datasets|modelscope|accelerate) " || true

echo ""
echo "============================================================"
echo " 第 2 节：配置 ModelScope SDK Token"
echo "============================================================"
if [[ "$MODELSCOPE_API_TOKEN" == "ms-请替换成你的令牌" ]]; then
    echo "[错误] 未配置 Token！二选一："
    echo "  1. 编辑本脚本开头，替换 MODELSCOPE_API_TOKEN 占位值"
    echo "  2. 先执行：export MODELSCOPE_API_TOKEN='ms-你的令牌' 再运行本脚本"
    echo "Token 获取：modelscope.cn -> 头像 -> 访问令牌"
    exit 1
fi
export MODELSCOPE_API_TOKEN

# 登录（凭据写入 SDK 本地配置，供 swift / modelscope CLI 后续使用）
python - <<'PYEOF'
import os
from modelscope.hub.api import HubApi
api = HubApi()
api.login(os.environ["MODELSCOPE_API_TOKEN"])
print("[OK] Token 登录成功")
PYEOF

# 持久化到 ~/.bashrc（新开终端自动生效；已写入则跳过，幂等可重复执行）
if ! grep -q "MODELSCOPE_API_TOKEN" ~/.bashrc 2>/dev/null; then
    echo "export MODELSCOPE_API_TOKEN='$MODELSCOPE_API_TOKEN'" >> ~/.bashrc
    echo "[OK] Token 已写入 ~/.bashrc（新终端自动生效）"
fi

echo ""
echo "============================================================"
echo " 第 3 节：环境自检"
echo "============================================================"
echo "磁盘空间："
df -h "$HOME" | tail -n 1
echo ""
python - <<'PYEOF'
import torch
import transformers, peft, swift, modelscope

print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"ms-swift     : {swift.__version__}")
print(f"modelscope   : {modelscope.__version__}")

assert torch.cuda.is_available(), \
    "[失败] GPU 不可用！请停止实例后选择 GPU 镜像重新启动 Notebook"
print(f"GPU          : {torch.cuda.get_device_name(0)}")
vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"显存         : {vram:.1f} GB")
print("[OK] 环境自检通过")
PYEOF

echo ""
echo "============================================================"
echo " 第 4 节：试拉私有数据集（验证 Token 权限）"
echo "============================================================"
# 私有仓库必须带 Token 才能下载；下载到本地后训练时分别引用
# train.jsonl / val.jsonl，保证训练/验证集严格分开
# 注意：用 Python API 而非 modelscope CLI——部分镜像的 CLI 入口不在 PATH
mkdir -p "$DATA_DIR"
python - <<'PYEOF'
import os
from modelscope import snapshot_download

data_dir = os.path.expanduser("~/medical_data")
os.makedirs(data_dir, exist_ok=True)
snapshot_download(
    repo_id="Huahai123654/medical-cot-zh",
    repo_type="dataset",
    local_dir=data_dir,
    token=os.environ.get("MODELSCOPE_API_TOKEN"),
)
print("[OK] 数据集下载完成")
PYEOF

echo ""
echo "已下载数据文件："
ls -lh "$DATA_DIR"/*.jsonl

python - "$DATA_DIR" <<'PYEOF'
import glob
import json
import sys

data_dir = sys.argv[1]
for path in sorted(glob.glob(f"{data_dir}/*.jsonl")):
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f)
        first = json.loads(f.readline())
    roles = [m["role"] for m in first["messages"]]
    print(f"[OK] {path.split('/')[-1]}: {n} 条，消息角色 {roles}")
PYEOF

echo ""
echo "============================================================"
echo " 环境初始化完成！"
echo "============================================================"
echo "数据集位置 : $DATA_DIR（train.jsonl 19134 条 / val.jsonl 1007 条）"
echo "下一步     : 上传 03_train_cloud.sh 后执行"
echo "             bash 03_train_cloud.sh          # 全量训练"
echo "             bash 03_train_cloud.sh resume   # 断点续训"
echo "注意       : 首次训练会自动下载基座模型 $MODEL_ID（约 15GB）"
echo "============================================================"
