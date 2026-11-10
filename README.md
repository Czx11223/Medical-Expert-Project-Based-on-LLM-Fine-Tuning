# 基于大模型微调的医疗专家

> 个人独立项目 · 本地可运行的医疗问诊大模型系统
>
> 基座 **Qwen2.5-7B-Instruct** + **PEFT/LoRA** + **BitsAndBytes 4bit 量化（QLoRA）**，
> 使用 2 万条中文医疗思维链（CoT）数据微调，提供 **新版 FastAPI + SSE 流式问诊界面** 与 **Gradio 旧版界面** 两套 Web 端。

---

## 1. 项目简介

### 1.1 要解决的问题

| 行业痛点 | 本项目的应对 |
|---|---|
| 医疗资源分布不均，基层误诊率高 | 本地部署医疗专家模型，低门槛辅助问诊 |
| 通用大模型医疗回答不专业、无推理过程 | 医疗 CoT 数据微调，回答"先推理后结论" |
| 全量微调 7B 模型需要 A100 级显卡 | QLoRA 4bit 量化，8GB 显存家用显卡即可训练 |

### 1.2 技术架构

```
medical_cot_zh.jsonl (2万条中文医疗CoT数据)
        │  data/preprocess.py（清洗 + ChatML格式化 + 划分训练/验证集）
        ▼
Qwen2.5-7B-Instruct (4bit NF4 量化加载)
        │  train/train_lora.py (PEFT + LoRA, 仅训练约0.5%参数)
        │  —— 或 ——  ms-swift（ModelScope 云端全量训练）
        ▼
outputs/lora_medical_qwen/final_adapter (LoRA适配器)
        │  inference/merge_lora.py (可选：合并为完整模型)
        ▼
┌───────────────────────────────┬──────────────────────────────┐
│ app/server.py (FastAPI + SSE) │ app/app.py (Gradio)          │
│ + app/static/index.html       │ 7860 端口 · 旧版界面          │
│ 8080 端口 · 新版界面（推荐）    │                              │
└───────────────────────────────┴──────────────────────────────┘
```

### 1.3 效果说明

微调后模型针对健康提问会按照以下结构作答（训练数据格式决定）：

```
【医学推理】
（逐步的医学思维链分析过程……）
【专业解答】
（专业、有条理的最终结论……）
```

### 1.4 两套 Web 界面对比

| 项 | 新版（推荐） | 旧版 |
|---|---|---|
| 入口 | `scripts\06_webui_new.bat` | `scripts\04_webui.bat` / 根目录 `启动问诊.bat` |
| 后端 | FastAPI + SSE（`app/server.py`） | Gradio（`app/app.py`） |
| 地址 | http://127.0.0.1:8080 | http://127.0.0.1:7860 |
| 前端 | 单文件 `app/static/index.html`（39KB，无构建步骤） | Gradio 内置组件 |
| 特点 | 侧栏多会话（浏览器本地存储）、示例问题、CoT 分段渲染、复制/重新回答/停止生成、temperature 与 max tokens 滑杆、医疗免责声明常驻 | 多轮对话、流式输出、示例问题、参数调节 |
| 额外依赖 | `fastapi`、`uvicorn`（**未写入 requirements.txt，需手动装，见 2.3**） | `gradio>=5.0` |

> 两套界面共用同一份模型加载逻辑（优先合并模型 → 其次基座 + LoRA → 最后裸基座），可同时保留，按演示需要选择启动。

---

## 2. 环境安装（Windows）

### 2.1 硬件要求

| 配置 | 显存 | 说明 |
|---|---|---|
| 推荐训练配置 | ≥ 16GB (RTX 4080/3090) | 默认参数，2万条×3轮约 4~8 小时 |
| 基础训练配置 | ≥ 12GB (RTX 3060/4070) | batch=1，约 10~15 小时 |
| **最低训练配置（已实测跑通）** | 8GB (RTX 4060 Laptop) | batch=1 + 累积16 + max_length≤1024 + CPU offload |
| 仅推理/Web演示 | ≥ 6GB | 4bit 加载 LoRA 或合并模型 |
| 无显卡 | — | 只能跑预处理，训练不可行 |

磁盘需求：基座模型缓存约 15GB + LoRA 输出约 0.16GB（合并完整模型另需 15GB）。

> **8GB 显卡说明**：`train/train_lora.py` 已内置两级显存优化——
> ① embed_tokens / lm_head 自动 offload 到 CPU（共省约 2GB 显存，代价是每步多几秒 CPU 计算）；
> ② 损失计算启用梯度检查点。实测 RTX 4060 Laptop（8GB）在
> `batch=1 + 累积16 + max_length=768` 下训练全程稳定无 OOM。

### 2.2 软件要求

- Windows 10 / 11（64 位）
- Python 3.10 或 3.11（bitsandbytes Windows 版要求 ≥ 3.10）
- NVIDIA 驱动 + CUDA 11.8 / 12.1（只需驱动，不必装完整 CUDA Toolkit）
- 建议使用国内 pip 镜像加速

### 2.3 安装步骤（依次执行）

```powershell
# 0. 进入项目根目录
cd d:\peixunwenjian\Medical-Expert-Project-Based-on-LLM-Fine-Tuning\medical_expert_llm

# 1. （推荐）创建独立虚拟环境，避免污染系统 Python
python -m venv venv
.\venv\Scripts\activate

# 2. 先按显卡 CUDA 版本安装 PyTorch（二选一，必须单独装！）
# CUDA 12.1（驱动较新的机器）：
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8（老机器）：
pip install torch --index-url https://download.pytorch.org/whl/cu118
# 不确定版本：控制面板 -> NVIDIA 控制面板 -> 系统信息 查看 CUDA 版本

# 3. 安装其余依赖（使用国内镜像更快）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 【新版界面必装】requirements.txt 尚未收录 Web 服务依赖，手动补装这两个
pip install "fastapi>=0.110" "uvicorn>=0.29" -i https://pypi.tuna.tsinghua.edu.cn/simple
#    不装这两项时 app/server.py 会直接 ImportError；仅用 Gradio 旧版则无需安装

# 5. 环境自检（确认 GPU / 依赖 / 数据就绪）
python scripts\00_check_env.py
```

> `scripts/00_check_env.py` 的依赖检查清单目前只覆盖 transformers / peft / bitsandbytes / gradio 等，
> **不含 fastapi / uvicorn**，所以第 4 步装没装它不会报错，请自行确认。

### 2.4 基座模型下载（三选一）

首次训练会自动从 HuggingFace 下载 Qwen2.5-7B-Instruct（约 15GB）。国内网络建议提前用下面任一方式下载：

**方式 A：HuggingFace 镜像（免安装）**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"   # 当前 PowerShell 会话生效
python train\train_lora.py                    # 随后自动从镜像下载
```

**方式 B：ModelScope 魔搭社区（国内速度最快）**

```powershell
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models\qwen2.5-7b-instruct
```

**方式 C：串行断点续传脚本（网络不稳时最省心）**

```powershell
# 先改脚本内 $dir 变量为你的实际项目路径（当前硬编码为旧机器路径）
notepad models\download_weights.ps1
.\models\download_weights.ps1
```

该脚本用 curl 逐片下载 4 个 safetensors 分片，带 `-C -` 断点续传、20KB/s 低速自动断线重连、`--ssl-no-revoke` 规避证书吊销检查失败。

`configs/train_config.yaml` 的 `base_model` **已默认为本地路径** `models/qwen2.5-7b-instruct`，按上述方式下载到该目录即可直接训练；若沿用 HuggingFace 在线缓存，改成仓库名 `Qwen/Qwen2.5-7B-Instruct`。

---

## 3. 一键运行流程

每个步骤都有对应的 bat 脚本，**双击 `scripts\` 下对应文件即可**。
脚本已固定使用项目 venv 的 Python，与系统 PATH 无关（找不到 venv 会给出提示）。

**演示/日常使用只需记一个入口**：

- 新版界面（推荐）：双击 `scripts\06_webui_new.bat` → 浏览器打开 http://127.0.0.1:8080
- 旧版界面：双击根目录 `启动问诊.bat` → 浏览器自动打开 http://127.0.0.1:7860

| 步骤 | 双击脚本 | 等价命令 | 产物 |
|---|---|---|---|
| 0 环境自检 | `scripts\00_check_env.bat` | `venv\Scripts\python scripts\00_check_env.py` | 自检报告 |
| 1 数据预处理 | `scripts\01_preprocess.bat` | `venv\Scripts\python data\preprocess.py` | `data\processed\train.json / val.json` |
| 2 LoRA 微调训练 | `scripts\02_train.bat` | `venv\Scripts\python train\train_lora.py` | `outputs\lora_medical_qwen\final_adapter` |
| 3 合并权重（可选） | `scripts\03_merge.bat` | `venv\Scripts\python inference\merge_lora.py` | `outputs\medical_expert_merged` |
| 4 旧版问诊网页 | `scripts\04_webui.bat` | `venv\Scripts\python app\app.py` | http://127.0.0.1:7860（Gradio） |
| 5 命令行测试 | `scripts\05_predict.bat` | `venv\Scripts\python inference\predict.py` | 交互式问答 |
| **6 新版问诊网页** | `scripts\06_webui_new.bat` | `venv\Scripts\python app\server.py --port 8080` | http://127.0.0.1:8080（FastAPI+SSE） |

> 首次跑通建议：`python train\train_lora.py --max_train_samples 200 --max_eval_samples 20`
> 先用 200 条样本验证全流程无报错，再进行全量训练。

### 3.1 本机实测验证记录（RTX 4060 Laptop / 8GB / Win11）

以下环节已在本机完成端到端验证，可放心复现：

| 环节 | 命令 | 结果 |
|---|---|---|
| 环境安装 | venv + torch cu121 + requirements.txt | 全部依赖就绪（numpy=2.4.6、sentencepiece=0.2.0） |
| 基座下载 | ModelScope → `models\qwen2.5-7b-instruct` | 15GB 完整（4 个 safetensors 分片） |
| 数据预处理 | `python data\preprocess.py` | 20141 条有效 → train 19134 / val 1007 |
| 快速训练 | `--max_train_samples 200 --max_eval_samples 20 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_length 768` | 12 步完成，train_loss=1.995，耗时 1098.7s，全程无 OOM |
| Gradio 推理 | `python app\app.py` | 7860 端口正常，模型输出【医学推理】+【专业解答】CoT 结构 |
| 新版前端 | `python app\server.py --port 8080` | 8080 端口正常，`/api/health` 返回模型状态，`/api/chat` SSE 流式逐字输出 |

### 3.2 本地全量训练参考命令（8GB 显存）

> **状态提示**：全量训练已于 2026-09-10 在 ModelScope 云端 GPU 完成（1,196 步 / 约 4 小时），正式权重已回迁至 `outputs\lora_medical_qwen\final_adapter`。本节命令保留作本地复现参考，无需再执行。详见 `docs\云端训练工作日志.md`。

本地全量训练建议过夜执行：

```powershell
# 方案A（推荐，约 5 小时）：子集 3000 条 x 1 epoch
python train\train_lora.py --max_train_samples 3000 ^
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_length 1024

# 方案B（效果更好，周末挂机）：全量 19134 条 x 1 epoch，约 30~33 小时
python train\train_lora.py ^
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --max_length 1024
```

> 训练时长估算依据：实测约 90 秒/优化步（batch=1 x 累积16，含 CPU offload 开销，电脑空闲时）。
> 方案A：3000 ÷ 16 = 187 步 ≈ 5 小时；方案B：19134 ÷ 16 ≈ 1196 步 ≈ 30 小时。
> 每 80 步保存一次 checkpoint（本地约 2 小时一个），中断可从 checkpoint 恢复；
> 同时按 eval_loss 自动挑选最优 checkpoint 作为最终模型。
> 训练完成后 `outputs\lora_medical_qwen\final_adapter` 会自动更新为正式权重，
> 两套 WebUI 均无需改参数，重启即加载新模型。

### 3.3 云端全量训练（已完成 · 2026-09-10）

本地 8GB 显卡实测约 90 秒/优化步，1,196 步约需 30 小时且三次因电源管理 / 显存中断。
全量训练已迁移 ModelScope 云端 GPU 完成，正式权重已回迁本地，本节为结果摘要。

| 项目 | 结果 |
|---|---|
| 框架 / 精度 | ms-swift 4.5.3 + QLoRA 4bit NF4 + BF16 + Flash Attention 2 |
| 超参 | 1 epoch / lr 2e-4 / batch 1 × accum 16 / max_length 1024 / 7 类线性层 |
| 规模与耗时 | 19,134 条 = 1,196 步，11.9 秒/步 ≈ **3.95 小时**（本地约 30.4 小时，加速约 7.7 倍） |
| 状态 | 全程无 OOM，正常结束 |
| 产物 | LoRA 适配器 161.5 MB（154 MiB）→ `outputs\lora_medical_qwen\final_adapter`；Hub `Huahai123654/medical-expert-lora` |
| 验证 | CLI 推理 / Gradio WebUI(7860) / 新版前端(8080) / API 端到端多重通过（严格【医学推理】→【专业解答】双段结构） |

- 旧 200 条快速版权重已备份为 `final_adapter_quick200`；`train_results.json` **仅对应 200 条快速训练**（train_samples=200、train_loss=1.995、耗时 1098.7s），保留为流程验证证据，未被云端全量结果覆盖。
- `outputs\cloud_adapter\` 为云端回迁时的临时目录，当前为空，可忽略或删除。
- 本地 `loss mask / 动态 Padding / 早停 / CPU offload` 属快速验证链路；云端全量使用 ms-swift 内置流程，汇报时不要混讲。
- 完整排障记录、回迁可核验证据与后续建议见 `docs\云端训练工作日志.md`；复现流程见 `modelscope_migration\云端操作指引.md`。

---

## 4. 各模块详细说明

### 4.1 数据预处理（data/preprocess.py）

- 读取 `medical_cot_zh.jsonl`（20171 条，字段：`Question` / `Complex_CoT` / `Response`）
- 清洗：过滤缺失字段样本、超长样本（> 1600 字符）；实测过滤 30 条，保留 20141 条
- 转换为 Qwen ChatML 多轮对话格式：

```json
{"messages": [
  {"role": "system",    "content": "你是一位经验丰富的医疗AI助手…"},
  {"role": "user",      "content": "患者提问……"},
  {"role": "assistant", "content": "【医学推理】…思维链…\n\n【专业解答】…最终回答…"}
]}
```

- 随机打乱后 95% / 5% 划分：训练集 19134 条，验证集 1007 条（seed=42 可复现）
- 常用参数：`--val_ratio 0.05`、`--max_samples 500`（调试）

### 4.2 LoRA 微调训练（train/train_lora.py）

关键设计（写项目文档可直接引用）：

1. **QLoRA 4bit 量化**：基座以 NF4 量化加载（约 5.6GB 显存），配双重量化；
2. **LoRA 参数高效微调**：`r=16, alpha=32, dropout=0.05`，挂载 Qwen 全部 7 类线性投影层（q/k/v/o/gate/up/down），可训练参数 40.37M（约 0.5%），基座权重冻结不变；
3. **Loss 屏蔽**：只对 assistant 回答部分计算交叉熵（user/system 的 label 置 -100），模型专注学"怎么答"而不是"怎么问"；
4. **动态 Padding**：batch 内按最长序列补齐（对齐到 8 的倍数），减少无效计算；
5. **梯度检查点 + 8bit 分页优化器**：进一步压缩显存峰值；
6. **早停机制**：验证集 loss 连续 3 次不下降自动停止，防过拟合。

默认超参（`configs/train_config.yaml`）：

| 超参 | 默认值 | 说明 |
|---|---|---|
| num_train_epochs | 1 | 轮数；时间充裕可加到 2~3 |
| learning_rate | 2e-4 | LoRA 典型学习率 |
| per_device_train_batch_size | 2 | 8GB 显卡改 1 |
| gradient_accumulation_steps | 8 | 等效 batch=16 |
| max_length | 1536 | 数据集实测最长 ~1400 token，全覆盖 |
| lr_scheduler_type | cosine | 余弦退火 |
| warmup_ratio | 0.03 | 预热比例 |

命令行参数可覆盖配置文件，例如：

```powershell
python train\train_lora.py --num_train_epochs 1 --per_device_train_batch_size 1
```

训练产物：
- `outputs/lora_medical_qwen/final_adapter/`：LoRA 适配器权重（adapter_model.safetensors，约 154 MiB）
- `outputs/lora_medical_qwen/checkpoint-*/`：过程检查点（保留最优 3 个）
- `outputs/lora_medical_qwen/train_results.json`：训练指标汇总（写论文用；当前内容为 200 条快速版）

### 4.3 推理（inference/predict.py）

两种加载方式自动选择（优先合并模型）：

```powershell
# 方式A：基座 + LoRA 适配器（默认，4bit 推理约 6GB 显存）
python inference\predict.py --lora_path outputs\lora_medical_qwen\final_adapter

# 方式B：合并后的完整模型
python inference\predict.py --merged_model outputs\medical_expert_merged

# 单条测试
python inference\predict.py --question "晚上失眠多梦怎么办？"
```

生成参数采用 Qwen 官方推荐：`temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05`。

### 4.4 合并权重（inference/merge_lora.py，可选）

在 CPU 上以 fp16 加载基座 → 挂载 LoRA → `merge_and_unload()` 代数合并 → 保存完整模型（约 15GB）。合并后可脱离基座独立部署，也方便传给 vLLM 等推理框架。**磁盘紧张可跳过，两套 WebUI 均支持直接加载 LoRA。**

### 4.5 新版问诊界面（app/server.py + app/static/index.html）

**推荐界面**。FastAPI 托管单文件前端，SSE 流式返回，无需 npm 构建。

```powershell
venv\Scripts\python app\server.py                # 默认 127.0.0.1:8080
venv\Scripts\python app\server.py --port 8081    # 换端口
```

接口约定：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回 `app/static/index.html` |
| GET | `/api/health` | 模型就绪状态、CUDA 可用性、是否正在生成（`busy`） |
| POST | `/api/chat` | SSE 流式问答 |
| GET | `/static/*` | 静态资源挂载 |

`/api/chat` 请求体与流式响应：

```json
// 请求
{"message": "最近失眠多梦怎么办？",
 "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}],
 "temperature": 0.7, "max_new_tokens": 1024}

// 响应（text/event-stream，每帧 data: {...}\n\n）
data: {"text": "【医学推理】"}
data: {"text": "…"}
data: {"done": true}
// 或异常：data: {"error": "..."}
```

实现要点：
- 系统提示词与训练数据（`data/preprocess.py`）**完全一致**，改动会破坏 CoT 输出格式；
- 历史只保留最近 6 轮（12 条消息）、单条内容截断 4000 字符，防超上下文；
- `GEN_LOCK` 单卡串行生成，避免并发把 8GB 显存挤爆；
- 生成在独立线程中执行，主线程消费 `TextIteratorStreamer` 逐字下发，支持前端"停止生成"；
- 模型加载优先级：`outputs/medical_expert_merged` → 基座 + `final_adapter` → 裸基座（并提示"未找到 LoRA"）。

前端功能：侧栏多会话（localStorage 本地保存，标注"会话仅保存在本机浏览器"）、示例问题一键提问、CoT 双段样式化渲染、复制 / 重新回答 / 有帮助 / 停止生成、temperature 与 max tokens 滑杆、急症提示与医疗免责声明。

### 4.6 旧版问诊界面（app/app.py，Gradio）

- 医疗问答：多轮对话上下文（自动携带最近 6 轮）、流式逐字输出；
- 内置 4 个示例问题一键填入；
- 可调 temperature / max_new_tokens；支持"重新回答"、"清空对话"；
- 页脚保留一行 AI 生成内容提示小字；
- 启动：`python app\app.py --port 7860`（自动打开浏览器）。

---

## 5. Windows 常见问题（FAQ）

| 现象 | 原因与解决 |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` / `'uvicorn'` | 新版前端依赖未安装（requirements.txt 未收录）。`pip install "fastapi>=0.110" "uvicorn>=0.29"`；只用 Gradio 旧版可忽略 |
| `torch.cuda.is_available()` 返回 False | 装成了 CPU 版 torch。卸载后从官方源重装：`pip uninstall torch` → `pip install torch --index-url https://download.pytorch.org/whl/cu121`（国内可加 `-i https://mirror.sjtu.edu.cn/pytorch-wheels/cu121` 镜像） |
| `module 'numpy._globals' has no attribute '_signature_descriptor'` | numpy 多版本残留导致 ABI 冲突。`pip uninstall numpy -y`（反复执行到无残留）→ `pip install numpy==2.4.6` |
| 导入 sentencepiece 时 Python 直接崩溃退出 | Windows 下 0.2.2 的 C 扩展有 bug。锁定 `sentencepiece==0.2.0` |
| 下载模型报 SSL 证书吊销检查失败 | 部分网络环境无法访问 CRL。curl 加 `--ssl-no-revoke`（`models\download_weights.ps1` 已内置），或改用 ModelScope |
| Gradio 启动报 `TypeError: argument of type 'bool' is not iterable` | 旧版 API 序列化 bug。升级 `pip install "gradio>=5.0"` |
| `No module named 'bitsandbytes'` / 安装失败 | Windows 需 bitsandbytes ≥ 0.43.0 且 Python ≥ 3.10；升级 pip 后重装：`python -m pip install -U pip` |
| 训练时报 CUDA out of memory | ① `configs/train_config.yaml` 中 `per_device_train_batch_size` 改 1；② `gradient_accumulation_steps` 改 16；③ `max_length` 改 1024 |
| 下载模型超时 / 连不上 huggingface.co | 设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"`；或用 ModelScope / `models\download_weights.ps1` 下载到本地（见 2.4） |
| 控制台中文乱码 | bat 脚本已内置 `chcp 65001`；手动跑命令时先执行 `chcp 65001` |
| DataLoader 卡死 / 卡在训练开始 | 已内置 `dataloader_num_workers=0` 规避 Windows 多进程问题；若仍出现，重启 Python 后重试 |
| 训练很慢 | 正常现象：8GB 显卡上每步约 90 秒（CPU offload 的代价）；请按 3.2 节选择子集训练或走云端 |
| 训练日志出现 `expandable_segments not supported on this platform` 警告 | Windows 平台限制，无害，可忽略 |
| `torch.cuda.is_bf16_supported()` 为 False | GTX 16 系等老显卡不支持 bf16，脚本已自动降级 fp16，无需处理 |
| 新版页面打不开 / 空白 | 确认 8080 未被占用：`python app\server.py --port 8081`；先访问 `/api/health` 确认模型已加载 |
| 新版页面提问后无逐字输出、最后一次性蹦出 | 中间有反向代理/杀软缓冲了 SSE。直连 `127.0.0.1:8080`，勿经过会缓冲的代理 |
| Gradio 页面打不开 | 检查 7860 端口占用：`python app\app.py --port 7861` 换端口；浏览器手动访问 http://127.0.0.1:7860 |
| 合并模型时磁盘不足 | 合并产物约 15GB，可跳过合并，WebUI 直接加载 LoRA 适配器 |
| 保存 checkpoint 报路径过长 | Windows 260 字符限制；将项目移到浅目录（如 `d:\medical_expert_llm`），或启用系统长路径支持 |

---

## 6. 项目结构总览

```
medical_expert_llm/
├── README.md                        # 本文档
├── 启动问诊.bat                      # 旧版一键入口（Gradio，等价 04_webui）
├── requirements.txt                 # Python 依赖清单（torch 需单独安装；fastapi/uvicorn 未收录，见 2.3）
├── .gitignore                       # git 忽略规则（模型/输出不入库）
├── configs/
│   └── train_config.yaml            # 训练总配置（模型/LoRA/超参/保存策略）
├── data/
│   ├── medical_cot_zh.jsonl         # 原始数据集（2万条中文医疗CoT，49.9MB）
│   ├── preprocess.py                # 数据预处理脚本（已验证：20141条有效）
│   └── processed/                   # 预处理产物（train.json 55.8MB / val.json 2.9MB，不入库）
├── train/
│   └── train_lora.py                # LoRA 4bit 微调训练脚本（核心）
├── inference/
│   ├── predict.py                   # 命令行推理测试（LoRA 或合并模型）
│   └── merge_lora.py                # LoRA 权重合并脚本
├── app/
│   ├── server.py                    # 【新版】FastAPI + SSE 问诊服务（8080）
│   ├── static/
│   │   └── index.html               # 【新版】单文件问诊前端（39KB，无构建）
│   └── app.py                       # 【旧版】Gradio 问诊界面（7860）
├── scripts/                         # Windows 一键启动脚本
│   ├── 00_check_env.bat / .py       # 环境自检
│   ├── 01_preprocess.bat            # 数据预处理
│   ├── 02_train.bat                 # 微调训练
│   ├── 03_merge.bat                 # 权重合并（可选）
│   ├── 04_webui.bat                 # 启动 Gradio 问诊网页（7860）
│   ├── 05_predict.bat               # 命令行测试
│   ├── 06_webui_new.bat             # 启动新版问诊网页（8080）
│   ├── test_web_api.py              # Gradio 服务端到端 API 测试（无浏览器自动验证）
│   └── diag_device_map.py           # 显存诊断：打印 4bit 加载后各模块设备分布与占用
├── models/                          # 基座模型目录（不入库）
│   ├── download_weights.ps1         # 串行断点续传下载脚本（需先改内嵌路径）
│   └── qwen2.5-7b-instruct/         # Qwen2.5-7B 本地权重（4 分片，约 15GB）
├── outputs/                         # 训练输出目录（运行后生成，不入库）
│   ├── lora_medical_qwen/
│   │   ├── final_adapter/           # 正式 LoRA 适配器（云端全量版，154 MiB）
│   │   ├── final_adapter_quick200/  # 200 条快速版备份
│   │   ├── checkpoint-12/           # 快速训练检查点
│   │   └── train_results.json       # 200 条快速训练指标
│   └── cloud_adapter/               # 云端回迁临时目录（空，可删）
├── modelscope_migration/            # 云端训练迁移脚本与指引
│   ├── 01_upload_dataset.py         # 上传数据集到 ModelScope
│   ├── 02_setup_env.sh              # 云端环境准备
│   ├── 03_train_cloud.sh            # ms-swift 云端训练启动
│   ├── 05_push_model.sh             # 推送 LoRA 到 Hub
│   ├── dataset_upload/              # 数据集上传工作区
│   └── 云端操作指引.md               # 云端复现流程
└── docs/
    ├── 项目说明.md                   # 项目配套说明（汇报素材，与 PPT 对应）
    └── 云端训练工作日志.md            # 云端训练全过程与回迁证据
```

> 注：README 早期版本提到的 `docs\基于大模型微调的医疗专家.pptx` 已不在仓库内，汇报 PPT 请另行索取/自备。

---

## 7. 快速验证清单（项目演示流程）

1. `python scripts\00_check_env.py` —— 展示环境就绪（GPU、依赖、数据）
2. `python data\preprocess.py` —— 展示数据统计与样例（CoT 格式讲解）
3. 展示训练证据：`outputs\lora_medical_qwen\final_adapter\adapter_config.json`（`r=16 / alpha=32 / 7 类目标层`）与 `docs\云端训练工作日志.md` 的 1,196 步云端记录
   > **注意**：`train_results.json` 是 200 条快速训练的结果（loss=1.995、耗时 1098.7s），**不能当作全量训练指标展示**；云端全量指标以工作日志为准
4. 新版界面（推荐）：`python app\server.py --port 8080` → 打开 http://127.0.0.1:8080 → 点击示例问题 → 展示【医学推理】+【专业解答】流式输出、侧栏会话、参数滑杆
5. 旧版界面（备选）：`python app\app.py` → http://127.0.0.1:7860
6. （可选，无需浏览器）另开一个终端：`python scripts\test_web_api.py`，自动向 Gradio 服务提问并打印模型回答，验证端到端可用
7. 对比演示（加分项）：临时执行 `python app\app.py --lora_path ""` 加载未微调基座，同一问题回答明显缺乏医疗推理结构

更多汇报素材见 `docs\项目说明.md`。
