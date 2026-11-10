# -*- coding: utf-8 -*-
"""
============================================================
医疗问诊 Web 界面（Gradio）
============================================================
功能：
  1. 医疗问答对话（多轮），流式逐字输出，先展示【医学推理】再给【专业解答】
  2. 提供示例问题一键提问、清空对话、生成参数调节

启动（在项目根目录执行）：
  python app/app.py                        # 自动寻找已训练模型
  python app/app.py --port 7860            # 自定义端口
  python app/app.py --lora_path ...        # 显式指定 LoRA 适配器

模型加载优先级（自动）：
  1. outputs/medical_expert_merged        （合并后的完整模型）
  2. outputs/lora_medical_qwen/final_adapter + 基座 （LoRA 适配器）
  3. 基座模型原样加载（未训练时也能跑，用于对比微调前后效果）
"""

import argparse
import threading
from pathlib import Path

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------- 模型路径 -------------------------
DEFAULT_BASE = str(PROJECT_ROOT / "models" / "qwen2.5-7b-instruct")
MERGED_DIR = PROJECT_ROOT / "outputs" / "medical_expert_merged"
LORA_DIR = PROJECT_ROOT / "outputs" / "lora_medical_qwen" / "final_adapter"

# 系统提示词：与训练数据（data/preprocess.py）完全一致，切勿随意修改
SYSTEM_PROMPT = (
    "你是一位经验丰富的医疗AI助手，具备中西医结合的临床知识。"
    "请针对用户提出的健康问题进行解答。回答时请先进行逐步的医学推理分析，"
    "再给出专业、严谨、有条理的结论。"
    "注意：你的回答仅供健康参考，不能替代执业医师的现场诊断与治疗建议。"
)

# 示例问题（演示用）
EXAMPLE_QUESTIONS = [
    "最近一到换季就咳嗽，晚上加重，有少量白痰，这是什么原因？",
    "经常熬夜后心慌、手抖，需要注意什么？",
    "1岁孩子夏季头皮长小结节、溃破流脓长期不愈，中医怎么看待这种病？",
    "体检发现幽门螺杆菌阳性，需要治疗吗？",
]


# ------------------------- 模型加载 -------------------------

def load_model(args):
    """按优先级加载模型并返回 (model, tokenizer, 模型描述)。"""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    ) if not args.no_4bit else None

    if not args.merged_model and MERGED_DIR.exists():
        args.merged_model = str(MERGED_DIR)
    if not args.lora_path and LORA_DIR.exists():
        args.lora_path = str(LORA_DIR)

    if args.merged_model:
        desc = f"完整模型（已合并LoRA）：{args.merged_model}"
        tokenizer = AutoTokenizer.from_pretrained(
            args.merged_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True)
    else:
        base = args.base_model
        desc = f"基座：{base}\nLoRA 适配器：{args.lora_path or '无（原始基座，未微调）'}"
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=quant_config,
            device_map="auto", trust_remote_code=True)
        if args.lora_path:
            model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    model.config.use_cache = True
    return model, tokenizer, desc


# ------------------------- 对话生成 -------------------------

@torch.no_grad()
def chat_respond(user_message: str, chat_history: list,
                 temperature: float, max_new_tokens: int):
    """
    Gradio 聊天回调（生成器函数，逐段 yield 实现流式输出）。
    chat_history 为 [[用户消息, 助手回复], ...] 的列表（含本轮 None 占位）。
    """
    # 1. 占位本轮问答
    chat_history = chat_history + [[user_message, ""]]
    user_message = user_message.strip()

    # 2. 组装多轮 messages：system + 最近 6 轮历史 + 当前提问
    #    截断历史防止超过上下文窗口
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in chat_history[-7:-1]:  # 最近 6 轮（不含当前空占位）
        if turn[0]:  # 用户消息
            messages.append({"role": "user", "content": turn[0]})
        if turn[1]:  # 助手消息
            messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": user_message})

    prompt = MODEL_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = MODEL_TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)

    # 3. 流式生成器：后台线程执行 generate，主线程逐段读取
    streamer = TextIteratorStreamer(
        MODEL_TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        do_sample=True,
        pad_token_id=MODEL_TOKENIZER.pad_token_id
        or MODEL_TOKENIZER.eos_token_id,
    )
    thread = threading.Thread(target=MODEL.generate, kwargs=gen_kwargs)
    thread.start()

    # 4. 每收到一段新文本就更新一次界面（流式打字效果）
    for new_text in streamer:
        chat_history[-1][1] += new_text
        yield chat_history, ""
    thread.join()


# ------------------------- 界面搭建 -------------------------

def build_ui(model_desc: str) -> gr.Blocks:
    """构建 Gradio Blocks 界面：免责声明区 + 对话区 + 参数区。"""
    with gr.Blocks(
            title="医疗智能问诊助手",
            theme=gr.themes.Soft(primary_hue="teal")) as demo:

        # ---- 标题 ----
        gr.Markdown(
            "# 🏥 医疗智能问诊助手\n"
            "基于 Qwen2.5-7B-Instruct + LoRA 微调 · 中文医疗思维链数据集（2万条）\n"
            f"<sub>当前加载：{model_desc}</sub>"
        )

        # ---- 对话区 ----
        chatbot = gr.Chatbot(
            label="问诊对话",
            height=460,
            placeholder="请输入健康问题，例如：最近总是失眠多梦，白天没精神，怎么调理？",
        )
        with gr.Row():
            msg_box = gr.Textbox(
                label="患者提问",
                placeholder="例如：最近总是失眠多梦，白天没精神，怎么调理？",
                scale=4,
            )
            submit_btn = gr.Button("提交", variant="primary", scale=1)
        with gr.Row():
            clear_btn = gr.Button("🗑 清空对话", scale=1)
            retry_btn = gr.Button("🔄 重新回答", scale=1)

        # ---- 示例问题 ----
        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=msg_box,
                    label="示例问题（点击自动填入）")

        # ---- 生成参数 ----
        with gr.Accordion("生成参数（可选）", open=False):
            with gr.Row():
                temp_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.1,
                                        label="temperature（越低越严谨）")
                token_slider = gr.Slider(256, 2048, value=1024, step=128,
                                         label="max_new_tokens（回答长度）")

        # ---------------- 事件绑定 ----------------

        # 提问：文本框回车 / 点击提交按钮
        msg_box.submit(
            chat_respond,
            inputs=[msg_box, chatbot, temp_slider, token_slider],
            outputs=[chatbot, msg_box],
        )
        submit_btn.click(
            chat_respond,
            inputs=[msg_box, chatbot, temp_slider, token_slider],
            outputs=[chatbot, msg_box],
        )

        # 清空对话
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        # 重新回答：去掉最后一条助手回复，重问最后一个问题
        def retry(history, temp, tokens):
            if not history:
                return history, ""
            last_user = history[-1][0] if history[-1][0] else (
                history[-2][0] if len(history) > 1 else "")
            history = history[:-1] if history[-1][1] else history
            # 删除最后一轮，重新生成
            if history and history[-1][0] == last_user:
                history = history[:-1]
            yield from chat_respond(last_user, history, temp, tokens)

        retry_btn.click(
            retry,
            inputs=[chatbot, temp_slider, token_slider],
            outputs=[chatbot, msg_box],
        )

    return demo


# ------------------------- 入口 -------------------------

def main():
    parser = argparse.ArgumentParser(description="医疗问诊 Web 服务")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--merged_model", type=str, default=None)
    parser.add_argument("--no_4bit", action="store_true",
                        help="关闭 4bit 量化（显存充足时可用）")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="生成公网分享链接（需可访问外网）")
    args = parser.parse_args()

    print("=" * 60)
    print(" 医疗智能问诊助手 Web 服务启动中 ...")
    print("=" * 60)

    # 加载模型为全局单例（避免每次对话重复加载）
    global MODEL, MODEL_TOKENIZER
    MODEL, MODEL_TOKENIZER, desc = load_model(args)
    print(f"[模型就绪] {desc}")

    demo = build_ui(desc)
    # queue() 启用流式队列，是逐字输出的必要条件
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=True,      # 自动打开默认浏览器
        show_error=True,
    )


if __name__ == "__main__":
    main()
