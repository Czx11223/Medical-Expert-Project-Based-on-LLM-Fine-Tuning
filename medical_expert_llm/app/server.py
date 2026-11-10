# -*- coding: utf-8 -*-
"""
============================================================
医疗智能问诊助手 · 新版前端服务（FastAPI + SSE 流式）
============================================================
替代 Gradio 界面：托管 app/static/index.html，并提供真实模型流式推理。

启动（在项目根目录）：
  venv\\Scripts\\python app\\server.py --port 8080

接口：
  GET  /              新版问诊界面
  GET  /api/health    服务与模型状态
  POST /api/chat      SSE 流式问答
       请求 {"message": str, "history": [{"role","content"}],
             "temperature": 0.7, "max_new_tokens": 1024}
       响应 data: {"text": "..."}  /  data: {"done": true}  /  data: {"error": "..."}
"""

import argparse
import json
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"

BASE_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-instruct"
MERGED_DIR = PROJECT_ROOT / "outputs" / "medical_expert_merged"
LORA_DIR = PROJECT_ROOT / "outputs" / "lora_medical_qwen" / "final_adapter"

# 与训练数据（data/preprocess.py）一致的系统提示词，切勿随意修改
SYSTEM_PROMPT = (
    "你是一位经验丰富的医疗AI助手，具备中西医结合的临床知识。"
    "请针对用户提出的健康问题进行解答。回答时请先进行逐步的医学推理分析，"
    "再给出专业、严谨、有条理的结论。"
    "注意：你的回答仅供健康参考，不能替代执业医师的现场诊断与治疗建议。"
)

MODEL = None
TOKENIZER = None
MODEL_DESC = ""
GEN_LOCK = threading.Lock()  # 单 GPU 串行生成，避免并发挤爆显存


def load_model():
    """加载模型：优先合并权重，其次基座 + LoRA 适配器（4bit 量化）。"""
    global MODEL, TOKENIZER, MODEL_DESC
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
    )
    if MERGED_DIR.exists():
        MODEL_DESC = "完整模型（已合并 LoRA）"
        TOKENIZER = AutoTokenizer.from_pretrained(str(MERGED_DIR), trust_remote_code=True)
        MODEL = AutoModelForCausalLM.from_pretrained(
            str(MERGED_DIR), quantization_config=quant_config,
            device_map="auto", trust_remote_code=True)
    else:
        MODEL_DESC = "Qwen2.5-7B-Instruct + 医疗 LoRA"
        TOKENIZER = AutoTokenizer.from_pretrained(str(BASE_DIR), trust_remote_code=True)
        MODEL = AutoModelForCausalLM.from_pretrained(
            str(BASE_DIR), quantization_config=quant_config,
            device_map="auto", trust_remote_code=True)
        if LORA_DIR.exists():
            MODEL = PeftModel.from_pretrained(MODEL, str(LORA_DIR))
        else:
            MODEL_DESC += "（未找到 LoRA，使用原始基座）"
    MODEL.eval()
    MODEL.config.use_cache = True
    print("[模型就绪] " + MODEL_DESC, flush=True)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    max_new_tokens: int = Field(default=1024, ge=64, le=2048)


def build_messages(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 只保留最近 6 轮有效历史，防止超出上下文窗口
    for turn in req.history[-12:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": req.message.strip()})
    return messages


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


app = FastAPI(title="医疗智能问诊助手")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_DESC,
            "cuda": torch.cuda.is_available(), "busy": GEN_LOCK.locked()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    messages = build_messages(req)
    prompt = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    streamer = TextIteratorStreamer(
        TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        do_sample=True,
        pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
    )

    def run_generation():
        try:
            with GEN_LOCK, torch.no_grad():
                MODEL.generate(**gen_kwargs)
        except Exception as exc:  # 生成异常也要结束流，前端才能收到错误
            print("[生成异常] " + str(exc), flush=True)
        finally:
            streamer.end()

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    def event_stream():
        try:
            for text in streamer:
                if text:
                    yield sse({"text": text})
            yield sse({"done": True})
        except Exception as exc:
            yield sse({"error": str(exc)})
        finally:
            thread.join(timeout=5)

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    parser = argparse.ArgumentParser(description="医疗问诊新版前端服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print("=" * 60)
    print(" 医疗智能问诊助手 · 新版界面启动中 ...")
    print("=" * 60, flush=True)
    load_model()
    print(f"[服务地址] http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
