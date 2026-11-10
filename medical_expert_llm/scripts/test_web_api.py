# -*- coding: utf-8 -*-
"""通过 Gradio API 测试 Web 服务端到端推理（验证 LoRA 适配器生效）。"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:7860"
QUESTION = "最近总是失眠多梦，白天没精神，应该怎么调理？"


def main():
    # 1. 提交推理请求（chat_respond: [问题, 历史, 温度, 最大新token]）
    payload = json.dumps({"data": [QUESTION, [], 0.7, 256]}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/gradio_api/call/chat_respond",
        data=payload, headers={"Content-Type": "application/json"})
    event_id = json.loads(urllib.request.urlopen(req, timeout=60).read())["event_id"]
    print(f"[1] 已提交请求 event_id={event_id}")

    # 2. 读取 SSE 流式结果
    final_history = None
    with urllib.request.urlopen(f"{BASE}/gradio_api/call/chat_respond/{event_id}", timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split("data:", 1)[1].strip()
                if event in ("complete", "generating"):
                    final_history = json.loads(data)[0]
                elif event == "error":
                    print(f"[错误] {data}")
                    sys.exit(1)

    # 3. 输出模型回答
    if not final_history:
        print("[失败] 未收到生成结果")
        sys.exit(1)
    answer = final_history[-1][1]
    print("[2] 模型回答（{} 字）".format(len(answer)))
    print("=" * 60)
    print(answer)
    print("=" * 60)
    ok = len(answer) > 30
    print("[结论] Web 端到端推理 " + ("成功 ✓" if ok else "疑似异常"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
