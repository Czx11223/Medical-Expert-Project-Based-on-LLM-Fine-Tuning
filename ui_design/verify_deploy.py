# -*- coding: utf-8 -*-
"""新版问诊前端真实部署验证：真实模型流式问答 + 会话持久化 + 响应式。"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8080"
OUT = Path(__file__).resolve().parent
errors = []
checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(("[PASS] " if cond else "[FAIL] ") + name, flush=True)


with sync_playwright() as p:
    local_chrome = Path.home() / "AppData/Local/ms-playwright/chromium-1210/chrome-win64/chrome.exe"
    kwargs = {"headless": True}
    if local_chrome.exists():
        kwargs["executable_path"] = str(local_chrome)
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text)
            if m.type == "error" else None)

    page.goto(BASE, wait_until="domcontentloaded")
    page.wait_for_timeout(600)

    ok("页面标题为产品名", page.title() == "医疗智能问诊助手")
    ok("示例卡片 4 张", page.locator(".example-card").count() == 4)
    ok("健康检查模型就绪", "模型就绪" in page.locator("#healthBadge").inner_text())
    ok("无历史时显示空态", page.locator(".rail-empty").is_visible())
    page.screenshot(path=str(OUT / "shot_deploy_home.png"))

    # 真实模型问答
    page.locator(".example-card").first.click()
    page.wait_for_selector("#messages .msg-user", timeout=10000)
    ok("用户消息上屏", page.locator("#messages .msg-user").count() == 1)
    ok("生成中显示停止按钮", "is-stop" in (page.locator("#sendBtn").get_attribute("class") or ""))
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "shot_deploy_streaming.png"))

    page.wait_for_selector("#messages .msg-actions", timeout=240000)
    page.wait_for_timeout(400)
    reason = page.locator("#messages .reason-body").last.inner_text()
    answer = page.locator("#messages .answer-body").last.inner_text()
    print("\n--- 医学推理（前 200 字）---\n" + reason[:200], flush=True)
    print("\n--- 专业解答（前 400 字）---\n" + answer[:400] + "\n", flush=True)
    ok("真实医学推理非空", len(reason) > 30)
    ok("真实专业解答非空", len(answer) > 50)
    ok("流式光标已清除", page.locator("#messages .caret").count() == 0)
    ok("回答含医学推理标题", "医学推理" in page.locator("#messages details.reasoning summary").last.inner_text())
    page.screenshot(path=str(OUT / "shot_deploy_answer.png"))

    # 会话持久化：刷新后恢复
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(700)
    ok("刷新后会话列表保留", page.locator("#sessionList .session").count() >= 1)
    ok("刷新后历史回答恢复", page.locator("#messages .msg-ai").count() == 1
       and len(page.locator("#messages .answer-body").last.inner_text()) > 50)

    # 折叠 / 复制 / 新建
    page.locator("#messages details.reasoning summary").last.click()
    page.wait_for_timeout(200)
    ok("推理可折叠", page.locator("#messages details.reasoning").last.get_attribute("open") is None)
    page.locator('#messages .text-btn[data-act="copy"]').last.click()
    page.wait_for_timeout(200)
    ok("复制 Toast", "show" in (page.locator("#toast").get_attribute("class") or ""))
    page.locator("#newChatBtn").click()
    page.wait_for_timeout(300)
    ok("新建问诊回到欢迎态", page.locator("#welcome").is_visible())

    # 移动端
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    ok("移动端单栏", not page.locator(".rail").is_visible())
    ok("移动端输入可用", page.locator("#input").is_visible())
    page.screenshot(path=str(OUT / "shot_deploy_mobile.png"))

    browser.close()

print("\n--- console/page errors ---")
print("\n".join(errors) if errors else "(none)")
failed = [n for n, c in checks if not c]
print("\n结果：%d/%d 项通过" % (len(checks) - len(failed), len(checks)))
sys.exit(1 if failed or errors else 0)
