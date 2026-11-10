# -*- coding: utf-8 -*-
"""对 UI 原型做真实运行验证：渲染、核心交互、控制台错误、截图。"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8899/medical-chat-ui.html"
OUT = Path(__file__).resolve().parent
errors = []
checks = []


def ok(name, cond):
    checks.append((name, bool(cond)))
    print(("[PASS] " if cond else "[FAIL] ") + name, flush=True)


with sync_playwright() as p:
    local_chrome = Path.home() / "AppData/Local/ms-playwright/chromium-1210/chrome-win64/chrome.exe"
    launch_kwargs = {"headless": True}
    if local_chrome.exists():
        launch_kwargs["executable_path"] = str(local_chrome)
    browser = p.chromium.launch(**launch_kwargs)
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text)
            if m.type == "error" else None)

    page.goto(BASE, wait_until="networkidle")
    page.wait_for_timeout(400)

    # 1. 主视图渲染
    ok("页面标题正确", "UI" in page.title())
    ok("示例卡片渲染 4 张", page.locator(".example-card").count() == 4)
    ok("发送按钮初始禁用", page.locator("#sendBtn").is_disabled())
    ok("欢迎语可见", page.locator("#welcome").is_visible())
    page.screenshot(path=str(OUT / "shot_prototype_empty.png"), full_page=False)

    # 2. 核心交互：示例提问 + 双段流式回答
    page.locator(".example-card").first.click()
    page.wait_for_selector(".msg-user", timeout=5000)
    ok("用户消息上屏", page.locator("#messages .msg-user").count() == 1)
    page.wait_for_selector("#messages .msg-actions", timeout=60000)
    reason = page.locator("#messages .reason-body").last.inner_text()
    answer = page.locator("#messages .answer-body").last.inner_text()
    ok("医学推理段非空", len(reason) > 30)
    ok("专业解答段非空且含列表", len(answer) > 60 and page.locator("#messages .answer-body li").count() > 0)
    ok("回答结束后生成标记清除", page.locator("#messages .caret").count() == 0)
    ok("重新回答按钮可用", page.locator("#retryBtn").is_enabled())
    page.screenshot(path=str(OUT / "shot_prototype_chat.png"), full_page=False)

    # 3. 推理折叠交互
    details = page.locator("#messages details.reasoning").last
    ok("推理默认展开", details.get_attribute("open") is not None)
    page.locator("#messages details.reasoning summary").last.click()
    page.wait_for_timeout(200)
    ok("点击后推理折叠", details.get_attribute("open") is None)

    # 4. 复制反馈 Toast
    page.locator('#messages .text-btn[data-act="copy"]').last.click()
    page.wait_for_timeout(200)
    ok("复制后 Toast 出现", "show" in (page.locator("#toast").get_attribute("class") or ""))

    # 5. 自定义问题 + Enter 发送
    page.locator("#input").fill("最近胃胀，饭后明显，持续一周了")
    ok("输入后发送按钮可用", page.locator("#sendBtn").is_enabled())
    page.locator("#input").press("Enter")
    page.wait_for_function("document.querySelectorAll('#messages .msg-actions').length >= 2", timeout=60000)
    ok("自定义问题得到通用回答", page.locator("#messages .msg-user").count() == 2)

    # 6. 清空对话
    page.locator(".composer-tools .text-btn").first.click()
    page.wait_for_timeout(300)
    ok("清空后回到欢迎态", page.locator("#welcome").is_visible()
       and page.locator("#messages .msg").count() == 0)

    # 7. 参数滑杆
    page.locator("#tempRange").evaluate(
        "el => { el.value = '0.3'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    ok("温度滑杆联动数值", page.locator("#tempVal").inner_text() == "0.3")

    # 8. 规范页 / 组件页
    page.locator("#tab-system").click()
    page.wait_for_timeout(200)
    ok("设计规范页可见且色板渲染", page.locator("#view-system").is_visible()
       and page.locator("#swatchPrimary .swatch").count() == 10)
    page.screenshot(path=str(OUT / "shot_system.png"), full_page=True)
    page.locator("#tab-components").click()
    page.wait_for_timeout(200)
    ok("组件库页可见", page.locator("#view-components").is_visible()
       and page.locator(".component-block").count() == 4)
    page.screenshot(path=str(OUT / "shot_components.png"), full_page=True)

    # 9. 移动端响应式
    page.locator("#tab-prototype").click()
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    ok("移动端单栏（会话栏隐藏）", not page.locator(".rail").is_visible())
    ok("移动端输入区仍可用", page.locator("#input").is_visible())
    page.screenshot(path=str(OUT / "shot_mobile.png"), full_page=False)

    browser.close()

print("\n--- console/page errors ---")
print("\n".join(errors) if errors else "(none)")
failed = [n for n, c in checks if not c]
print("\n结果：%d/%d 项通过" % (len(checks) - len(failed), len(checks)))
sys.exit(1 if failed or errors else 0)
