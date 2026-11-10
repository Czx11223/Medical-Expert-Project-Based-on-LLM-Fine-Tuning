"""线上版最终回答完整性验证：等到生成结束再检查双段结构。"""
import sys, time, os
from playwright.sync_api import sync_playwright

CHROME = r"C:/Users/20454/AppData/Local/ms-playwright/chromium-1210/chrome-win64/chrome.exe"
if not os.path.exists(CHROME):
    CHROME = r"C:/Users/20454/AppData/Local/ms-playwright/chromium-1210/chrome-win/chrome.exe"
URL = "https://medical-chat.app.workbuddy.host/?sharecode=U3O7R2TAIK"

fails = []

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("  -> " + str(extra)) if extra and not cond else ""))
    if not cond:
        fails.append(name)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))

    page.goto(URL, wait_until="networkidle", timeout=90000)
    time.sleep(2)

    # 点击真实示例卡片（ex-title 所在按钮）
    card = page.locator("button:has(.ex-title), .example-grid button, #exampleGrid button").first
    if card.count() > 0:
        card.click()
        ok("点击示例卡片提问", True)
    else:
        ta = page.locator("textarea").first
        ta.fill("最近总是失眠，晚上很难入睡，白天很疲惫，怎么办？")
        ta.press("Enter")
        ok("输入框兜底提问", True)

    # 等生成结束：文本连续 12 秒不再变化
    last, stable_since = "", time.time()
    deadline = time.time() + 240
    body = ""
    while time.time() < deadline:
        try:
            body = page.locator("#messages").inner_text()
        except Exception:
            body = ""
        if body and body == last and len(body) > 200:
            if time.time() - stable_since >= 12:
                break
        else:
            stable_since = time.time()
            last = body
        time.sleep(3)

    ok("回答生成完成", len(body) > 200, len(body))
    has_reason = ("医学推理" in body)
    has_answer = ("专业解答" in body)
    ok("包含医学推理段", has_reason)
    ok("包含专业解答段", has_answer)
    # 专业解答段后应有实质内容（不是只有标题）
    seg = body.split("专业解答")[-1] if has_answer else ""
    ok("专业解答有实质内容", len(seg.strip()) > 100, repr(seg[:80]))
    ok("无报错字样", "Error" not in body and "错误" not in body)
    ok("全程无 JS 报错", len(errors) == 0, errors[:3])

    page.screenshot(path="shot_online_final.png", full_page=False)
    browser.close()

print("\n结果：%d 项失败" % len(fails))
sys.exit(1 if fails else 0)
