"""Drive the web UI headlessly: human vs algo-easy clicks, then an AI-vs-AI (JEV vs algo) game."""
import sys, time
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:38320/"
with sync_playwright() as p:
    br = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
    pg = br.new_page(viewport={"width": 1500, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(1500)
    # 1) human (black) vs JEV (white): click two intersections
    pg.click("text=人 vs JEV"); pg.click("#btnNew"); pg.wait_for_timeout(500)
    box = pg.locator("#board").bounding_box()
    for (r, c) in [(7, 7), (7, 8)]:
        k = box["width"] / 630; pg.mouse.click(box["x"] + (35 + c * 40) * k, box["y"] + (35 + r * 40) * k)
        pg.wait_for_function("() => !document.getElementById('banner').textContent.includes('思考中')", timeout=30000)
        pg.wait_for_timeout(300)
    print("human-vs-jev banner:", pg.inner_text("#banner"))
    print("log entries:", pg.locator(".mv").count())
    pg.screenshot(path="docs/ui_human_vs_jev.png")
    # 2) JEV vs algo, autoplay until finished
    pg.click("text=JEV vs 算法"); pg.fill("#delay", "0"); pg.click("#btnNew")
    pg.wait_for_function("() => /获胜|平局/.test(document.getElementById('banner').textContent)", timeout=600000)
    print("jev-vs-algo banner:", pg.inner_text("#banner"))
    pg.screenshot(path="docs/ui_jev_vs_algo.png")
    print("page errors:", errors, "| ui error text:", pg.inner_text("#err"))
    br.close()
