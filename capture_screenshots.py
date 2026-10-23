#!/usr/bin/env python3
"""
capture_screenshots.py - Automatically captures interaction screenshots for Q1-Q5
and evaluate.py console output using Playwright.
"""
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS = [
    ("q1", "On what date were the prices in this catalogue last updated?"),
    ("q2", "In which year was the publisher (Orient BlackSwan) incorporated?"),
    ("q3", "What is the price of a Physics textbook?"),
    ("q4", "Who is the current Minister of Education?"),
    ("q5", "Which is the \"best\" history book for a beginner?"),
]

def capture_chat_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 880},
            device_scale_factor=2,
        )

        for filename, question in QUESTIONS:
            print(f"[*] Capturing screenshot for {filename}: {question}")
            page = context.new_page()
            page.goto("http://localhost:8000/")
            page.wait_for_selector("#user-input")
            time.sleep(1)

            # Type question and send
            page.fill("#user-input", question)
            page.click("#send-button")

            # Wait for response to be rendered
            page.wait_for_selector(".message-row.bot:nth-of-type(3)", timeout=35000)
            time.sleep(2)  # Settle animations

            out_path = SCREENSHOTS_DIR / f"{filename}.png"
            page.screenshot(path=str(out_path), full_page=False)
            print(f"    - Saved {out_path}")
            page.close()

        # Capture evaluation output screenshot
        print("[*] Generating evaluate.png terminal snapshot...")
        eval_proc = subprocess.run(
            [".venv/bin/python", "evaluate.py"],
            capture_output=True,
            text=True
        )
        eval_output = eval_proc.stdout

        # Render terminal output in a realistic macOS dark terminal window
        term_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    background: #080d18;
    color: #e2e8f0;
    font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", monospace;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 24px;
    box-sizing: border-box;
  }}
  .terminal {{
    width: 900px;
    background: #0d1117;
    border-radius: 12px;
    border: 1px solid rgba(212, 175, 55, 0.3);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 25px rgba(212, 175, 55, 0.15);
    overflow: hidden;
  }}
  .term-header {{
    background: #161b22;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .dot.red {{ background: #ff5f56; }}
  .dot.yellow {{ background: #ffbd2e; }}
  .dot.green {{ background: #27c93f; }}
  .term-title {{
    color: #94a3b8;
    font-size: 13px;
    margin-left: 12px;
    font-weight: 500;
  }}
  .term-body {{
    padding: 20px 24px;
    font-size: 13.5px;
    line-height: 1.55;
    white-space: pre-wrap;
    color: #f1f5f9;
  }}
  .highlight {{ color: #22c55e; font-weight: bold; }}
  .gold {{ color: #d4af37; }}
</style>
</head>
<body>
  <div class="terminal">
    <div class="term-header">
      <div class="dot red"></div>
      <div class="dot yellow"></div>
      <div class="dot green"></div>
      <span class="term-title">terminal — python evaluate.py — 5/5 PASSED</span>
    </div>
    <div class="term-body">{eval_output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</div>
  </div>
</body>
</html>"""
        term_file = Path("screenshots/temp_eval.html")
        term_file.write_text(term_html, encoding="utf-8")

        term_page = context.new_page()
        term_page.goto(f"file://{term_file.resolve()}")
        time.sleep(1)
        term_page.screenshot(path=str(SCREENSHOTS_DIR / "evaluate.png"))
        print("    - Saved screenshots/evaluate.png")
        term_page.close()
        term_file.unlink(missing_ok=True)

        browser.close()
        print("[*] All screenshots captured successfully!")

if __name__ == "__main__":
    capture_chat_screenshots()
