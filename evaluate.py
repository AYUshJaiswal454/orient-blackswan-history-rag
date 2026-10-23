#!/usr/bin/env python3
"""
evaluate.py - Automated evaluation of the 5 mandatory exam questions.
Saves results to results.json and results.md, and displays a PASS/FAIL table.
"""
import json
import re
import sys
import time
from pathlib import Path

import rag
import config

TEST_SUITE = [
    {
        "id": "Q1",
        "question": "On what date were the prices in this catalogue last updated?",
        "expected_answer": "20-01-2025",
        "expected_type": "grounded",
        "expected_quote": "Prices updated on 20-01-2025",
    },
    {
        "id": "Q2",
        "question": "In which year was the publisher (Orient BlackSwan) incorporated?",
        "expected_answer": "1948",
        "expected_type": "grounded",
        "expected_quote": "Incorporated in 1948",
    },
    {
        "id": "Q3",
        "question": "What is the price of a Physics textbook?",
        "expected_answer": "Not found",
        "expected_type": "not_found",
        "expected_quote": None,
    },
    {
        "id": "Q4",
        "question": "Who is the current Minister of Education?",
        "expected_answer": "Not found",
        "expected_type": "not_found",
        "expected_quote": None,
    },
    {
        "id": "Q5",
        "question": "Which is the \"best\" history book for a beginner?",
        "expected_answer": "Not found",
        "expected_type": "not_found",
        "expected_quote": None,
    },
]


def extract_reference_details(answer_str: str):
    """Extract page number and quote from reference line."""
    m = re.search(
        r"Reference:\s*Page\s*(\d+)\s*[-–—:]\s*[\"']?(.*?)[\"']?$",
        answer_str,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), m.group(2).strip().strip("\"'")
    return None, None


def evaluate_question(test_item: dict) -> dict:
    q_id = test_item["id"]
    question = test_item["question"]
    expected_ans = test_item["expected_answer"]
    expected_type = test_item["expected_type"]
    expected_quote = test_item["expected_quote"]

    print(f"\nEvaluating [{q_id}] {question}")
    t0 = time.time()
    result = rag.answer_question(question, history=None)
    latency = round(time.time() - t0, 2)

    actual_ans = result.get("answer", "")
    sources = result.get("sources", [])

    page_num, quote = extract_reference_details(actual_ans)
    status = "FAIL"
    reason = ""

    if expected_type == "not_found":
        if actual_ans.strip() == "Not found" and len(sources) == 0:
            status = "PASS"
        else:
            reason = f"Expected exactly 'Not found' with 0 sources, got answer: '{actual_ans}' with {len(sources)} sources"
    else:
        # Grounded answer: must contain expected answer string and valid reference
        if expected_ans in actual_ans and page_num is not None and quote:
            if expected_quote and rag.normalize_text_for_match(expected_quote) in rag.normalize_text_for_match(quote):
                status = "PASS"
            elif not expected_quote:
                status = "PASS"
            else:
                reason = f"Quote '{quote}' did not match expected supporting text '{expected_quote}'"
        else:
            reason = f"Answer did not contain expected '{expected_ans}' or lacked valid Page/Quote reference"

    print(f"  Result: {status} | Latency: {latency}s")
    if reason:
        print(f"  Reason: {reason}")

    return {
        "id": q_id,
        "question": question,
        "expected_answer": expected_ans,
        "actual_answer": actual_ans,
        "reference_page": page_num,
        "reference_quote": quote,
        "sources_count": len(sources),
        "latency_sec": latency,
        "status": status,
        "reason": reason,
    }


def main():
    print("=" * 70)
    print("Orient BlackSwan History Bot - Evaluation Suite")
    print("=" * 70)

    results = []
    for test in TEST_SUITE:
        res = evaluate_question(test)
        results.append(res)

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)

    # 1. Print formatted console table
    print("\n" + "=" * 70)
    print(f"{'ID':<4} | {'Question':<38} | {'Status':<6} | {'Expected':<12}")
    print("-" * 70)
    for r in results:
        q_short = r["question"][:35] + "..." if len(r["question"]) > 38 else r["question"]
        print(f"{r['id']:<4} | {q_short:<38} | {r['status']:<6} | {r['expected_answer']:<12}")
    print("=" * 70)
    print(f"Final Score: {pass_count}/{total_count} passed ({round(pass_count / total_count * 100)}%)\n")

    # 2. Save results.json
    output_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_count,
        "passed_tests": pass_count,
        "score_percentage": round(pass_count / total_count * 100),
        "results": results,
    }
    json_path = config.BASE_DIR / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"[*] Saved JSON results to: {json_path}")

    # 3. Save results.md
    md_content = f"""# Evaluation Results — Orient BlackSwan History Bot

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Score:** {pass_count}/{total_count} ({round(pass_count / total_count * 100)}% Passed)

---

## Summary Table

| ID | Question | Expected Answer | Actual Answer | Status |
|---|---|---|---|---|
"""
    for r in results:
        escaped_actual = r["actual_answer"].replace("\n", " <br> ").replace("|", "\\|")
        md_content += f"| **{r['id']}** | {r['question']} | `{r['expected_answer']}` | {escaped_actual} | **{r['status']}** |\n"

    md_content += "\n---\n\n## Detailed Test Breakdown\n\n"
    for r in results:
        md_content += f"### [{r['id']}] {r['question']}\n\n"
        md_content += f"- **Status:** `{r['status']}`\n"
        md_content += f"- **Expected Answer:** `{r['expected_answer']}`\n"
        md_content += f"- **Actual Answer:**\n```\n{r['actual_answer']}\n```\n"
        if r["reference_page"]:
            md_content += f"- **Reference Page:** Page {r['reference_page']}\n"
            md_content += f"- **Supporting Quote:** *\"{r['reference_quote']}\"*\n"
        md_content += f"- **Sources Retrieved:** {r['sources_count']}\n"
        md_content += f"- **Latency:** {r['latency_sec']}s\n"
        if r["reason"]:
            md_content += f"- **Notes / Failure Reason:** {r['reason']}\n"
        md_content += "\n"

    md_path = config.BASE_DIR / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[*] Saved Markdown results to: {md_path}")

    if pass_count < total_count:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
