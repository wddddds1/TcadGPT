#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add simple COT-style summaries to Elmer DPO pairs.
"""

import os
import re
import json
import random
import argparse
from typing import Any, Dict, List

CONFIG = {
    "IN_DIR": "dpo_pairs_elmer",
    "OUT_DIR": "dpo_pairs_elmer_cot",
    "SEED": 20250103,
    "MAX_ITEMS_INLINE": 8,
    "MAX_EXAMPLES": 6,
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_json_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def cap_and_examples(arr: List[str]) -> str:
    n = len(arr)
    if n <= CONFIG["MAX_ITEMS_INLINE"]:
        return "；".join(arr)
    head = "；".join(arr[:CONFIG["MAX_EXAMPLES"]])
    return f"共 {n} 项；示例：{head}"


def extract_sections(code: str) -> List[str]:
    secs = []
    for line in code.splitlines():
        s = line.strip()
        if not s or "=" in s:
            continue
        if s.lower().startswith("end"):
            continue
        secs.append(s.split()[0])
    return list(dict.fromkeys(secs))


def summarize_code(code: str) -> str:
    sections = extract_sections(code)
    parts = ["按指令组织 .sif 的主要 section。"]
    if sections:
        parts.append("包含 section：" + cap_and_examples(sections) + "。")
    parts.append("保持求解器与输出配置一致，确保语法完整。")
    return " ".join(parts)


def process_file(path_in: str, out_dir: str) -> None:
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = (data.get("dpo_pairs") or {}).get("code") or []
    for p in pairs:
        chosen = p.get("chosen") or ""
        summary = summarize_code(chosen)
        p["chosen_cot"] = summary
        p["rejected_cot"] = summary

    out = dict(data)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(path_in))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {os.path.basename(path_in)} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=CONFIG["IN_DIR"])
    ap.add_argument("--out-dir", default=CONFIG["OUT_DIR"])
    args = ap.parse_args()

    random.seed(CONFIG["SEED"])
    files = list_json_files(args.in_dir)
    if not files:
        print(f"[Error] no files in {args.in_dir}")
        return
    for p in files:
        process_file(p, args.out_dir)


if __name__ == "__main__":
    main()
