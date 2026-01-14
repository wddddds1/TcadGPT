#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean instruction_variants to avoid section list phrasing.
Reads outputs/instruction_aug_elmer_dpo_full_v3 and writes v4.
"""

import os
import re
import json

INPUT_FOLDER = "outputs/instruction_aug_elmer_dpo_full_v7"
OUTPUT_FOLDER = "outputs/instruction_aug_elmer_dpo_full_v8"

SECTION_WORDS = [
    "Header", "Constants", "Simulation", "Body Force", "Body",
    "Material", "Equation", "Solver", "Boundary Condition",
    "Initial Condition", "Component", "Global",
]

SECTION_RE = re.compile(
    r"包含[^。]*?(?:"
    + "|".join(re.escape(w) for w in SECTION_WORDS)
    + r")[^。]*?(?:部分|段落|section)?[。]",
    flags=re.IGNORECASE,
)


def normalize_variant(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    s = SECTION_RE.sub("包含必要的基础设置。", s)
    s = re.sub(r"(?:材料|边界条件)块数量[^。]*。", "材料和边界按要求配置。", s)
    s = re.sub(r"共\\s*\\d+\\s*项", "若干项", s)
    s = re.sub(r"[，；]\\s*[，；]+", "，", s)
    return s.strip()


def main() -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = []
    for dp, _, fns in os.walk(INPUT_FOLDER):
        for fn in fns:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(dp, fn))
    files.sort()
    if not files:
        print(f"[Error] no files in {INPUT_FOLDER}")
        return

    for path in files:
        out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(path))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recs = data.get("alpaca_records") or []
        if recs:
            variants = recs[0].get("instruction_variants") or []
            if variants:
                recs[0]["instruction_variants"] = [normalize_variant(v) for v in variants]
            for p in (data.get("dpo_pairs") or {}).get("code") or []:
                if p.get("instruction_variants"):
                    p["instruction_variants"] = [normalize_variant(v) for v in p["instruction_variants"]]
        data["alpaca_records"] = recs
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] {os.path.basename(path)} -> {out_path}")


if __name__ == "__main__":
    main()
