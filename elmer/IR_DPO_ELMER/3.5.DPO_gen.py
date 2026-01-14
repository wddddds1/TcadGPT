#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Elmer DPO negative sample generator.
"""

import os
import re
import json
import random
import argparse
from typing import Any, Dict, List, Optional

CONFIG = {
    "IN_DIR": "elmer_alpaca_out",
    "OUT_DIR": "dpo_pairs_elmer",
    "SEED": 20250102,
    "N_NUMERIC": 2,
    "MAX_REJECTED_PER_REC": 4,
}

NUM_RE = re.compile(r"(?<![\w/.-])[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w/.-])")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_json_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".json"):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def pick_chosen_code(data: dict) -> str:
    code = (data.get("source_code") or "").strip()
    return code


def mutate_numeric(code: str) -> Optional[str]:
    matches = list(NUM_RE.finditer(code))
    if not matches:
        return None
    m = random.choice(matches)
    try:
        val = float(m.group(0))
    except Exception:
        return None
    factor = random.choice([10.0, 0.1, 1.1, 0.9])
    new_val = f"{val * factor:.6g}"
    return code[:m.start()] + new_val + code[m.end():]


def omit_line(code: str, key: str) -> Optional[str]:
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if key.lower() in ln.lower():
            return "\n".join(lines[:i] + lines[i + 1:]) + "\n"
    return None


def build_rejected_variants(code: str) -> List[Dict[str, str]]:
    out = []
    num = mutate_numeric(code)
    if num:
        out.append({"type": "numeric", "code": num})
    for key, t in [
        ("Mesh DB", "omit_meshdb"),
        ("Output File", "omit_output"),
        ("Procedure", "omit_procedure"),
    ]:
        cand = omit_line(code, key)
        if cand:
            out.append({"type": t, "code": cand})
    return out


def process_file(path_in: str, out_dir: str) -> None:
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    chosen = pick_chosen_code(data)
    if not chosen:
        return

    rejected_candidates = build_rejected_variants(chosen)
    random.shuffle(rejected_candidates)
    rejected_candidates = rejected_candidates[:CONFIG["MAX_REJECTED_PER_REC"]]

    dpo_pairs = []
    for rc in rejected_candidates:
        dpo_pairs.append({
            "type": rc["type"],
            "chosen": chosen,
            "rejected": rc["code"],
        })

    out = dict(data)
    out["dpo_pairs"] = {"code": dpo_pairs}

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(path_in))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {os.path.basename(path_in)} -> {out_path} pairs={len(dpo_pairs)}")


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
