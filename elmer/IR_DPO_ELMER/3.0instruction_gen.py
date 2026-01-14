#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Alpaca-style instructions from Elmer IR.
"""

import os
import re
import json
import math
import random
import argparse
from typing import Any, Dict, List, Tuple

CONFIG = {
    "IN_DIR": "elmer_augmented_IR",
    "OUT_DIR": "elmer_alpaca_out",
    "GLOB_SUFFIX": ".json",
    "NUM_VARIANTS_PER_FILE": 1,
    "SEED": 20250101,
    "MAX_ITEMS_INLINE": 8,
    "MAX_EXAMPLES": 6,
    "MAX_CHARS_PER_INSTRUCTION": 1200,
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


def extract_numbers(text: str) -> List[str]:
    return sorted(set(m.group(0) for m in NUM_RE.finditer(text or "")))


def build_numeric_whitelist(ir: Dict[str, Any]) -> set:
    nums = set(ir.get("numbers") or [])
    return set(str(n) for n in nums if isinstance(n, (int, float)))


def cap_and_examples(arr: List[str], max_items: int, max_examples: int) -> str:
    n = len(arr)
    if n <= max_items:
        return "；".join(arr)
    head = "；".join(arr[:max_examples])
    return f"共 {n} 项；示例：{head}"


def summarize_sections(ir: Dict[str, Any]) -> Dict[str, Any]:
    sections = ir.get("sections") or []
    names = [s.get("name") for s in sections if s.get("name")]
    solver_eqs = []
    solver_procs = []
    mesh_db = []
    outputs = []
    bcs = 0
    materials = 0
    include_paths = ir.get("meta", {}).get("include_paths") or []
    includes = ir.get("meta", {}).get("includes") or []
    solver_ext = ir.get("meta", {}).get("solver_ext") or {}

    for sec in sections:
        name = (sec.get("name") or "").lower()
        if name == "solver":
            for line in sec.get("lines", []):
                key = (line.get("key") or "").lower()
                val = (line.get("value") or "").strip()
                if key == "equation" and val:
                    solver_eqs.append(val)
                if key == "procedure" and val:
                    solver_procs.append(val)
        if name == "header":
            for line in sec.get("lines", []):
                if (line.get("key") or "").lower() == "mesh db":
                    mesh_db.append(line.get("value") or "")
        if name == "simulation":
            for line in sec.get("lines", []):
                key = (line.get("key") or "").lower()
                if key in ("output file", "output file name", "output format", "post file"):
                    outputs.append(f"{line.get('key')} = {line.get('value')}")
        if name == "boundary":
            bcs += 1
        if name == "material":
            materials += 1

    return {
        "section_names": names,
        "solver_eqs": solver_eqs,
        "solver_procs": solver_procs,
        "mesh_db": mesh_db,
        "outputs": outputs,
        "bc_count": bcs,
        "material_count": materials,
        "include_paths": include_paths,
        "includes": includes,
        "solver_ext": solver_ext,
    }


def render_instruction(ir: Dict[str, Any]) -> Tuple[str, List[str]]:
    fx = summarize_sections(ir)
    parts = ["请依据以下清单编写可运行的 ELMER .sif 文件："]

    if fx["section_names"]:
        parts.append("包含的 section：" + cap_and_examples(fx["section_names"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["mesh_db"]:
        parts.append("网格目录（Mesh DB）：" + "；".join(fx["mesh_db"]) + "。")
    if fx["include_paths"]:
        parts.append("Include Path：" + cap_and_examples(fx["include_paths"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["includes"]:
        parts.append("Include：" + cap_and_examples(fx["includes"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["solver_eqs"]:
        parts.append("求解器方程（Equation）：" + cap_and_examples(fx["solver_eqs"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["solver_procs"]:
        parts.append("求解器过程（Procedure）：" + cap_and_examples(fx["solver_procs"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["solver_ext"]:
        ext_parts = []
        for sid, kv in fx["solver_ext"].items():
            keys = list(kv.keys())
            ext_parts.append(f"Solver {sid} 扩展项：{cap_and_examples(keys, CONFIG['MAX_ITEMS_INLINE'], CONFIG['MAX_EXAMPLES'])}")
        parts.append("；".join(ext_parts) + "。")
    if fx["outputs"]:
        parts.append("输出相关配置：" + cap_and_examples(fx["outputs"], CONFIG["MAX_ITEMS_INLINE"], CONFIG["MAX_EXAMPLES"]) + "。")
    if fx["bc_count"]:
        parts.append(f"边界条件块数量：{fx['bc_count']}。")
    if fx["material_count"]:
        parts.append(f"材料块数量：{fx['material_count']}。")

    text = " ".join(parts).strip()
    return text, fx["section_names"]


def truncate_soft(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def process_file(path_in: str, out_dir: str, nvar: int) -> Tuple[int, str]:
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)
    ir = data.get("ir") or {}

    records = []
    for _ in range(nvar):
        text, sections_used = render_instruction(ir)
        text = truncate_soft(text, CONFIG["MAX_CHARS_PER_INSTRUCTION"])
        records.append({
            "instruction": text,
            "input": "",
            "output": "",
            "meta": {"sections_used": sections_used},
        })

    out_data = dict(data)
    out_data["alpaca_records"] = records

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(path_in))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    return len(records), f"[OK] {os.path.basename(path_in)} -> {out_path}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=CONFIG["IN_DIR"])
    ap.add_argument("--out-dir", default=CONFIG["OUT_DIR"])
    ap.add_argument("--num-variants", type=int, default=CONFIG["NUM_VARIANTS_PER_FILE"])
    args = ap.parse_args()

    random.seed(CONFIG["SEED"])
    files = list_json_files(args.in_dir)
    if not files:
        print(f"[Error] no IR files in {args.in_dir}")
        return

    total = 0
    for p in files:
        n, msg = process_file(p, args.out_dir, args.num_variants)
        total += n
        print(msg)
    print(f"[Done] total instructions: {total}")


if __name__ == "__main__":
    main()
