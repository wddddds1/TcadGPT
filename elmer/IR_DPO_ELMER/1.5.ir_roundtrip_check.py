#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Round-trip fidelity check for Elmer .sif IR.
"""

import argparse
import importlib.util
from typing import List


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("elmer_ir", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load module: {path}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def normalize_line(line: str, strip_comment) -> str:
    s = strip_comment(line).strip()
    s = " ".join(s.split())
    return s


def collect_core_lines(text: str, strip_comment) -> List[str]:
    out = []
    for ln in text.splitlines():
        s = normalize_line(ln, strip_comment)
        if s:
            out.append(s)
    return out


def coverage_ratio(src_lines: List[str], rt_lines: List[str]) -> float:
    if not src_lines:
        return 1.0
    rt_set = set(rt_lines)
    hit = sum(1 for ln in src_lines if ln in rt_set)
    return hit / max(1, len(src_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sif", required=True)
    ap.add_argument("--parser", default="elmer_coder/1.IR_batch.py")
    args = ap.parse_args()

    mod = load_module(args.parser)

    with open(args.sif, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()

    ir = mod.parse_to_lite_ir(src, args.sif)
    rt = mod.render_sif(ir)

    src_lines = collect_core_lines(src, mod.strip_inline_comment)
    rt_lines = collect_core_lines(rt, mod.strip_inline_comment)

    ratio = coverage_ratio(src_lines, rt_lines)
    print("source_lines", len(src_lines))
    print("roundtrip_lines", len(rt_lines))
    print("coverage_ratio", f"{ratio:.3f}")
    print("sections", [s.get("name") for s in ir.get("sections", [])])


if __name__ == "__main__":
    main()
