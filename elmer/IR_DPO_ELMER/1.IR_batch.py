#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch IR extractor for Elmer .sif files.

Usage:
  python 1.IR_batch.py --in-root data/sources/elmer/official_sif --out-dir elmer_IR
"""

import os
import re
import json
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_IN_ROOT = "data/sources/elmer/official_sif"
DEFAULT_OUT_DIR = "elmer_IR"

NUM_RE = re.compile(r"(?<![\w/.-])[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w/.-])")


def sha1_short(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:n]


def rel_safe_name(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    stem = rel.replace(os.sep, "__")
    return f"{stem}__{sha1_short(rel)}"


def write_json(obj: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def walk_sif_files(root: str) -> List[str]:
    out = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".sif"):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def strip_inline_comment(line: str) -> str:
    # Keep full line if it is a pure comment line
    s = line.lstrip()
    if s.startswith("!") or s.startswith("#"):
        return line
    for mark in ("!", "#"):
        if mark in line:
            idx = line.find(mark)
            return line[:idx]
    return line


SECTION_NAMES = [
    "Boundary Condition",
    "Initial Condition",
    "Body Force",
    "Body",
    "Material",
    "Equation",
    "Solver",
    "Simulation",
    "Header",
    "Constants",
    "Component",
]

GLOBAL_COMMANDS = ["Check Keywords", "RUN", "Include"]
LOOSE_KV_KEYS = ["Include Path", "Results Directory"]


def _match_section_header(s: str) -> Optional[Tuple[str, Optional[str]]]:
    low = s.lower()
    for name in sorted(SECTION_NAMES, key=len, reverse=True):
        nlow = name.lower()
        if low == nlow or low.startswith(nlow + " "):
            rest = s[len(name):].strip()
            return name, rest if rest else None
    return None


def parse_section_header(line: str) -> Optional[Tuple[str, Optional[str]]]:
    # Section header examples: "Simulation", "Solver 1", "Boundary Condition 2"
    s = line.strip()
    if not s or "=" in s:
        return None
    if s.lower().startswith("end"):
        return None
    return _match_section_header(s)


def parse_key_value(line: str) -> Optional[Tuple[str, str]]:
    if "=" not in line:
        return None
    left, right = line.split("=", 1)
    key = left.strip()
    val = right.strip()
    if not key:
        return None
    return key, val


def parse_global_command(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s:
        return None
    for cmd in GLOBAL_COMMANDS:
        if s.lower().startswith(cmd.lower()):
            rest = s[len(cmd):].strip()
            return cmd, rest
    parts = s.split()
    if not parts:
        return None
    key = parts[0]
    val = " ".join(parts[1:]).strip()
    return key, val


def parse_loose_kv(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s or "=" in s:
        return None
    for key in LOOSE_KV_KEYS:
        if s.lower().startswith(key.lower() + " "):
            val = s[len(key):].strip()
            return key, val
    return None


def parse_solver_ext(line: str) -> Optional[Tuple[str, str, str]]:
    m = re.match(r"^Solver\s+(\d+)\s*::\s*([^=]+?)\s*=\s*(.+)$", line.strip(), flags=re.I)
    if not m:
        return None
    solver_id, key, val = m.group(1), m.group(2).strip(), m.group(3).strip()
    return solver_id, key, val


def extract_numbers(text: str) -> List[float]:
    nums = []
    for m in NUM_RE.finditer(text or ""):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            continue
    return nums


def parse_to_lite_ir(text: str, source_file: str) -> Dict[str, Any]:
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    global_sec: Optional[Dict[str, Any]] = None
    solver_ext: Dict[str, Dict[str, str]] = {}
    include_paths: List[str] = []
    includes: List[str] = []
    keywords = []
    numbers = []

    for raw in text.splitlines():
        line = strip_inline_comment(raw).rstrip()
        if not line.strip():
            continue

        # Preserve full-line comments as raw lines
        if line.lstrip().startswith(("!", "#")):
            if cur is None:
                if global_sec is None:
                    global_sec = {"name": "Global", "tag": None, "lines": []}
                    sections.append(global_sec)
                global_sec["lines"].append({"raw": line, "key": "__comment__", "value": None})
            else:
                cur["lines"].append({"raw": line, "key": "__comment__", "value": None})
            continue

        header = parse_section_header(line)
        if header:
            name, tag = header
            cur = {"name": name, "tag": tag, "lines": []}
            sections.append(cur)
            continue

        if line.strip().lower().startswith("end"):
            cur = None
            continue

        kv = parse_key_value(line)
        if not kv:
            kv = parse_loose_kv(line)
        if kv and cur is not None:
            key, val = kv
            cur["lines"].append({"raw": line, "key": key, "value": val})
            keywords.append(key)
            numbers.extend(extract_numbers(val))
            if key.lower() == "include path":
                include_paths.append(val)
            if key.lower() == "include":
                includes.append(val)
            continue

        if cur is not None:
            # Continuation lines (indented MATC/Real blocks)
            if line[:1].isspace() and cur["lines"]:
                last = cur["lines"][-1]
                cont = last.get("cont") or []
                cont.append(line.strip())
                last["cont"] = cont
                numbers.extend(extract_numbers(line))
                continue
            # Keep raw lines that don't match key/value (e.g., arrays or loose syntax)
            cur["lines"].append({"raw": line, "key": None, "value": None})
            numbers.extend(extract_numbers(line))
            continue

        # Global commands or loose lines outside sections
        gkv = parse_key_value(line)
        if not gkv:
            gkv = parse_loose_kv(line)
        if not gkv:
            gkv = parse_global_command(line)
        if gkv:
            ext = parse_solver_ext(line)
            if ext:
                solver_id, skey, sval = ext
                solver_ext.setdefault(solver_id, {})[skey] = sval
            if global_sec is None:
                global_sec = {"name": "Global", "tag": None, "lines": []}
                sections.append(global_sec)
            key, val = gkv
            global_sec["lines"].append({"raw": line, "key": key, "value": val})
            keywords.append(key)
            numbers.extend(extract_numbers(val))
            if key.lower() == "include":
                includes.append(val)

    meta = {
        "source_file": source_file,
        "section_count": len(sections),
        "section_names": [s["name"] for s in sections],
    }
    if solver_ext:
        meta["solver_ext"] = solver_ext
    if include_paths:
        meta["include_paths"] = include_paths
    if includes:
        meta["includes"] = includes

    def _count_section(name: str) -> int:
        return sum(1 for s in sections if (s.get("name") or "").lower() == name.lower())

    meta["solver_count"] = _count_section("Solver")
    meta["bc_count"] = _count_section("Boundary Condition")
    meta["material_count"] = _count_section("Material")
    meta["body_count"] = _count_section("Body")
    meta["equation_count"] = _count_section("Equation")

    ir = {
        "sections": sections,
        "keywords": sorted(set(keywords)),
        "numbers": sorted(set(numbers)),
        "meta": meta,
    }
    return ir


def render_sif(ir: Dict[str, Any]) -> str:
    sections = ir.get("sections") or []
    lines: List[str] = []
    for sec in sections:
        name = sec.get("name") or ""
        tag = sec.get("tag")
        if name == "Global":
            for item in sec.get("lines", []):
                raw = item.get("raw")
                if raw:
                    lines.append(raw)
            lines.append("")
            continue
        header = f"{name} {tag}".strip() if tag else str(name)
        lines.append(header)
        for item in sec.get("lines", []):
            raw = item.get("raw")
            if raw:
                lines.append(raw)
            for cont in item.get("cont") or []:
                if cont.startswith((" ", "\t")):
                    lines.append(cont)
                else:
                    lines.append("  " + cont)
        lines.append("End")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def process_file(path: str, root: str, out_dir: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ir = parse_to_lite_ir(text, path)
    obj = {
        "source_code": text,
        "ir": ir,
        "meta": {"source_file": path},
    }
    out_name = rel_safe_name(path, root) + ".json"
    out_path = os.path.join(out_dir, out_name)
    write_json(obj, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default=DEFAULT_IN_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    files = walk_sif_files(args.in_root)
    if not files:
        print(f"[Error] No .sif files found under {args.in_root}")
        return

    for p in files:
        out_path = process_file(p, args.in_root, args.out_dir)
        print(f"[OK] {p} -> {out_path}")


if __name__ == "__main__":
    main()
