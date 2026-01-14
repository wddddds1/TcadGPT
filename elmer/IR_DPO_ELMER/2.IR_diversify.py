#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Elmer .sif IR diversification (lightweight).

Strategy:
- Keep section order.
- Jitter a small number of numeric literals per variant.
"""

import os
import re
import json
import random
import argparse
import importlib.util
from typing import Any, Dict, List, Tuple

CONFIG = {
    "IN_DIR": "elmer_IR",
    "OUT_DIR": "elmer_augmented_IR",
    "MAX_PER_FILE": 3,
    "SEED": 42,
    "JITTER_REL": 0.05,
    "EXTRACTOR_PATH": "elmer_coder/1.IR_batch.py",
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


def load_parser(path: str):
    spec = importlib.util.spec_from_file_location("elmer_ir", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load parser: {path}"
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def jitter_value(x: float, rel: float) -> float:
    return x * (1.0 + random.uniform(-rel, rel))


SKIP_PATTERNS = [
    r"\bSolver\s+\d+\b",
    r"\bBody\s+\d+\b",
    r"\bEquation\s+\d+\b",
    r"\bMaterial\s+\d+\b",
    r"\bBoundary Condition\s+\d+\b",
    r"\bInitial Condition\s+\d+\b",
    r"\bBody Force\s+\d+\b",
    r"\bSolver\s*\d+::",
    r"\bBody\s*\d+::",
    r"\bBoundary Condition\s*\d+::",
    r"\bTarget Boundaries\b",
    r"\bTarget Bodies\b",
    r"\bActive Solvers\b",
]


def should_skip_line(raw: str) -> bool:
    s = raw.strip()
    if not s:
        return True
    # Avoid perturbing identifiers or indices (e.g., Solver 1, Body 2).
    for pat in SKIP_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def jitter_line(line: str, rel: float) -> Tuple[str, bool]:
    # Jitter the first numeric token found
    def _repl(m):
        try:
            val = float(m.group(0))
        except Exception:
            return m.group(0)
        new_val = jitter_value(val, rel)
        return f"{new_val:.6g}"

    new_line, n = NUM_RE.subn(_repl, line, count=1)
    return new_line, n > 0


def diversify_one(data: Dict[str, Any], max_variants: int) -> List[Dict[str, Any]]:
    out = []
    base_sections = data.get("ir", {}).get("sections") or []
    parser = load_parser(CONFIG["EXTRACTOR_PATH"])
    for _ in range(max_variants):
        sections = json.loads(json.dumps(base_sections))
        candidates = []
        for s_idx, sec in enumerate(sections):
            for l_idx, item in enumerate(sec.get("lines", [])):
                raw = item.get("raw")
                if not raw or should_skip_line(raw):
                    continue
                candidates.append((s_idx, l_idx))

        if not candidates:
            continue

        s_idx, l_idx = random.choice(candidates)
        item = sections[s_idx]["lines"][l_idx]
        raw = item.get("raw")
        new_raw, did = jitter_line(raw, CONFIG["JITTER_REL"])
        if did:
            item["raw"] = new_raw
        source_code = parser.render_sif({"sections": sections})
        ir = parser.parse_to_lite_ir(source_code, data.get("meta", {}).get("source_file", ""))
        new_data = dict(data)
        new_data["ir"] = ir
        new_data["source_code"] = source_code
        out.append(new_data)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=CONFIG["IN_DIR"])
    ap.add_argument("--out-dir", default=CONFIG["OUT_DIR"])
    ap.add_argument("--max-per-file", type=int, default=CONFIG["MAX_PER_FILE"])
    args = ap.parse_args()

    random.seed(CONFIG["SEED"])
    ensure_dir(args.out_dir)
    files = list_json_files(args.in_dir)
    if not files:
        print(f"[Error] no IR files in {args.in_dir}")
        return

    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        variants = diversify_one(data, args.max_per_file)
        for i, v in enumerate(variants, start=1):
            out_name = os.path.splitext(os.path.basename(p))[0] + f"__aug{i}.json"
            out_path = os.path.join(args.out_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(v, fw, ensure_ascii=False, indent=2)
            print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
