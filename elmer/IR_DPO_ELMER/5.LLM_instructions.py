#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instruction augmentation for Elmer DPO records.
"""

import os
import re
import json
import time
import random
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FOLDER = "outputs/dpo_pairs_elmer_cot_full_v3"
OUTPUT_FOLDER = "outputs/instruction_aug_elmer_dpo_full_v7"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 128
MAX_RETRY = 5
RETRY_BASE_DELAY = 1.6

client = openai.Client(
    api_key="sk-REDACTED",
    base_url="https://api.deepseek.com"
)

NUM_TOKEN = re.compile(r"(?<![\w/.-])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w/.-])")


def extract_numbers(text: str) -> set:
    return set(NUM_TOKEN.findall(text or ""))


def build_numbers_note(rec: dict) -> str:
    candidates = []
    for k in ("instruction", "source_code"):
        v = rec.get(k, "")
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    if not candidates:
        return ""
    nums = sorted(extract_numbers("\n".join(candidates)))
    return "、".join(nums[:200])


def build_prompt(instruction: str, numbers_note: str) -> str:
    return f"""
你是 ELMER .sif 方向的高质量“指令润色器”。请将输入的“原始 instruction”改写为 5 条中文指令，保持事实不变，不新增或删除关键条件。
输出必须为 JSON 数组，长度为 5，顺序固定。

【目标】
把生硬、清单式的描述，改成“真实用户在提需求”时会说的话。长款给足细节但自然，短款要简洁自然，绝不出现“共X项/列出X条/目录清单/流程箭头”等机械用语。
每条不超过 80 个中文字符，尽量一句话完成。

【五种风格（顺序固定，全部必须产出）】
1) 科研任务型（参数全给）
2) 工程师口吻型（核心参数为主）
3) 简介描述型（描述目标为主，数值少）
4) 需求拆解型（先背景后步骤，但不用流程箭头）
5) 模糊目标型（只给方向与边界）

【硬性约束】
- 一律中文。
- 不要出现 section 字样，不要列出 section 名称或数量。
- 不要使用“包含的清单/条目/如下”等元话术。
- 不要出现流程箭头（如 A→B→C）。
- 不要新增新求解器/材料/边界/数值等未出现的信息。
- 具体数值只能来自原 instruction（可省略，不可新增）。
- 必须严格输出 JSON 数组：["...", "...", "...", "...", "..."]，不得包含反引号/代码块/多余文字。

可用数值白名单（仅供约束，不必全部使用）：{numbers_note}

原始 instruction：
{instruction}
""".strip()

def normalize_variant(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    # Remove section lists/counts and replace with neutral phrasing.
    s = re.sub(r"包含[^。]*section[^。]*。", "包含必要的基础段落。", s, flags=re.IGNORECASE)
    s = re.sub(r"\\bsection\\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:共|共有)?\\s*\\d+\\s*个[^。]*?段落。", "包含必要的基础段落。", s)
    s = re.sub(r"(?:材料|边界条件)块数量[^。]*。", "材料和边界按要求配置。", s)
    s = re.sub(r"共\\s*\\d+\\s*项", "若干项", s)
    s = re.sub(r"[，；]\\s*[，；]+", "，", s)
    return s.strip()


def _parse_variants(text: str) -> list:
    s = (text or "").strip()
    if not s:
        return []
    if s.startswith("```"):
        s = s.strip("` \n\t")
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        try:
            import ast
            out = ast.literal_eval(s)
            return out if isinstance(out, list) else []
        except Exception:
            return []


def call_api(instruction: str, numbers_note: str, idx: int) -> list:
    content = build_prompt(instruction, numbers_note)
    for retry in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是高质量工程指令改写助手。"},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
            )
            out = resp.choices[0].message.content.strip()
            variants = _parse_variants(out)
            return [normalize_variant(v) for v in variants if isinstance(v, str)]
        except Exception:
            time.sleep(RETRY_BASE_DELAY ** retry)
    return []


def process_file(path_in: str, out_dir: str) -> None:
    out_path = os.path.join(out_dir, os.path.basename(path_in))
    if os.path.exists(out_path):
        return
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("alpaca_records") or []
    if not recs:
        return

    inst = recs[0].get("instruction") or ""
    numbers_note = build_numbers_note(data)
    variants = call_api(inst, numbers_note, 0)
    if not isinstance(variants, list):
        variants = []

    recs[0]["instruction_variants"] = variants
    for p in (data.get("dpo_pairs") or {}).get("code") or []:
        p["instruction_variants"] = variants

    out = dict(data)
    out["alpaca_records"] = recs

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] {os.path.basename(path_in)} -> {out_path}")


def main() -> None:
    files = []
    for dp, _, fns in os.walk(INPUT_FOLDER):
        for fn in fns:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(dp, fn))
    files.sort()
    if not files:
        print(f"[Error] no files in {INPUT_FOLDER}")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_file, p, OUTPUT_FOLDER) for p in files]
        for f in as_completed(futures):
            f.result()


if __name__ == "__main__":
    main()
