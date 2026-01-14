#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finalize Elmer DPO dataset (similar format to tcad_coder).
"""

import os
import json
import time
import hashlib
import threading
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FOLDER = "outputs/instruction_aug_elmer_dpo_full_v8"
OUTPUT_FOLDER = "outputs/cot_aug_elmer_dpo_full_v8"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_JSONL = os.path.join(OUTPUT_FOLDER, "dpo_elmer_dataset_full_v8.jsonl")

MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 128
MAX_RETRY = 5
RETRY_BASE_DELAY = 2.0

client = openai.Client(
    api_key="sk-REDACTED",
    base_url="https://api.deepseek.com"
)


def build_prompt(instruction_text: str) -> str:
    return f"""
你是 ELMER .sif 方向的中文助理。请写一段连续、自然、简洁的中文说明文字，
总结该指令的建模/求解与输出要点。不要使用小节标题，不要输出代码或反引号。
数值只来自 instruction；若未给出则用概括性描述。

instruction：
{instruction_text}
""".strip()


def call_api(instruction_text: str) -> str:
    content = build_prompt(instruction_text)
    for retry in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是高质量中文说明写作助手。"},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            time.sleep(RETRY_BASE_DELAY ** retry)
    return ""


def assemble_response(text: str, code: str) -> str:
    parts = []
    if text:
        parts.append(text)
        parts.append("")
    if code:
        parts.append("#### 完整代码\n")
        parts.append("```plaintext\n" + code.strip() + "\n```")
    return "\n".join(parts).rstrip()


def uid_for_pair(inst: str, chosen: str, rejected: str) -> str:
    raw = (inst or "") + "\n" + (chosen or "") + "\n" + (rejected or "")
    return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()


def load_existing_uids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    uids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("_uid")
                if uid:
                    uids.add(uid)
            except Exception:
                continue
    return uids


def process_file(path_in: str) -> list:
    with open(path_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    recs = data.get("alpaca_records") or []
    if not recs:
        return []
    base_inst = recs[0].get("instruction") or ""
    variants = recs[0].get("instruction_variants") or []
    inst_list = [v for v in variants if isinstance(v, str) and v.strip()]
    if not inst_list:
        inst_list = [base_inst] if base_inst.strip() else []

    out_items = []
    for inst in inst_list:
        paragraph = call_api(inst)
        for pair in (data.get("dpo_pairs") or {}).get("code") or []:
            chosen = pair.get("chosen") or ""
            rejected = pair.get("rejected") or ""
            uid = uid_for_pair(inst, chosen, rejected)
            out_items.append({
                "_uid": uid,
                "conversations": [{"from": "human", "value": inst}],
                "chosen": {"from": "gpt", "value": assemble_response(paragraph, chosen)},
                "rejected": {"from": "gpt", "value": assemble_response(paragraph, rejected)},
            })
    return out_items


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

    existing_uids = load_existing_uids(OUTPUT_JSONL)
    lock = threading.Lock()
    written = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_file, p) for p in files]
            for f in as_completed(futures):
                items = f.result()
                if not items:
                    continue
                with lock:
                    for item in items:
                        uid = item.get("_uid")
                        if uid and uid in existing_uids:
                            continue
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        if uid:
                            existing_uids.add(uid)
                        written += 1

    print(f"[Done] appended={written} -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
