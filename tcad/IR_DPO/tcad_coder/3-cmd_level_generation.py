import os
import json
import re
import time
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

# 初始化 DeepSeek 客户端
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")

input_folder = "/Users/wddddds/TcadGPT/TcadGPT/data/sources/Applications_Library"
output_file = "/data/processed_json/v13/code_enhance/cmd_level_tasks.jsonl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

output_lock = ThreadPoolExecutor(1)
processed_count = 0
processed_lock = ThreadPoolExecutor(1)

def clean_response(text):
    return re.sub(r'```(json)?', '', text, flags=re.IGNORECASE).strip()

def generate_instruction_and_explanation(cmd_content, max_retries=10, retry_delay=5):
    prompt = f"""
你是一个 TCAD 语料构造专家，请将下面这份完整的 TCAD .cmd 仿真文件内容转换为一个 Alpaca 格式的高质量训练数据。

要求如下：
1. 输出为一个 JSON 对象，格式为：{{"instruction": ..., "input": "", "output": <解释部分>}}，其中：
   - instruction 是模拟真实用户的自然语言请求，说明这个 .cmd 文件的仿真目标、器件类型、预期生成完整仿真脚本等，必须为中文。
   - output 是一段中文解释，详细说明这段 .cmd 文件的仿真流程、主要结构设置、关键参数、仿真目标及物理意义等。**不要输出任何原始代码，仅输出解释文字。**
2. 如果出现一些特殊的路径，请改为 /Path/to/your/file 之类。
3. 返回内容必须是纯 JSON 格式，不能加 Markdown 格式包裹或额外说明。

以下是 .cmd 文件内容：
{cmd_content}
"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个训练数据生成专家，请只输出 JSON 格式的 instruction 和解释 output，input 保持为空。"},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_text = response.choices[0].message.content.strip()
            cleaned = clean_response(raw_text)
            return json.loads(cleaned)

        except Exception as e:
            print(f"[RETRY {attempt}/{max_retries}] Failed on generation: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                print("[ERROR] All retries failed.")
                return None

def process_cmd_file(file_path, file_idx, total_files, start_time):
    global processed_count
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cmd_content = f.read().strip()
    except:
        return 0, 1

    record_raw = generate_instruction_and_explanation(cmd_content)
    if not record_raw:
        return 0, 1

    final_record = {
        "instruction": record_raw["instruction"],
        "input": "",
        "output": cmd_content + "\n\n; 以下是对上述 TCAD 脚本的总结和解释：\n" + record_raw["output"]
    }

    with output_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    with processed_lock:
        processed_count += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed_count
        remaining = total_files - processed_count
        eta = timedelta(seconds=int(avg_time * remaining))
        print(f"Processed {processed_count} / {total_files} | Avg time/file: {avg_time:.2f}s | Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta} => {os.path.basename(file_path)}")

    return 1, 0

def process_all_cmds(max_workers=400):
    files = []
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith(".cmd"):
                files.append(os.path.join(root, filename))

    total_files = len(files)
    tasks = []
    results = []
    start_time = time.time()
    print(f"[START] Processing {total_files} .cmd files with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, in_path in enumerate(files, 1):
            tasks.append(executor.submit(process_cmd_file, in_path, idx, total_files, start_time))

        for future in as_completed(tasks):
            results.append(future.result())

    elapsed = time.time() - start_time
    total_samples = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)

    print("\n[SUMMARY]")
    print(f"  Total files:    {total_files}")
    print(f"  Total samples:  {total_samples}")
    print(f"  Failed files:   {total_failed}")
    print(f"  Elapsed time:   {elapsed:.1f} seconds")
    print(f"  Avg time/file:  {elapsed / total_files:.2f} seconds")

if __name__ == "__main__":
    process_all_cmds()
