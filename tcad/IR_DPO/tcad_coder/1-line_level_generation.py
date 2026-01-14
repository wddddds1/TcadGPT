import os
import json
import re
import time
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")

input_folder = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/split_cmd_code/code_line"  # 替换为你的输入路径
output_folder = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/code_enhance/1-line_generation"  # 替换为你的输出路径
os.makedirs(output_folder, exist_ok=True)

def clean_json_response(text):
    return re.sub(r'```(json)?', '', text, flags=re.IGNORECASE).strip()

def generate_line_qa(code_line):
    prompt = f"""
你是一个专业的 TCAD 训练数据构造专家，下面是一段 TCAD 脚本中的单行代码：

{code_line}

请为这段代码反向构造一个 Alpaca 格式的问答对。
要求如下：
- instruction 是根据code_line生成的一段真实科研用户可能会提的要求，描述他们想完成的任务。
- input 留空。
- output 是根据code_line写的 代码，注释，以及对这行代码每个参数的解释（注释一般写在;符号后面），并且在最后尽量添加物理意义或原理的详细解释，
解释这行代码是干啥的，在整体起到什么作用？
- 如果该行代码是注释、空行、花括号、或者语义不完整的代码（例如某段字符串的一部分），请不要返回任何内容。
- 输出格式是 JSON 对象，字段为 instruction, input, output。
- 请用中文输出。
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个训练数据生成专家，输出必须是结构化 JSON。"},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        cleaned = clean_json_response(raw)
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            return None
        return obj
    except:
        return None

def process_file(file_path, file_idx, total_files):
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename.replace(".json", "_lineqa.jsonl"))
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 0, 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return 0, 1

    output_records = []
    for block in data.get("annotated_blocks", []):
        for line in block.get("annotated_lines", []):
            code = line.get("code_line", "").strip()
            if not code or code.startswith(";") or code.startswith("//") or code in ["{", "}"]:
                continue
            result = generate_line_qa(code)
            if result:
                output_records.append(result)

    if output_records:
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in output_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[{file_idx:3d}/{total_files:3d}] {filename:<50} -> {len(output_records)} lines")
    return len(output_records), 0

def process_all(max_workers=200):
    files = []
    for root, _, filenames in os.walk(input_folder):
        for f in filenames:
            if f.endswith(".json"):
                files.append(os.path.join(root, f))

    total_files = len(files)
    results = []
    start = time.time()
    print(f"[START] Processing {total_files} files with {max_workers} threads")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f, i+1, total_files) for i, f in enumerate(files)]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.time() - start
    total_lines = sum(r[0] for r in results)
    failed = sum(r[1] for r in results)
    print("\n[SUMMARY]")
    print(f"  Total files:   {total_files}")
    print(f"  Total lines:   {total_lines}")
    print(f"  Failed files:  {failed}")
    print(f"  Time elapsed:  {elapsed:.1f}s")

if __name__ == "__main__":
    process_all()