import os
import json
import re
import time
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 DeepSeek 客户端
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")

input_folder = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/split_cmd_code/code_block"
output_folder = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/code_enhance/2-block_generation_and_comments"
os.makedirs(output_folder, exist_ok=True)

def clean_json_response(text):
    return re.sub(r'```(json)?', '', text, flags=re.IGNORECASE).strip()

def generate_block_alpaca(block_content):
    prompt = f"""
你是一个语料构造专家，目标是将以下 TCAD 仿真代码转换为两个高质量 Alpaca 格式的问答样本。

要求如下：
1. 输出为 JSON 数组（长度为2），每个元素结构为：{{"instruction": ..., "input": ..., "output": ...}}
2. 第一条问答：你需要根据这段代码的功能内容，反向模拟一个真实用户的自然语言请求，然后以该请求为 instruction，input 为空，output 为完整代码。
    问题要像人提的，尽量说明在tcad的哪个流程中而不是泛泛的说"请帮我编写一段TCAD仿真代码"。
3. 第二条问答：你需要将这段代码中的注释全部去除，然后以去除注释后的代码作为 input，然后模仿用户来提要求添加注释，output 中请输出 input 的代码增加非常翔实的注释。
4. 注意：
  - instruction 内容必须自然、具体，贴近真实科研或工程需求。
  - output 中的代码必须结构清晰，格式整洁，避免多余空格和注释冗余。
  - output 中不得含 markdown 标记或额外解释，必须是直接可训练的 JSON 数据。
  - 如果代码里设计到PATH，请修改成 /Path/to/your/file 之类的，而不要用原本的真实path。
  - TCAD代码添加注释的方式为 ;符号后面的都是注释，比如：
; Reinitializing SDE 
(sde:clear)

(define Wsi 0.5)  ; Width  of Silicon Region
(define Hsi 1.0)        ; Height of Silicon Region
(define Wpo 0.5)        ; Width  of Poly Region
(define Hpo 0.5)        ; Height of Poly Region

下面是待处理的 TCAD 仿真代码：
{block_content}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个训练数据生成专家，输出内容必须为可直接用于微调的 JSON 数组，每条为 Alpaca 格式问答样本。"},
                {"role": "user", "content": prompt}
            ]
        )
        raw_text = response.choices[0].message.content.strip()
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)

    except Exception as e:
        return []

def process_json_file(file_path, output_path, file_idx, total_files):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 0, 0, 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return 0, 0, 1

    alpaca_records = []
    block_count = 0
    for block in data.get("logical_blocks", []):
        block_content = block.get("block_content", "").strip()
        if block_content:
            block_count += 1
            records = generate_block_alpaca(block_content)
            alpaca_records.extend(records)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in alpaca_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{file_idx:3d}/{total_files:3d}] Processed {os.path.basename(file_path):<60} -> {len(alpaca_records)} samples")
    return len(alpaca_records), block_count, 0

def process_all(max_workers=100):
    files = []
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith(".json"):
                files.append(os.path.join(root, filename))

    total_files = len(files)
    tasks = []
    results = []

    start_time = time.time()
    print(f"[START] Processing {total_files} files with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, in_path in enumerate(files, 1):
            flat_name = os.path.relpath(in_path, input_folder).replace(os.sep, "__")
            out_path = os.path.join(output_folder, flat_name.replace(".json", "_alpaca.jsonl"))
            tasks.append(executor.submit(process_json_file, in_path, out_path, idx, total_files))

        for future in as_completed(tasks):
            results.append(future.result())

    elapsed = time.time() - start_time
    total_samples = sum(r[0] for r in results)
    total_blocks = sum(r[1] for r in results)
    total_failed = sum(r[2] for r in results)

    print("\n[SUMMARY]")
    print(f"  Total files:    {total_files}")
    print(f"  Total blocks:   {total_blocks}")
    print(f"  Total samples:  {total_samples}")
    print(f"  Failed files:   {total_failed}")
    print(f"  Elapsed time:   {elapsed:.1f} seconds")
    print(f"  Avg time/file:  {elapsed / total_files:.2f} seconds")

if __name__ == "__main__":
    process_all()
