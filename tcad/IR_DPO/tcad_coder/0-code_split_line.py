import os
import json
import time
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 DeepSeek 客户端
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")


def extract_valuable_lines(block_content):
    """提取非空、非注释、有意义的代码行"""
    lines = block_content.splitlines()
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith(';')]
    return lines


def annotate_lines_with_model(block_content, lines):
    """调用大模型为每一行生成解释"""
    prompt = f"""你将被给定一个半导体TCAD仿真代码块如下：

{block_content}

请将其中有意义的每一行单独提取出来，并为每一行添加中文解释。输出格式如下：
[
  {{
    "code_line": "具体代码行",
    "explanation": "中文解释"
  }},
  ...
]
注意：
- 保留原始代码行的格式，不能做任何改写；
- 跳过空行、注释行、无意义的格式行；
- 只输出JSON数组，不能有任何其他文字或代码标识。
"""
    retry = 0
    while retry < 3:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是TCAD仿真专家，擅长解释每一行仿真脚本的作用"},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content.strip()
            return json.loads(result)
        except Exception as e:
            print(f"标注请求失败（第 {retry+1} 次）：{e}")
            time.sleep(5 * (2 ** retry))
            retry += 1
    return []

def process_json_file(json_path, output_path):
    """读取原始JSON，逐块处理每行标注"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取失败 {json_path}: {e}")
        return

    logical_blocks = data.get("logical_blocks", [])
    annotated_blocks = []

    for block in logical_blocks:
        block_content = block.get("block_content", "")
        explanation = block.get("description", "")
        lines = extract_valuable_lines(block_content)
        if not lines:
            continue
        annotations = annotate_lines_with_model(block_content, lines)
        annotated_blocks.append({
            "block_description": explanation,
            "block_content": block_content,
            "annotated_lines": annotations
        })

    result = {
        "original_file": data.get("original_file"),
        "original_path": data.get("original_path"),
        "annotated_blocks": annotated_blocks,
        "success": len(annotated_blocks) > 0
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def is_already_processed(output_path):
    """检查输出文件是否已经成功生成且包含内容"""
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("success", False) and len(data.get("annotated_blocks", [])) > 0
    except Exception:
        return False


def walk_and_process_all(input_dir, output_dir, max_workers=20):
    """并发处理所有JSON文件，已处理文件自动跳过"""
    all_tasks = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.json') or file == 'failed_files.md':
                continue
            input_path = os.path.join(root, file)
            relative = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative)

            if is_already_processed(output_path):
                continue  # 已处理成功，跳过

            all_tasks.append((input_path, output_path))

    print(f"准备处理 {len(all_tasks)} 个新文件（跳过已完成）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_json_file, inp, out): inp
            for inp, out in all_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            inp = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"处理失败 {inp}: {e}")



if __name__ == "__main__":
    input_json_dir = '/data/processed_json/v13/split_cmd_code/code_block'
    output_annotated_dir = '/data/processed_json/v13/split_cmd_code/code_line'
    walk_and_process_all(input_json_dir, output_annotated_dir, max_workers=200)
