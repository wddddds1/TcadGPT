import os
import json
import re
import time
import openai
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 DeepSeek 客户端
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")

# 输入输出路径
input_path = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/code_enhance/generate_code_tasks.json"
output_path = "/Users/wddddds/TcadGPT/TcadGPT/data/processed_json/v13/code_enhance/generate_code_tasks_augmented.json"

def clean_json_response(text):
    return re.sub(r'```(json)?', '', text, flags=re.IGNORECASE).strip()

def generate_alternative_instructions(instruction, output_code):
    prompt = f"""
你是一个 TCAD 自然语言理解专家。用户的原始问题是：

Instruction:
{instruction}

Output:
{output_code}

请根据上述指令和代码，生成 5条以上 个风格各异、描述方式不同的 instruction，模拟用户的真实提问行为。要求有的问题描述的比较仔细，有的则只给一些大概的很粗略的要求。

输出格式为 JSON 数组，如：
["instruction1", "instruction2", ...]
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个训练数据增强专家，输出格式必须是 JSON 数组，每个元素是不同表述的 instruction。"},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        cleaned = clean_json_response(raw)
        return json.loads(cleaned)
    except Exception as e:
        print(f"[ERROR] 生成instruction失败: {e}")
        return []

def enhance_output_with_comments(output_code):
    prompt = f"""
你是一个 TCAD 教程撰写专家。请为以下 TCAD 代码添加详细注释，并在代码后附加一段自然语言的详细解释：

{output_code}

输出格式为纯文本：包含添加注释后的完整代码 + 一段解释性文字。
注释的添加符号一般是;，在后面添加注释即可。
解释性文字包括给出这段代码的详细解释，包括代码整体的逻辑和物理意义，各个参数的意义等。
不要包含 Markdown 标记或说明。
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个训练数据生成专家，返回注释后的代码和详细解释。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] 增强output失败: {e}")
        return output_code

def augment_single_qa(example, index=None, total=None, start_time=None):
    instruction = example["instruction"].strip()
    output_code = example["output"].strip()

    new_instructions = generate_alternative_instructions(instruction, output_code)
    enhanced_output = enhance_output_with_comments(output_code)

    new_examples = []
    for new_inst in new_instructions:
        new_example = {
            "instruction": new_inst.strip(),
            "input": "",
            "output": enhanced_output
        }
        new_examples.append(new_example)
        print("增强样本:", json.dumps(new_example, ensure_ascii=False))

    if index is not None and total is not None and start_time is not None:
        elapsed = time.time() - start_time
        avg_time = elapsed / (index + 1)
        remaining = avg_time * (total - index - 1)
        eta = datetime.now() + timedelta(seconds=remaining)
        print(f"[{index+1}/{total}] 处理完成，预计完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    return new_examples

def process_all(input_path, output_path, max_workers=1000):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    start_time = time.time()
    all_augmented = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(augment_single_qa, item, idx, total, start_time): idx for idx, item in enumerate(data)}
        for future in as_completed(futures):
            all_augmented.extend(future.result())

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_augmented, f, ensure_ascii=False, indent=2)

    print(f"增强完成，共生成样本数：{len(all_augmented)}")

if __name__ == "__main__":
    process_all(input_path, output_path)