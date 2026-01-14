import os
import time
import json
import re
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-REDACTED"
MODEL = "deepseek-chat"

client = openai.Client(api_key=API_KEY, base_url="https://api.deepseek.com")

def process_md_document(file_path):
    sections = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    content = [line for line in content if not (line.startswith("![]") or line.startswith("Figure"))]

    current_section = []
    for line in content:
        if line.startswith('#'):
            if current_section:
                section_text = ' '.join(current_section).strip()
                if len(section_text) >= 100:
                    sections.append(section_text)
            current_section = [line.strip()]
        else:
            current_section.append(line.strip())

    if current_section:
        section_text = ' '.join(current_section).strip()
        if len(section_text) >= 100:
            sections.append(section_text)

    return sections

def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', text)
        return json.loads(repaired)

def generate_keywords(content, index, total_docs, fail_log_path):
    start_time = time.time()
    print(f"\n线程 {index} 正在处理：\n{content[:200]}...\n")

    retry_count = 0
    max_retries = 3
    backoff_time = 30
    result_text = "{}"

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你将被给定一段英文或中文的 ELMER（有限元多物理场）技术资料。你的任务是：\n"
                                   "请从中提取所有可以进一步提问的关键词，关键词应包括但不限于：\n"
                                   "- 指令名（如 Solver, Body Force, Mesh DB）\n"
                                   "- 参数名（如 BDF Order, Linear System GMRES Restart）\n"
                                   "- 模型名（如 Navier-Stokes, Reynolds）\n"
                                   "- 软件模块名（如 ElmerSolver, ElmerGrid）\n"
                                   "- 所涉及的物理机制或方程（如 对流扩散方程、热传导）\n"
                                   "\n"
                                   "输出格式如下：\n"
                                   "{\n"
                                   "  \"keywords\": [\"关键词1\", \"关键词2\", ...]\n"
                                   "}\n"
                                   "请严格按照格式输出，不能有任何变动。"
                                   "请注意：\n"
                                   "- 如果文本无技术内容，如目录页，请不要生成关键词\n"
                                   "- 关键词保持与原文一致（中文提中文，英文提英文）"
                                   "- 如果给的文本中不包含技术相关的内容，请勿提取不相关的关键词，包括书籍信息页、目录页等\n"
                                   "- 关键词列表要覆盖内容中所有可提问的技术术语\n"
                                   "- 不要遗漏重要模型、操作命令或参数名\n"
                                   "- 如果设计指令或代码，务必挖掘所有出现的指令和代码，每个指令都要作为一个关键词提取。\n"
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=4096,
                temperature=0.3,
            )
            result_text = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"线程 {index} 请求失败（重试 {retry_count + 1}/{max_retries}）: {e}")
            retry_count += 1
            time.sleep(backoff_time)
            backoff_time *= 2

    try:
        parsed = safe_json_loads(result_text)
        keywords = parsed.get("keywords", [])
    except Exception as e:
        print(f"线程 {index} JSON解析失败：{e}")
        print("原始输出内容：", result_text)
        keywords = []

    elapsed = time.time() - start_time
    print(f"线程 {index} 完成，耗时 {elapsed:.2f}s，关键词数：{len(keywords)}")

    if len(keywords) == 0:
        with open(fail_log_path, 'a', encoding='utf-8') as fail_log:
            fail_log.write(content.strip() + "\n\n--- 段落分隔 ---\n\n")

    print(
        "text: ", content, "\n"
        "keywords: ", keywords, "\n"
    )

    return {
        "text": content,
        "keywords": keywords,
        "success": len(keywords) > 0
    }

def extract_keywords_from_file(doc_path, fail_log_path):
    docs = process_md_document(doc_path)
    total_docs = len(docs)
    output = []
    success_count = 0
    keyword_total = 0

    print(f"\n开始处理文件：{doc_path}（共 {total_docs} 段）")

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(generate_keywords, content, i, total_docs, fail_log_path): i
            for i, content in enumerate(docs, start=1)
        }

        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            if result["success"]:
                success_count += 1
                keyword_total += len(result["keywords"])

    fail_count = total_docs - success_count
    print(f"\n统计结果：成功 {success_count} 段，失败 {fail_count} 段，提取关键词总数 {keyword_total}\n")

    return output

if __name__ == "__main__":
    input_md_dir = '原始数据'
    output_jsonl_dir = 'data/sources/elmer/keyword_pair'
    fail_log_path = os.path.join(output_jsonl_dir, 'failed_segments.md')

    os.makedirs(output_jsonl_dir, exist_ok=True)

    for root, _, files in os.walk(input_md_dir):
        for filename in files:
            if filename.endswith(".md"):
                relative_path = os.path.relpath(os.path.join(root, filename), input_md_dir)
                jsonl_name = relative_path.replace('.md', '.jsonl').replace(os.sep, '__')
                output_file_path = os.path.join(output_jsonl_dir, jsonl_name)

                if os.path.exists(output_file_path):
                    print(f"跳过已处理文件：{relative_path}")
                    continue

                file_path = os.path.join(root, filename)
                output = extract_keywords_from_file(file_path, fail_log_path)

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for item in output:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                print(f"{relative_path} 提取关键词完成，结果保存在：{output_file_path}\n")
