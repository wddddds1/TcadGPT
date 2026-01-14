import os
import json
import time
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 DeepSeek 客户端
# client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")


input_folder = "data/sources/elmer/keyword_pair"
output_folder = "data/sources/elmer/alpaca_output"
os.makedirs(output_folder, exist_ok=True)


def build_prompt(paragraph, keyword):
    return (
        """
        你的任务是：根据提供的专业段落与关键词，生成结构化的中文问答数据，用于微调 Alpaca 格式模型。
        
        输出必须为 JSON 数组，每条数据结构如下：
        {
          "instruction": "用户问题",
          "input": "",
          "output": "系统回答"
        }
        
        请严格遵守以下规范：
        
        1. 只生成中文内容，不得出现英文。
        
        2. 段落可能为目录、参考文献等非技术内容，若不包含完整语义句或关键词相关信息（非仅提及关键词），请跳过该关键词，输出空数组：[]。
           同时，如果关键词并非合理的技术关键词，也请直接跳过该关键词。
        
        3. 问题与回答应根据关键词类型采用不同风格：
           - **概念/物理原理类**：定义、机制、原理、相关模型等。
           - **公式/变量类**：物理意义、计算方法、单位、建模方式等。
           - **仿真指令/命令类**：用途、格式、参数、典型用法、组合方式等。
           - **软件模块类**：功能、作用、启动方式、与其他模块的关系等。
           - **输入参数/配置项类**：作用、使用方式、有哪些可选参数、仿真用途、仿真影响、单位、取值范围、默认值等。
        
        4. 每个关键词应尽量覆盖其在段落中涉及的全部知识点，从多个角度提问，确保问答内容不重复、不等价表达。
        
        5. 回答应准确、清晰、有操作性，可包含命令、公式等专业信息。指令和参数类问题可简洁明了但需要给一个例子，概念定义公式类需要详细深入的解释。
        
        6. 示例提问风格（可参考，不局限）：
           - 在 ElmerSolver 中，如何设置线性系统的 GMRES 重启参数？
           - 非线性 Gauss–Seidel 迭代的主要优势是什么？
           - 在 sif 文件中，如何指定网格数据库目录？
           - Navier-Stokes 求解器的 Flow Model 选项支持哪些取值？
        
        7. 若段落中关键词信息重复出现，请为每个用途分别生成问题；同一关键词生成多个问题时，禁止仅改写语序或词汇。
        
        8. 输出必须为合法的 JSON 数组，仅返回数据本身，不得包含 Markdown 代码块标记（如```json）。
    
        9. 当关键词是指令或参数时，问题中尽量**不要直接出现指令名**，应通过功能描述模仿用户提问，回答中再清楚指出指令名与示例。
            比如当关键词是 BDF Order 时，不要问：
            "在 sif 文件中，BDF Order 的作用是什么？"然后回答"BDF Order 用于指定时间离散阶数..."
            而要问:"在 sif 文件中用于指定时间离散阶数的关键字是什么？"然后回答"关键字是 BDF Order，用法为...比如..."
            
        10. **严禁输出与段落内容无关或不可验证的信息**，尤其不得生成：
            - “根据上述段落”、“该例中”、“如图所示”等内容；
            - 微调训练中无法获取段落上下文，因此禁止出现引用段落的语言。
            
        11. 当关键词是指令或参数时，生成的问答对中必须要包含一个如下形式的问题: 
            "在(求解步骤，如 ElmerSolver)中，(具体功能的描述)的指令是什么？"，回答"(xx功能)的指令是，用法为xx，比如（举一个具体的例子。）"
            
        12. 当生成公式时，请确保其能正确渲染，请正确使用$或$$来包裹，在一段文本中的公式请用$来包裹。
        
        13. 务必注意不要在开头或结尾添加```json和```。
            
        """
        f"""
        段落内容如下：
        {paragraph}

        关键词：
        {keyword}
        """
    )


def call_api(paragraph, keyword, index):
    content = build_prompt(paragraph, keyword)
    for retry in range(2):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个高质量数据生成助手。"},
                    {"role": "user", "content": content}
                ]
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"[Thread-{index}] 请求失败，重试 {retry + 1}/5：{e}")
            time.sleep(2 ** retry)
    return None


def process_jsonl_file(file_path, output_path, stats):
    # 检查输出文件是否已存在且非空
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"⏩ 跳过已处理文件: {os.path.basename(file_path)}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_keywords = 0
    processed_keywords = 0
    output = []
    new_data_count = 0
    start_time = time.time()

    tasks = []
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for line in lines:
            data = json.loads(line)
            text = data["text"]
            keywords = data.get("keywords", [])
            total_keywords += len(keywords)
            for i, kw in enumerate(keywords):
                tasks.append(executor.submit(call_api, text, kw, i))

        for future in as_completed(tasks):
            result = future.result()
            if result:
                try:
                    blocks = json.loads(result)
                    if isinstance(blocks, list):
                        output.extend(blocks)
                        added_count = len(blocks)
                    else:
                        output.append(blocks)
                        added_count = 1

                    new_data_count += added_count
                    if new_data_count % 500 == 0:
                        print(f"✅ 已新增 {new_data_count} 条问答（当前文档累计：{len(output)}）")

                except Exception as e:
                    print(f"⚠️ JSON解析失败：{e}，原始内容：{result}")

            processed_keywords += 1
            stats["total_done"] += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / processed_keywords if processed_keywords else 0
            remain_current = avg_time * (total_keywords - processed_keywords)
            remain_total = avg_time * (stats["total_all"] - stats["total_done"])

            print(f"[{os.path.basename(file_path)}] 当前: {processed_keywords}/{total_keywords}，"
                  f"全局: {stats['total_done']}/{stats['total_all']}，平均: {avg_time:.1f}s，"
                  f"剩余: {remain_current:.1f}s（当前），{remain_total:.1f}s（总）")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in output:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    stats = {"total_all": 0, "total_done": 0}
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]

    # 首先统计总关键词数
    for file in jsonl_files:
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    stats["total_all"] += len(json.loads(line).get("keywords", []))
                except:
                    continue

    print(f"\n📊 总关键词数：{stats['total_all']}，文件数：{len(jsonl_files)}")

    for filename in jsonl_files:
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace(".jsonl", "_alpaca.jsonl")
        output_path = os.path.join(output_folder, output_filename)

        # 检查输出文件是否已存在
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # 如果文件已存在且非空，则跳过处理
            print(f"\n⏩ 跳过已处理文件: {filename}")

            # 更新已处理的关键词统计
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        stats["total_done"] += len(json.loads(line).get("keywords", []))
                    except:
                        continue
            continue

        print(f"\n🚀 开始处理文件: {filename}")
        process_jsonl_file(input_path, output_path, stats)

    print("\n✅ 所有处理完成！")
