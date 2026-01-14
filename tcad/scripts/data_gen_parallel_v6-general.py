import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "Bearer sk-REDACTED"
# MODEL = "Pro/deepseek-ai/DeepSeek-V3"
MODEL = "deepseek-ai/DeepSeek-V2.5"

# 创建一个 Session 来复用连接，提高效率
session = requests.Session()
session.headers.update({
    "Authorization": API_KEY,
    "Content-Type": "application/json"
})

# 设定重试策略
retries = Retry(
    total=5,
    backoff_factor=1,       # 指数退避策略（1s, 2s, 4s, 8s, 16s）
    status_forcelist=[500, 502, 503, 504, 408],  # 这些状态码触发重试
    allowed_methods=["POST"]
)
session.mount("https://", HTTPAdapter(max_retries=retries))


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
                if len(section_text) >= 1000:
                    sections.append(section_text)
            current_section = [line.strip()]
        else:
            current_section.append(line.strip())

    if current_section:
        section_text = ' '.join(current_section).strip()
        if len(section_text) >= 10:
            sections.append(section_text)

    return sections


def generate_task(content, index, total_docs, session):
    start_time = time.time()
    print(f"线程 {index} 正在生成任务指令，处理内容:\n{content[:200]}...\n")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "你的任务是根据给定的文本生成高质量的微调数据集，在content中你将被给定一段专业领域的领域限定资料，你需要根据这段资料生成如下格式的数据"
                        "你的输出应该严格按照规定的格式：应该是一个json格式输出，包含三个元素，分别是instruction，input和output，以花括号包裹"
                        """
                          {
                            "instruction": "User instruction.",
                            "input": "",
                            "output": "Expected system output.",
                          },
                        """
                        "要求生成的问题必须跟材料相关，要专业和难，重点针对指令、代码、公式、操作步骤、科学原理、软件逻辑等。问题要有多样性，回答内容要丰富以及专业和结构化。"
                        "output应按照你对你生成的instruction的正常回复进行生成，你会怎么回答就怎么生成output。"
                        "一定注意instruction应该是通用的，不能出现比如“探讨本专著中...”等限定性的内容。"
                        "格式必须严格按照上述格式，并且重复生成，中间不能有任何其他符号。"
                        "提供给你的content中有时会有HTML标记或者没渲染的Latex公式代码，比如表格标记符号等，在你的生成中只输出用户可读的文字，公式一定要确保能正确渲染。"
                        "你看到的content主要是技术手册和领域教材。请你针对每个技术点、每个不同的指令都提一个问题，确保content中所有的重要的点都有相应的问题覆盖。"
                        "一些问题的例子如下：光学声子散射对迁移率有什么影响？"
                        "怎样在 Sprocess 中定义新的掺杂剂种类？"
                        "如何保存模拟结构为TDR文件？"
                        "离子掺杂工艺有哪些关键参数，如何设置仿真命令？"
                        "如何指定Transient命令中的时间步长控制？"
                        "当模拟金属 - 半导体接触时，应选择哪些物理模型？"
                        "在 Sentaurus Device 中，如何使用物理模型接口（PMI）来定义新的物理模型？"
                        "在 Sentaurus Visual 中，启动软件的命令行指令是什么？"
                        "在 Sentaurus Process 和 Sentaurus Interconnect 接口中，用于加载命令文件的按钮是什么？"
                        "Tcl 命令中，如何创建一个新的曲线数据集？"
                        "如何精确地在Hydrodynamic模型中考虑载流子的非平衡分布？"
                        "使用Sentaurus Device可以求解几种类型的温度？如果想要求解这些温度需要什么温度方程？"
                        "反型层中的量子化效应对迁移率有什么影响？"
                        "确保你生成的数据集包含所有content中重要的内容，尤其是关于指令代码的，哪怕是同种或相似的指令也要分别生成问题和回答。"
                        "不要限定生成问答的数量，要持续生成直到所有重要的内容都被覆盖。但一定注意不要生成相同的问题，每个问题都要有区别。"
                        "如果可能的话，生成的回答要简略，比如询问一个操作的指令是什么，就回答xx的指令是xx，再举个例子即可。"
                        "比如：在 Sprocess 中，如何设置氧化速率的控制参数？回答：通过 “pdbSet < 相关参数> < 值 >” 来设置，如 “pdbSet Diffuse MinGrowthStep < 最小生长步长 >” 和 “pdbSet Diffuse MaxGrowthStep < 最大生长步长 >”。"
                        "你生成的output不要局限于上面给出例子中的格式，而是根据需要调整合适的格式。"
                        "instruction和input的区别是：instruction是用户指令，input是对指令的补充，可以留空，不要翻译成中文，请选择摘取有意义的一段话。"
                        "你的output中应尽可能提供能够解决问题的指令、代码、公式等，用合适的格式描述。"
                        "不要在问题中包含指令，比如，不要问在 Sentaurus 中，如何使用 sde:set-window-position 命令设置窗口位置？而要问在 Sentaurus 中，设置窗口位置的命令是什么？并在output中给出sde:set-window-position这个指令。"
                        "再次强调，请尽可能生成最全面的数据集，涵盖提供给你的content中所有重要的内容，尤其是指令相关的。请尽可能的多生成数据。"
                        "请一定注意不要生成完全相同的问题，这点非常重要。但可以生成稍微不同的问题，比如对功能相似的指令各生成一个问题。"
                        "请用中文生成，instruction和output都必须用中文，严禁使用英文。"
                        "请一定确保content中所有的重要的信息都被某个问题提问到了，哪怕是非常相似的功能，也要分成两个问题提问，一定确保不能出现content中介绍了的功能但没有生成相关问题这种情况。"
                        "确保尽量多的生成问题！只要有不同就可以生成。每次至少生成10个问题，可以更多不能更少，但要确保每个问题都有不同。"},
            {"role": "user", "content": content}
        ],
        "stream": False,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 2,
        "response_format": {"type": "text"}
    }

    retry_count = 0
    max_retries = 10
    backoff_time = 1  # 初始重试等待时间

    while retry_count < max_retries:
        try:
            response = session.post(API_URL, json=payload)
            response.raise_for_status()  # 检查 HTTP 状态码
            response_json = response.json()

            choices = response_json.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                generated_content = message.get("content", "")
            else:
                generated_content = "错误: 返回的choices列表为空。"

            break  # 成功获取数据，退出循环

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"线程 {index} 请求失败（重试 {retry_count + 1}/{max_retries}）: {e}")
            retry_count += 1
            time.sleep(backoff_time)
            backoff_time *= 2  # 指数退避

            if retry_count == max_retries:
                generated_content = f"错误: 超过最大重试次数。最后错误: {e}"

    elapsed_time = time.time() - start_time
    print(f"线程 {index} 生成完成，耗时 {elapsed_time:.2f} 秒。\n生成内容:\n{generated_content[:200]}...\n")

    return generated_content


def data_gen(doc_path):
    docs = process_md_document(doc_path)
    total_docs = len(docs)
    output = []
    finished = 0

    print(f"开始并行处理文档，总共有 {total_docs} 部分待生成任务指令。\n")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(generate_task, content, i, total_docs, session): i for i, content in enumerate(docs, start=1)}

        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            finished += 1
            print(f"已完成 {finished}/{total_docs}。")

    print("所有部分并行生成完成。")
    return output


if __name__ == "__main__":
    save_path = 'data/resources/V2/V2_raw'
    md_path = 'data/resources/V2/V2_allmd'

    for filename in os.listdir(md_path):
        if filename.endswith(".md") and filename[:-3] + '.txt' not in os.listdir(save_path):
            file_path = os.path.join(md_path, filename)
            output = data_gen(file_path)

            output_filename = filename.replace('.md', '.txt')
            output_file_path = os.path.join(save_path, output_filename)

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write("\n".join(output))
