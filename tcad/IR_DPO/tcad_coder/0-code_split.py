import os
import time
import json
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize DeepSeek client
client = openai.Client(api_key='sk-REDACTED', base_url="https://api.deepseek.com")


def read_cmd_file(file_path):
    """读取.cmd文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def get_relative_path_filename(input_dir, file_path):
    """
    根据输入目录和文件路径生成相对路径文件名
    例如：输入目录为 /data/sources，文件路径为 /data/sources/folder/file.cmd
         则返回 folder_file.json
    """
    # 获取相对路径并标准化
    rel_path = os.path.relpath(file_path, input_dir)
    # 替换路径分隔符为下划线，并去掉.cmd后缀
    filename = rel_path.replace(os.sep, '_').replace('.cmd', '.json')
    return filename


def should_skip_processing(cmd_path, output_dir, input_dir, force_reprocess):
    """检查是否应该跳过文件处理"""
    json_filename = get_relative_path_filename(input_dir, cmd_path)
    json_path = os.path.join(output_dir, json_filename)

    if not force_reprocess and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('success', False) and len(data.get('logical_blocks', [])) > 0:
                    return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


def process_cmd_file(file_path, output_dir, input_dir, index, total_files, fail_log_path, force_reprocess=False):
    """处理单个.cmd文件"""
    # 检查是否跳过处理
    if not force_reprocess and should_skip_processing(file_path, output_dir, input_dir, force_reprocess):
        print(f"跳过已处理文件: {os.path.basename(file_path)}")
        return True

    start_time = time.time()
    print(f"\n线程 {index} 正在处理文件: {os.path.basename(file_path)} ({index}/{total_files})")

    content = read_cmd_file(file_path)

    system_prompt = """你将被给定一个完整的半导体TCAD仿真脚本文件（.cmd格式）。你的任务是：
                    1. 分析并理解整个脚本的逻辑结构
                    2. 将脚本分割成多个逻辑块，每个逻辑块应该是一个完整的功能单元
                    3. 为每个逻辑块添加简要中文说明

                    逻辑块分割标准：
                    - 按功能划分（如初始化、网格定义、物理模型设置、求解器配置等）
                    - 按命令组划分（相关联的命令组应放在一起）
                    - 保持逻辑完整性（一个逻辑块应能独立完成一个子任务）

                    输出格式如下，请确保你的输出直接可以作为json文件被load，不要在开头添加```json这种标识：
                    {
                      "logical_blocks": [
                        {
                          "block_content": "块内容",
                          "description": "功能说明"
                        },
                        ...
                      ],
                      "original_file": "原文件名"
                    }
                    请严格按照格式输出，不能有任何变动。
                    块内容必须按照原始代码一模一样的输出，不能有任何一点点的变化或省略。
                    你的输出会被直接作为json文件load，因此请勿输出任何```json这种标识，直接输出json。
                    再次强调！开头一定不要输出```json！第一个字符一定是 {
                    """

    retry_count = 0
    max_retries = 3
    backoff_time = 30
    result_text = "{}"

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]
            )
            result_text = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"线程 {index} 请求失败（重试 {retry_count + 1}/{max_retries}）: {e}")
            retry_count += 1
            time.sleep(backoff_time)
            backoff_time *= 2
            if retry_count == max_retries:
                result_text = "{}"

    # 创建失效json目录
    failed_json_dir = os.path.join(output_dir, "失效json")
    os.makedirs(failed_json_dir, exist_ok=True)

    try:
        parsed = json.loads(result_text)
        logical_blocks = parsed.get("logical_blocks", [])
        success = len(logical_blocks) > 0
    except Exception as e:
        print(f"线程 {index} JSON解析失败：{e}")
        # 直接保存到失效json目录
        failed_filename = get_relative_path_filename(input_dir, file_path)
        failed_path = os.path.join(failed_json_dir, failed_filename)

        with open(failed_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"线程 {index} 已将失败文件保存到: {failed_path}")
        return False  # 直接返回，不继续处理

    # 准备结果数据
    result = {
        "original_file": os.path.basename(file_path),
        "original_path": file_path,
        "original_content": content,
        "logical_blocks": logical_blocks,
        "success": success,
        "process_time": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 安全保存结果（只有解析成功的才保存到输出目录）
    output_filename = get_relative_path_filename(input_dir, file_path)
    output_path = os.path.join(output_dir, output_filename)

    # 使用临时文件确保写入原子性
    temp_path = output_path + '.tmp'
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 重命名临时文件到目标文件
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
    except Exception as e:
        print(f"文件保存失败: {e}")
        return False

    elapsed = time.time() - start_time
    print(f"线程 {index} 完成 {os.path.basename(file_path)}，耗时 {elapsed:.2f}s，逻辑块数：{len(logical_blocks)}")

    if not success:
        with open(fail_log_path, 'a', encoding='utf-8') as fail_log:
            fail_log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            fail_log.write(f"文件处理失败: {os.path.basename(file_path)}\n")
            fail_log.write(f"文件路径: {file_path}\n")
            fail_log.write(f"错误内容: {content[:200]}...\n\n--- 文件分隔 ---\n\n")

    return success


def process_cmd_files_in_directory(input_dir, output_dir, force_reprocess=False):
    """处理目录中的所有.cmd文件"""
    fail_log_path = os.path.join(output_dir, 'failed_files.md')
    os.makedirs(output_dir, exist_ok=True)

    # 初始化失败日志
    with open(fail_log_path, 'w', encoding='utf-8') as fail_log:
        fail_log.write(f"处理失败的文件记录\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 收集所有.cmd文件
    cmd_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".cmd"):
                full_path = os.path.join(root, filename)
                cmd_files.append(full_path)

    total_files = len(cmd_files)
    print(f"找到 {total_files} 个.cmd文件待处理")
    if force_reprocess:
        print("警告: 强制重新处理模式已启用，将忽略已有结果文件")

    success_count = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for idx, file_path in enumerate(cmd_files):
            if not force_reprocess and should_skip_processing(file_path, output_dir, input_dir, force_reprocess):
                skipped_count += 1
                continue

            future = executor.submit(
                process_cmd_file,
                file_path,
                output_dir,
                input_dir,
                idx + 1,
                total_files,
                fail_log_path,
                force_reprocess
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"文件处理异常: {e}")

    # 统计逻辑块总数（只统计成功文件）
    block_count = 0
    processed_files = 0
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and filename != 'failed_files.md':
            try:
                with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('success', False):
                        block_count += len(data.get('logical_blocks', []))
                        processed_files += 1
            except:
                continue

    # 统计失效json数量
    failed_json_dir = os.path.join(output_dir, "失效json")
    failed_json_count = 0
    if os.path.exists(failed_json_dir):
        failed_json_count = len([f for f in os.listdir(failed_json_dir) if f.endswith('.json')])

    print(f"\n处理完成:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_count}")
    print(f"跳过已处理: {skipped_count}")
    print(f"失败文件: {total_files - success_count - skipped_count}")
    print(f"共提取逻辑块: {block_count} (来自 {processed_files} 个文件)")
    print(f"失效JSON文件: {failed_json_count} 个")
    print(f"失败记录保存在: {fail_log_path}")


if __name__ == "__main__":
    input_dir = '/data/sources/Applications_Library/semiconductor_database-main'
    output_dir = '/data/processed_json/v13/split_cmd_code/code_block'
    force_reprocess = False

    process_cmd_files_in_directory(input_dir, output_dir, force_reprocess)