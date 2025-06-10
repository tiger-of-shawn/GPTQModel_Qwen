import subprocess
import os
import sys
import datetime # 导入 datetime 模块用于生成时间戳日志名
import argparse
import os


def run_benchmark(input_file: str, output_file: str, calc_script: str, batch_size: int, model_path: str, log_dir: str='logs'):
    """
    运行推理和计算任务，并将日志保存到独立文件。
    在运行前，确保输出文件是空的。

    Args:
        input_file (str): 输入数据文件路径。
        output_file (str): 输出结果文件路径。
        calc_script (str): 用于计算结果的 Python 脚本路径。
        batch_size (int): 批处理大小。
    """
    if 'AWQ' in model_path:
        inference_script = "inference_awq.py"
    elif 'GPTQ' in model_path:
        inference_script = "inference_gptq.py"
    else:
        print('wrong model type!!!!')
        exit(0)
    # 获取文件名的最后部分，用于友好的提示信息和日志文件名
    benchmark_base_name = os.path.basename(input_file).replace('.jsonl', '')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True) # 确保 logs 目录存在

    # 为每个基准测试创建独立的日志文件
    log_file_path = os.path.join(log_dir, f"{benchmark_base_name}_{timestamp}.log")

    print(f"--- 开始处理 {benchmark_base_name} (批处理大小: {batch_size}) inference_script:  {inference_script}---")
    print(f"日志将保存到: {log_file_path}")

    # --- 优化点 1: 预处理输出文件 ---
    if os.path.exists(output_file):
        print(f"检测到旧的输出文件 '{output_file}'，正在删除...")
        try:
            os.remove(output_file)
            print("旧文件删除成功。")
        except OSError as e:
            print(f"错误: 无法删除旧文件 '{output_file}': {e}")
            sys.exit(1)

    # 打开日志文件，用于重定向标准输出和标准错误
    with open(log_file_path, 'w') as log_f:
        # 构建推理命令
        inference_command = [
            "python", inference_script,
            "--input-file", input_file,
            "--output-file", output_file,
            "--batch-size", str(batch_size),
            "--model_path", model_path
        ]

        try:
            # 执行推理脚本，并将其输出重定向到日志文件
            print(f"正在执行推理命令: {' '.join(inference_command)}")
            subprocess.run(inference_command, check=True, text=True, stdout=log_f, stderr=log_f)
            print(f"推理成功: {input_file}")
        except subprocess.CalledProcessError as e:
            print(f"错误: 推理失败 ({input_file}). 详细信息请查看日志文件: {log_file_path}")
            # 由于 stdout/stderr 已重定向到文件，这里不打印 e.stdout/e.stderr
            sys.exit(1)

        # # 构建计算命令
        # calc_command = [
        #     "python", calc_script, output_file
        # ]

        # try:
        #     # 执行计算脚本，并将其输出重定向到日志文件
        #     print(f"正在执行计算命令: {' '.join(calc_command)}")
        #     subprocess.run(calc_command, check=True, text=True, stdout=log_f, stderr=log_f)
        #     print(f"计算成功: {output_file}")
        # except subprocess.CalledProcessError as e:
        #     print(f"错误: 计算失败 ({output_file}). 详细信息请查看日志文件: {log_file_path}")
        #     sys.exit(1)

    print(f"--- 完成处理 {benchmark_base_name} ---")
    print("-" * 50) # 添加分隔线增加可读性

# --- 定义所有基准测试 ---
BENCHMARKS = [
    {
        "input_file": "data/jsonls/omnibench.jsonl",
        "output_file": "output/omnibench.jsonl",
        "calc_script": "cal_omni.py",
        "batch_size": 4
    },
    {
        "input_file": "data/jsonls/libri_test_other.jsonl",
        "output_file": "output/libri_test_other.jsonl",
        "calc_script": "cal_wer.py",
        "batch_size": 4
    },
    {
        "input_file": "data/jsonls/wenetspeech_test_net.jsonl",
        "output_file": "output/wenetspeech_test_net.jsonl",
        "calc_script": "cal_wer.py",
        "batch_size": 4
    },
    {
        "input_file": "data/jsonls/mmlu_pro.jsonl",
        "output_file": "output/mmlu_pro.jsonl",
        "calc_script": "cal_mmlu.py",
        "batch_size": 4
    },
    {
        "input_file": "data/jsonls/videomme_50.jsonl",
        "output_file": "output/videomme_50.jsonl",
        "calc_script": "cal_vidmme.py",
        "batch_size": 1
    }
]

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行模型基准测试，支持通过命令行传入模型路径和CUDA设备ID。")
    parser.add_argument(
        "--model_path",
        type=str,
        default='/mnt/shawn/models/Qwen2.5-Omni-7B-AWQ-thinker-talker',
        help="指定模型的路径。默认值为 '/mnt/shawn/models/Qwen2.5-Omni-7B-AWQ-thinker-talker'。"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default='6',
        help="指定 CUDA 设备ID。例如 '0', '1', 或 '0,1'。默认值为 '6'。"
    )
    args = parser.parse_args()

    model_path = args.model_path
    cuda_device = args.cuda_device

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"--- CUDA_VISIBLE_DEVICES 已设置为: {os.environ['CUDA_VISIBLE_DEVICES']} ---")


    # 使用 os.path.basename 从 model_path 中提取目录名作为 postfix，或者您可以根据需要自定义
    # 为了避免文件名中的点和横杠可能导致的问题，这里进行了替换
    postfix = os.path.basename(model_path).replace('.', '_').replace('-', '_')

    print(f"--- 开始运行所有基准测试任务，模型路径: {model_path}---")
    for bench_config in BENCHMARKS:
        run_benchmark(
            input_file=bench_config["input_file"],
            output_file=bench_config["output_file"] + postfix,
            calc_script=bench_config["calc_script"],
            batch_size=bench_config["batch_size"],
            model_path=model_path,
            log_dir=f'logs_{postfix}'
        )
    print("--- 所有基准测试任务完成！---")