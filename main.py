#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimultaneousTranslation主执行脚本
实现：
1. 获取数据集（SimultaneousTranslation 的 data/test/used 下已生成的数据集）
2. 转换数据集为parquet格式，并根据draft mode模板生成完整的prompt文本（包含instruction、格式说明和用户输入）
3. 在外部加载XCOMET模型，方便后续评分环节直接调用
4. 调用Qwen2.5-3B，使用步骤2生成的prompt，Mode=draft，生成并获取回答
5. 检查格式是否正确，如果格式正确，则抽取其中的<translate>部分，作为初稿，格式分数记为1；如果格式不正确，则不做更多操作，直接将格式分数记为0
6. 对格式正确的翻译调用XCOMET，输出有错误的部分
7. 再次调用Qwen2.5-3B，Mode=repair，prompt中包含：mode、原文、初稿和错误的span，让模型再次思考，获取回答
8. 检查repair生成回答格式是否正确，如果格式正确，则抽取其中的<translate>部分，作为终稿，复盘格式分数记为1；如果格式不正确，则不做更多操作，直接将复盘格式分数记为0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd
from datasets import Dataset

# 设置tokenizers并行性，避免fork后的警告
# 在导入transformers之前设置，避免tokenizers在fork前使用并行性
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.process_data import read_jsonl_files, make_prefix
from transformers import AutoTokenizer
from xcomet_loader import XCOMETLoader
from qwen_generator import QwenGenerator
from utils import check_and_extract_translate_tag, format_error_spans_for_prompt, split_into_segments

def _parse_gpu_list(gpu_str: Optional[str]) -> List[int]:
    """把类似 '0,1,4' 的字符串转成 [0,1,4]，None 或 '' 返回空列表。"""
    if not gpu_str:
        return []
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]


def map_physical_to_logical(physical_gpus: Optional[str]) -> List[int]:
    """
    把“物理 GPU 序号”（用户通过 --xcomet_gpus / --qwen_gpus 传进来的）
    映射成当前进程下的“逻辑 GPU id”（torch / Lightning / vLLM 看见的那套）。

    规则：
    - 如果没有设置 CUDA_VISIBLE_DEVICES：物理 == 逻辑，直接返回用户的列表；
    - 如果设置了 CUDA_VISIBLE_DEVICES，例如 '1,2,4'：
        这一串就是“可见的物理 GPU 列表”。
        我们把用户传进来的物理 id 映射到这个列表里的 index：
          CUDA_VISIBLE_DEVICES=1,2,4
          → 逻辑 0 -> 物理 1
          → 逻辑 1 -> 物理 2
          → 逻辑 2 -> 物理 4
        例如用户传 --xcomet_gpus 1,4 → 逻辑 id = [0,2]

    映射失败（用户给了不在可见列表里的物理 GPU，比如 env=1,2,4 但传 0）
    就直接报错。
    """
    phys_list = _parse_gpu_list(physical_gpus)
    if not phys_list:
        return []

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # Case 1: 未设 CUDA_VISIBLE_DEVICES，物理 == 逻辑
    if not visible_env:
        return phys_list

    visible_phys = _parse_gpu_list(visible_env)  # 例如 [1,2,4]

    logical_ids: List[int] = []
    for g in phys_list:
        if g not in visible_phys:
            raise ValueError(
                f"[GPU Mapping Error] 物理 GPU {g} 不在 CUDA_VISIBLE_DEVICES={visible_env} 中，"
                f"请确保二者一致，或取消外部的 CUDA_VISIBLE_DEVICES 限制。"
            )
        logical_ids.append(visible_phys.index(g))
    return logical_ids


def report_gpu_mapping(role: str, phys_gpus: Optional[str], logical_ids: List[int]):
    """打印一下映射情况（方便你 debug）。"""
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if not phys_gpus:
        print(f"[{role}] 使用 CPU（未指定 --{role.lower()}_gpus）")
        return

    print(f"[{role}] 用户传入的 GPU（物理）: {phys_gpus}")
    if visible_env:
        print(f"[{role}] 当前 CUDA_VISIBLE_DEVICES = {visible_env}")
        print(f"[{role}] 映射后的逻辑 GPU id = {logical_ids}")
    else:
        print(f"[{role}] 未设置 CUDA_VISIBLE_DEVICES，物理 == 逻辑 = {logical_ids}")


def log_stats(msg: str, output_file: str = None):
    """
    同时打印到终端 & 追加写入一个全局的 stats 日志 txt。
    如果提供了 output_file，会根据它所在目录创建日志文件；
    否则默认写在当前工作目录。
    """
    print(msg)

    # 确定日志文件路径（比如和 output_file 在同一目录）
    if output_file:
        log_dir = os.path.dirname(os.path.abspath(output_file))
    else:
        log_dir = os.getcwd()

    log_path = os.path.join(log_dir, "xcomet_all_stats.txt")

    # 追加写入
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        
def load_dataset_from_test_files(test_files: List[str], base_dir: str = None):
    """
    从训练文件中加载数据集
    
    Args:
        test_files: 训练文件路径列表
        base_dir: 基础目录，如果文件路径是相对路径则基于此目录
    
    Returns:
        处理后的数据列表
    """
    if base_dir:
        test_files = [os.path.join(base_dir, f) if not os.path.isabs(f) else f for f in test_files]
    
    print(f"[Dataset] Loading training files: {test_files}")
    processed_data = read_jsonl_files(test_files)
    print(f"[Dataset] Loaded {len(processed_data)} samples")
    return processed_data


def process_data_for_qwen(data: List[Dict], tokenizer_path: str, tokenizer=None, output_file: Optional[str] = None):
    """
    将数据转换为Qwen2.5-3B可以处理的格式，并生成完整的prompt文本
    
    此函数不仅进行格式转换（JSONL -> Parquet），还会：
    1. 根据draft mode模板生成完整的prompt文本（包含instruction、格式说明和用户输入）
    2. 将生成的prompt保存到数据集的"prompt"字段中
    3. 保存为parquet格式以便后续使用
    
    Args:
        data: 原始数据列表（包含src_text, tgt_text等字段）
        tokenizer_path: tokenizer路径（用于chat模板，draft mode通常不需要）
        tokenizer: 可选的tokenizer对象（如果已加载）
        output_file: 输出文件路径（可选，如果提供则保存为parquet）
    
    Returns:
        处理后的数据集（包含prompt字段）
    """
    if tokenizer is None:
        print(f"[Process] Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    processed_samples = []
    for idx, example in enumerate(tqdm(data, desc="Processing data")):
        # 生成draft mode的prompt（复用base模板）
        prompt = make_prefix(example, template_type='draft', tokenizer=tokenizer)
        
        processed_sample = {
            "data_source": example.get('data_source', 'unknown'),
            "lang_pair": example.get('lg', 'unknown-unkown'),
            "src_text": example.get('src_text', ''),
            "tgt_text": example.get('tgt_text', ''),
            "prompt": prompt,
            "mode": "draft",
            "index": idx,
        }
        processed_samples.append(processed_sample)
    
    dataset = Dataset.from_list(processed_samples)
    
    if output_file:
        print(f"[Process] Saving processed dataset to {output_file}...")
        dataset.to_parquet(output_file)
        print(f"[Process] Dataset saved successfully")
    
    return dataset


def get_xcomet_gpu_setting(args):
    """
    统一的XCOMET GPU选择逻辑
    
    Returns:
        tuple: (xcomet_gpu_ids, should_set_env)
            - xcomet_gpu_ids: GPU ID字符串（如"0,1"）或None（CPU模式）
            - should_set_env: 是否需要设置CUDA_VISIBLE_DEVICES
    """
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if args.xcomet_cpu:
        return None, False
    elif args.xcomet_gpus:
        return args.xcomet_gpus, True
    elif original_cuda_visible and not args.qwen_gpus:
        # 如果只设置了CUDA_VISIBLE_DEVICES，且没有分别设置qwen_gpus，则两者都使用CUDA_VISIBLE_DEVICES
        return original_cuda_visible, True
    else:
        # 默认CPU模式
        return None, False


def main():
    parser = argparse.ArgumentParser(description='SimultaneousTranslation主脚本')
    
    # 数据集相关参数
    parser.add_argument('--data_dir', type=str, default='data', help='数据根目录（默认项目根目录下的 data）')
    parser.add_argument(
        '--test_files',
        nargs='+',
        default=None,
        help='（可选）要使用的数据集文件名，位于 data/test/used 下；'
             '如果不指定，将自动扫描 data/test/used 目录下所有 *.parquet / *.jsonl'
    )
    parser.add_argument('--tokenizer_path', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                       help='Tokenizer路径或HuggingFace模型ID')
    parser.add_argument('--pipeline_mode', type=str, choices=['baseline', 'extended'], default='baseline',
                       help='处理流程模式：baseline或extended（短句级refinement）')
    
    # XCOMET相关参数
    parser.add_argument('--xcomet_ckpt', type=str, default=None,
                       help='XCOMET checkpoint路径。如果为None，将从环境变量WORD_QE_CKPT获取')
    parser.add_argument('--load_xcomet', action='store_true', default=True,
                       help='是否加载XCOMET模型')
    parser.add_argument('--xcomet_cpu', action='store_true', default=False,
                       help='强制XCOMET使用CPU模式（避免CUDA错误）')
    parser.add_argument('--xcomet_gpus', type=str, default=None,
                       help='XCOMET使用的GPU编号，如"0,1"（默认：自动分配）')
    
    # Qwen模型相关参数
    parser.add_argument('--qwen_model_path', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                       help='Qwen2.5-3B模型路径或HuggingFace模型ID')
    parser.add_argument('--use_vllm', action='store_true', default=True,
                       help='是否使用vllm进行推理（推荐）')
    parser.add_argument('--no_use_vllm', dest='use_vllm', action='store_false',
                       help='不使用vllm，使用transformers')
    parser.add_argument('--qwen_gpus', type=str, default=None,
                       help='Qwen使用的GPU编号，如"2,3"（默认：使用CUDA_VISIBLE_DEVICES中剩余的GPU）')
    parser.add_argument('--qwen_cpu', action='store_true', default=False,
                       help='强制Qwen使用CPU模式（不使用GPU）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,
                       help='vLLM的GPU内存使用率（0.0-1.0），默认0.85（test_time: 0.85）')
    parser.add_argument('--vllm_max_num_seqs', type=int, default=64,
                       help='vLLM并发序列上限，减小可降低warm-up显存压力（默认64）')
    
    # 生成参数（参考 test_time/vllm_infer.py 的配置）
    parser.add_argument('--max_tokens', type=int, default=2048, help='最大生成token数（test_time: 2048）')
    parser.add_argument('--temperature', type=float, default=0.2, help='采样温度（test_time: 0.2）')
    parser.add_argument('--top_p', type=float, default=0.95, help='nucleus sampling参数（test_time: 0.95）')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小（test_time: 16）')
    parser.add_argument('--xcomet_batch_size', type=int, default=32, help='XCOMET批处理大小（默认32）')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='处理的样本数量（None表示处理全部）')
    
    # 输入/输出参数
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件路径（JSON格式，保存生成结果）')
    
    args = parser.parse_args()
    extended_mode = args.pipeline_mode == 'extended'

    # ===== 统一解析 GPU 参数：把物理 id → 逻辑 id =====
    # 注意：这里不改 CUDA_VISIBLE_DEVICES，只是做一个“解释层”。
    try:
        xcomet_logical_gpus = map_physical_to_logical(args.xcomet_gpus)
        qwen_logical_gpus = map_physical_to_logical(args.qwen_gpus)
    except ValueError as e:
        # 用户传了不在 CUDA_VISIBLE_DEVICES 里的物理 GPU，直接报错退出
        print(str(e))
        sys.exit(1)

    # 打印一下映射情况（可选）
    report_gpu_mapping("XCOMET", args.xcomet_gpus, xcomet_logical_gpus)
    report_gpu_mapping("Qwen", args.qwen_gpus, qwen_logical_gpus)

    
    # ========== 0. GPU环境检测和分配（仅打印，不修改环境） ==========
    print("\n" + "="*60)
    print("GPU环境检测和分配")
    print("="*60)
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ 检测到 {num_gpus} 个GPU可用")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # 显示当前的 GPU 使用计划（这里只是打印，不做真正绑定）
        original_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        # XCOMET 计划使用
        if args.xcomet_cpu:
            print(f"\n[XCOMET] 计划使用: CPU（--xcomet_cpu）")
        elif args.xcomet_gpus:
            print(f"\n[XCOMET] 计划使用物理 GPU: {args.xcomet_gpus}")
        elif original_cuda_env and not args.qwen_gpus:
            print(f"\n[XCOMET] 计划使用环境变量 CUDA_VISIBLE_DEVICES={original_cuda_env}")
        else:
            print(f"\n[XCOMET] 计划使用: CPU（默认）")
        
        # Qwen 计划使用
        if args.qwen_cpu:
            print(f"[Qwen] 计划使用: CPU（--qwen_cpu）")
        elif args.qwen_gpus:
            print(f"[Qwen] 计划使用物理 GPU: {args.qwen_gpus}")
        elif original_cuda_env and not args.xcomet_gpus:
            print(f"[Qwen] 计划使用环境变量 CUDA_VISIBLE_DEVICES={original_cuda_env}")
        else:
            print(f"[Qwen] 计划使用: CPU（默认）")
    else:
        print("✗ 未检测到GPU，将使用CPU模式（非常慢）")
    

    # ========== 1. 获取数据集 ==========
    print("\n" + "="*60)
    print("步骤1: 获取数据集（data/test/used 下的已生成数据）")
    print("="*60)

    # data_dir 仍然是项目根目录下的 data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(root_dir, args.data_dir)
    used_dir = Path(data_dir) / "test" / "used"

    if not used_dir.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {used_dir}")

    # 1）优先使用参数里显式指定的文件名（相对于 data/test/used）
    if args.test_files:
        test_files = [str(used_dir / f) for f in args.test_files]
    else:
        # 2）否则自动扫描 data/test/used 目录：
        #    - 如果有 *.parquet，就全部用 parquet
        #    - 否则用所有 *.jsonl
        parquet_candidates = sorted(used_dir.glob("*.parquet"))
        jsonl_candidates = sorted(used_dir.glob("*.jsonl"))

        if parquet_candidates:
            test_files = [str(p) for p in parquet_candidates]
            print(f"[Dataset] 在 {used_dir} 下发现 {len(test_files)} 个 parquet 文件，将直接加载：")
        elif jsonl_candidates:
            test_files = [str(p) for p in jsonl_candidates]
            print(f"[Dataset] 在 {used_dir} 下发现 {len(test_files)} 个 jsonl 文件，将先转换为 parquet：")
        else:
            raise FileNotFoundError(f"在 {used_dir} 下没有找到任何 *.parquet 或 *.jsonl 文件")

    print(f"[Dataset] 使用的文件列表：")
    for p in test_files:
        print(f"  - {p}")

    # ========== 2. 转换数据集为Qwen格式并生成prompt ==========
    print("\n" + "="*60)
    print("步骤2: 转换数据集为parquet格式并生成prompt（draft mode，复用rl模板）")
    print("="*60)

    # 加载tokenizer（用于生成prompt和后续repair mode）
    print(f"[Process] Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 为每个数据文件分别处理并保存parquet
    all_datasets = []
    parquet_files = []

    # 这里不再用 data/train/parquet，而是直接把 parquet 放在 data/test/used 下面
    parquet_dir = used_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Process] Parquet文件将保存到: {parquet_dir}")

    for train_file in test_files:
        data_file_path = Path(train_file)
        parquet_filename = data_file_path.stem + '.parquet'
        parquet_file_path = parquet_dir / parquet_filename
        parquet_files.append(str(parquet_file_path))

        # 如果本身就是 parquet，并且已经在 used_dir 里，直接加载
        if data_file_path.suffix == ".parquet" and data_file_path.exists():
            print(f"[Process] 直接加载已有的parquet文件: {data_file_path}")
            dataset = Dataset.from_parquet(str(data_file_path))
            print(f"[Process] 成功加载 {len(dataset)} 个样本")
        # 如果目标 parquet 已经存在，也直接加载（例如之前已经从 jsonl 转好）
        elif parquet_file_path.exists():
            print(f"[Process] 发现已处理的parquet文件: {parquet_file_path}")
            print(f"[Process] 直接加载，跳过数据转换...")
            dataset = Dataset.from_parquet(str(parquet_file_path))
            print(f"[Process] 成功加载 {len(dataset)} 个样本")
        else:
            # 否则视为 jsonl，读取后转换
            print(f"[Process] 未找到parquet文件，将从 JSONL 转换并保存: {parquet_file_path}")

            file_data = []
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        file_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[Line {line_num}] JSON parse failed → Line content: {repr(line)}")

            # 转换为Qwen格式并保存
            dataset = process_data_for_qwen(
                file_data,
                tokenizer_path=args.tokenizer_path,
                tokenizer=tokenizer,
                output_file=str(parquet_file_path)
            )
            print(f"[Process] 成功转换并保存 {len(dataset)} 个样本到 {parquet_file_path}")

        # 为每个数据集添加文件来源信息，并重新编号索引
        dataset = dataset.map(
            lambda example, idx: {**example, "index": idx, "source_file": data_file_path.name},
            with_indices=True
        )
        all_datasets.append(dataset)
    
    # 合并所有数据集
    if len(all_datasets) > 1:
        print(f"[Process] 合并 {len(all_datasets)} 个数据集...")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(all_datasets)
        # 重新编号索引（合并后从0开始连续编号）
        dataset = dataset.map(lambda example, idx: {**example, "index": idx}, with_indices=True)
        print(f"[Process] 合并后共 {len(dataset)} 个样本")
    else:
        dataset = all_datasets[0]
        # 确保索引从0开始
        if len(dataset) > 0:
            dataset = dataset.map(lambda example, idx: {**example, "index": idx}, with_indices=True)
    
    # 如果指定了num_samples，只使用前N个样本
    if args.num_samples and len(dataset) > args.num_samples:
        print(f"[Dataset] 限制使用前 {args.num_samples} 个样本（总共 {len(dataset)} 个）")
        dataset = dataset.select(range(args.num_samples))
        # 重新编号索引
        dataset = dataset.map(lambda example, idx: {**example, "index": idx}, with_indices=True)
    
    
    # ========== 3. 加载XCOMET模型 ==========
    print("\n" + "="*60)
    print("步骤3: 加载XCOMET模型")
    print("="*60)

    xcomet_loader = None
    if args.load_xcomet:

        # 1. 找 checkpoint
        if args.xcomet_ckpt:
            xcomet_ckpt = args.xcomet_ckpt
        elif os.getenv("WORD_QE_CKPT"):
            xcomet_ckpt = os.getenv("WORD_QE_CKPT")
        else:
            default_ckpt = "/ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt"
            xcomet_ckpt = default_ckpt if os.path.exists(default_ckpt) else None

        if not xcomet_ckpt:
            print("[XCOMET] No checkpoint found, skipping.")
        else:
            # 2. CPU 模式
            if args.xcomet_cpu:
                print("[XCOMET] 强制使用 CPU 模式")
                xcomet_loader = XCOMETLoader(xcomet_ckpt, force_cpu=True, gpu_ids=None)

            # 3. GPU 模式
            else:
                if not xcomet_logical_gpus:
                    print("[XCOMET] 未指定 --xcomet_gpus，使用 CPU 以避免误用 GPU")
                    xcomet_loader = XCOMETLoader(xcomet_ckpt, force_cpu=True, gpu_ids=None)
                else:
                    print(f"[XCOMET] 使用逻辑 GPU id: {xcomet_logical_gpus}")
                    gpu_str = ",".join(str(i) for i in xcomet_logical_gpus)
                    xcomet_loader = XCOMETLoader(xcomet_ckpt, force_cpu=False, gpu_ids=gpu_str)

            print("[XCOMET] XCOMET model loaded successfully")
    else:
        print("[XCOMET] Skipping XCOMET loading (--load_xcomet=False)")
        
    # ========== 4. 调用Qwen2.5-3B生成draft翻译 ==========
    print("\n" + "="*60)
    print("步骤4: 调用Qwen2.5-3B生成draft翻译")
    print("="*60)

    print(f"[Qwen] Loading Qwen model from {args.qwen_model_path}...")

    # ===== Qwen 的 GPU：必须用物理 GPU id 来设置 CUDA_VISIBLE_DEVICES =====
    if args.qwen_cpu:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        print("[Qwen] 强制使用 CPU")
        qwen_visible_phys = None

    elif args.qwen_gpus:
        # 用户给的是物理 GPU id
        qwen_visible_phys = args.qwen_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = qwen_visible_phys
        print(f"[Qwen] 可见物理 GPU: {qwen_visible_phys}")

    else:
        # 默认为 CPU
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        qwen_visible_phys = None
        print("[Qwen] 默认使用 CPU（未指定 --qwen_gpus）")

    # ===== vLLM 参数 =====
    vllm_extra_kwargs = {}
    if args.use_vllm:
        if args.vllm_max_num_seqs:
            vllm_extra_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
        vllm_extra_kwargs["disable_custom_all_reduce"] = True

    # ===== 启动 Qwen =====
    try:
        qwen_generator = QwenGenerator(
            model_path=args.qwen_model_path,
            use_vllm=args.use_vllm and not args.qwen_cpu,
            device="cpu" if args.qwen_cpu else None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            **vllm_extra_kwargs
        )
        print("[Qwen] Qwen model loaded successfully")

    except Exception as e:
        print(f"[Qwen] Failed to load Qwen model: {e}")
        qwen_generator = QwenGenerator(
            model_path=args.qwen_model_path,
            use_vllm=False,
            device="cpu",
        )
        
    # ========== 阶段式处理：先完成所有阶段的所有数据，再进入下一阶段 ==========
    dataset_df = dataset.to_pandas()
    num_samples = len(dataset_df)
    print(f"\n[Stage] 开始阶段式处理，共 {num_samples} 个样本")
    
    # 初始化所有结果
    results = []
    for _, row in dataset_df.iterrows():
        results.append({
            "index": int(row['index']),
            "data_source": row['data_source'],
            "lang_pair": row['lang_pair'],
            "src_text": row['src_text'],
            "tgt_text": row['tgt_text'],
            "draft_prompt": row['prompt'],
        })
        
    if extended_mode:
        # ========== 扩展模式流程 ==========

        # ========== 阶段1：批量生成完整原文的初稿翻译 ==========
        print("\n" + "="*60)
        print("阶段1: 批量生成完整原文的初稿翻译")
        print("="*60)

        # 收集所有完整原文的prompt
        draft_prompts: List[str] = [r["draft_prompt"] for r in results]

        # 批量生成所有完整原文的初稿
        all_draft_generated_texts: List[str] = []
        for i in tqdm(range(0, len(draft_prompts), args.batch_size), desc="生成完整初稿"):
            batch_prompts = draft_prompts[i:i + args.batch_size]
            try:
                generated_texts = qwen_generator.generate_draft(
                    batch_prompts,
                    mode="draft",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                if isinstance(generated_texts, str):
                    generated_texts = [generated_texts]
                if len(generated_texts) != len(batch_prompts):
                    print(
                        f"[Warning] 生成文本数量({len(generated_texts)})与批次大小({len(batch_prompts)})不匹配"
                    )
                    generated_texts = list(generated_texts) + [""] * (
                        len(batch_prompts) - len(generated_texts)
                    )
            except Exception as e:
                print(f"\n[Error] Generation failed for batch starting at index {i}: {e}")
                import traceback

                traceback.print_exc()
                generated_texts = [""] * len(batch_prompts)

            all_draft_generated_texts.extend(generated_texts)

        # 保存每个样本的完整初稿生成结果
        for idx in range(num_samples):
            if idx < len(all_draft_generated_texts):
                results[idx]["draft_generated_text"] = all_draft_generated_texts[idx]
            else:
                results[idx]["draft_generated_text"] = ""

        # ========== 阶段2：对所有初稿进行格式检查，提取<translate>标签中的内容 ==========
        print("\n" + "="*60)
        print("阶段2: 对所有初稿进行格式检查，提取<translate>标签中的内容")
        print("="*60)

        for idx in tqdm(range(num_samples), desc="检查初稿格式"):
            draft_text = results[idx].get("draft_generated_text", "")
            format_valid, draft_translation, format_score = check_and_extract_translate_tag(
                draft_text
            )
            results[idx]["draft_format_score"] = format_score
            results[idx]["draft_translation"] = draft_translation if format_valid else None

        # ========== 阶段3：把完整的初稿翻译切分为初稿短句 ==========
        print("\n" + "="*60)
        print("阶段3: 把完整的初稿翻译切分为初稿短句")
        print("="*60)

        for idx in tqdm(range(num_samples), desc="切分初稿翻译"):
            draft_translation = results[idx].get("draft_translation")
            if draft_translation:
                draft_segments = split_into_segments(draft_translation)
                if not draft_segments and draft_translation:
                    draft_segments = [draft_translation.strip()]
            else:
                draft_segments = []

            results[idx]["draft_segments"] = draft_segments
            results[idx]["draft_segment_results"] = [
                {"score": None, "error_spans": []} for _ in draft_segments
            ]

        # ========== 阶段4：对所有初稿短句进行XCOMET评分 ==========
        print("\n" + "="*60)
        print("阶段4: 对所有初稿短句进行XCOMET评分")
        print("="*60)

        if xcomet_loader:
            # 构建三元组：完整原文、初稿短句、完整参考翻译
            draft_segment_triplets = []
            draft_segment_mapping = []
            for idx in range(num_samples):
                src_text = results[idx]["src_text"]  # 完整原文
                ref_text = results[idx]["tgt_text"]  # 完整参考翻译
                draft_segments = results[idx].get("draft_segments", [])

                for seg_idx, draft_seg in enumerate(draft_segments):
                    if draft_seg:
                        draft_segment_triplets.append(
                            {
                                "src": str(src_text).strip(),
                                "mt": str(draft_seg).strip(),
                                "ref": str(ref_text).strip(),
                            }
                        )
                        draft_segment_mapping.append((idx, seg_idx))

            if draft_segment_triplets:
                try:
                    print(
                        f"[XCOMET] 批量评分 {len(draft_segment_triplets)} 个初稿短句（使用完整原文和完整参考翻译）..."
                    )
                    segment_results = xcomet_loader.predict(
                        draft_segment_triplets,
                        batch_size=args.xcomet_batch_size,
                        return_system_score=True,
                    )
                    for result_idx, (sample_idx, seg_idx) in enumerate(draft_segment_mapping):
                        if result_idx < len(segment_results):
                            results[sample_idx]["draft_segment_results"][seg_idx] = segment_results[
                                result_idx
                            ]
                        else:
                            results[sample_idx]["draft_segment_results"][seg_idx] = {
                                "score": None,
                                "error_spans": [],
                                "error": "XCOMET result index out of range",
                            }
                except Exception as e:
                    error_count = getattr(xcomet_loader, "_error_count", 0)
                    xcomet_loader._error_count = error_count + 1
                    if error_count < 3:
                        print(f"[Warning] 批量XCOMET评分失败: {str(e)[:100]}")
                    elif error_count == 3:
                        print("[Warning] XCOMET错误过多，后续错误将静默处理...")
                    for sample_idx, seg_idx in draft_segment_mapping:
                        if seg_idx < len(results[sample_idx].get("draft_segment_results", [])):
                            if not results[sample_idx]["draft_segment_results"][seg_idx]:
                                results[sample_idx]["draft_segment_results"][seg_idx] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": str(e)[:200]
                                    if error_count < 3
                                    else "Multiple errors (suppressed)",
                                }
            else:
                print("[XCOMET] 没有需要评分的初稿短句")

            # 汇总所有短句的评分，用于计算完整初稿的平均错误片段数量和初稿短句的平均XCOMET得分
            for idx in range(num_samples):
                segment_results = results[idx].get("draft_segment_results", [])
                if not segment_results:
                    continue
                scores = [
                    seg_res.get("score")
                    for seg_res in segment_results
                    if isinstance(seg_res, dict) and seg_res.get("score") is not None
                ]
                combined_spans = []
                for seg_res in segment_results:
                    if isinstance(seg_res, dict) and seg_res.get("error_spans"):
                        combined_spans.extend(seg_res["error_spans"])
                results[idx]["xcomet_draft"] = {
                    "score": (sum(scores) / len(scores)) if scores else None,
                    "error_spans": combined_spans,
                }
        else:
            print("[XCOMET] 未加载，跳过初稿短句评分")

        # ========== 阶段5：批量生成所有初稿短句的润色短句 ==========
        print("\n" + "="*60)
        print("阶段5: 批量生成所有初稿短句的润色短句")
        print("="*60)

        repair_prompts: List[str] = []
        repair_mapping: List[tuple] = []  # (sample_idx, segment_idx)

        for idx in range(num_samples):
            src_text = results[idx]["src_text"]  # 完整原文
            draft_segments = results[idx].get("draft_segments", [])
            segment_results = results[idx].get("draft_segment_results", [])

            results[idx]["repair_segment_outputs"] = [None] * len(draft_segments)
            results[idx]["final_segments"] = [None] * len(draft_segments)
            results[idx]["repair_segment_prompts"] = [None] * len(draft_segments)
            results[idx]["repair_segment_format_scores"] = [0] * len(draft_segments)

            for seg_idx, draft_seg in enumerate(draft_segments):
                if not draft_seg:
                    continue

                segment_errors = []
                if seg_idx < len(segment_results) and isinstance(segment_results[seg_idx], dict):
                    segment_errors = segment_results[seg_idx].get("error_spans", []) or []

                if segment_errors == []:
                    # 没有错误片段，跳过润色
                    continue

                # 生成repair prompt：包含完整原文、初稿短句、错误片段
                repair_example = {
                    "lg": results[idx]["lang_pair"],
                    "src_text": src_text,
                }
                repair_prompt = make_prefix(
                    repair_example,
                    template_type="repair",
                    tokenizer=tokenizer,
                    st_mode="extended",
                    error_spans=segment_errors,
                    draft_translation=draft_seg,
                )
                repair_prompts.append(repair_prompt)
                repair_mapping.append((idx, seg_idx))
                results[idx]["repair_segment_prompts"][seg_idx] = repair_prompt

        if repair_prompts:
            all_repair_generated_texts: List[str] = []
            for i in tqdm(range(0, len(repair_prompts), args.batch_size), desc="生成润色短句"):
                batch_prompts = repair_prompts[i : i + args.batch_size]
                try:
                    repair_texts = qwen_generator.generate_draft(
                        batch_prompts,
                        mode="repair",
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    if isinstance(repair_texts, str):
                        repair_texts = [repair_texts]
                    if len(repair_texts) != len(batch_prompts):
                        print(
                            f"[Warning] Repair生成文本数量({len(repair_texts)})与批次大小({len(batch_prompts)})不匹配"
                        )
                        repair_texts = list(repair_texts) + [""] * (
                            len(batch_prompts) - len(repair_texts)
                        )
                except Exception as e:
                    print(
                        f"\n[Error] Repair generation failed for batch starting at index {i}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    repair_texts = [""] * len(batch_prompts)

                all_repair_generated_texts.extend(repair_texts)

            # 提取润色短句的translate
            for prompt_idx, (sample_idx, segment_idx) in enumerate(repair_mapping):
                if prompt_idx >= len(all_repair_generated_texts):
                    continue
                repair_text = all_repair_generated_texts[prompt_idx]
                results[sample_idx]["repair_segment_outputs"][segment_idx] = repair_text
                valid, final_text, format_score = check_and_extract_translate_tag(repair_text)
                if valid and final_text:
                    results[sample_idx]["final_segments"][segment_idx] = final_text
                    results[sample_idx]["repair_segment_format_scores"][segment_idx] = format_score
                else:
                    # 回退到初稿短句
                    draft_seg = (
                        results[sample_idx]["draft_segments"][segment_idx]
                        if segment_idx < len(results[sample_idx]["draft_segments"])
                        else None
                    )
                    results[sample_idx]["final_segments"][segment_idx] = draft_seg
                    results[sample_idx]["repair_segment_format_scores"][segment_idx] = 0

        # ========== 阶段6：汇总终稿短句 ==========
        print("\n" + "="*60)
        print("阶段6: 汇总终稿短句")
        print("="*60)

        for idx in range(num_samples):
            draft_segments = results[idx].get("draft_segments", [])
            final_segments = results[idx].get("final_segments") or []

            # 确保所有短句都有最终结果
            for seg_idx in range(len(draft_segments)):
                if not final_segments[seg_idx]:
                    draft_seg = (
                        draft_segments[seg_idx] if seg_idx < len(draft_segments) else None
                    )
                    if draft_seg:
                        final_segments[seg_idx] = draft_seg

            has_all_drafts = all(
                draft_segments[seg_idx] for seg_idx in range(len(draft_segments))
            )

            if has_all_drafts and final_segments:
                combined_translation = " ".join(
                    seg.strip() for seg in final_segments if seg and seg.strip()
                )
                results[idx]["final_translation"] = (
                    combined_translation if combined_translation else None
                )
            else:
                results[idx]["final_translation"] = None

            segment_format_scores = results[idx].get("repair_segment_format_scores", [])
            if segment_format_scores:
                results[idx]["repair_format_score"] = (
                    1 if all(score == 1 for score in segment_format_scores) else 0
                )
            else:
                results[idx]["repair_format_score"] = 0

            if results[idx].get("repair_segment_outputs"):
                results[idx]["repair_generated_text"] = " ".join(
                    filter(None, results[idx]["repair_segment_outputs"])
                )
            else:
                results[idx]["repair_generated_text"] = None

        # ========== 阶段7：对所有终稿翻译进行XCOMET评分 ==========
        print("\n" + "="*60)
        print("阶段7: 对所有终稿翻译进行XCOMET评分")
        print("="*60)

        if xcomet_loader:
            final_xcomet_triplets = []
            final_xcomet_indices = []
            for idx in range(num_samples):
                final_translation = results[idx].get("final_translation")
                if final_translation:
                    src_text = (
                        str(results[idx]["src_text"]).strip()
                        if results[idx]["src_text"]
                        else ""
                    )
                    ref_text = (
                        str(results[idx]["tgt_text"]).strip()
                        if results[idx]["tgt_text"]
                        else ""
                    )
                    if src_text and ref_text:
                        final_xcomet_triplets.append(
                            {"src": src_text, "mt": final_translation, "ref": ref_text}
                        )
                        final_xcomet_indices.append(idx)

            if final_xcomet_triplets:
                try:
                    print(
                        f"[XCOMET] 批量评分 {len(final_xcomet_triplets)} 个终稿翻译..."
                    )
                    xcomet_final_results = xcomet_loader.predict(
                        final_xcomet_triplets,
                        batch_size=args.xcomet_batch_size,
                        return_system_score=True,
                    )
                    for result_idx, final_idx in enumerate(final_xcomet_indices):
                        if result_idx < len(xcomet_final_results):
                            xcomet_analysis_final = xcomet_final_results[result_idx]
                            results[final_idx]["xcomet_final"] = xcomet_analysis_final
                        else:
                            results[final_idx]["xcomet_final"] = {
                                "score": None,
                                "error_spans": [],
                                "error": "XCOMET final result index out of range",
                            }
                except Exception as e:
                    error_count = getattr(xcomet_loader, "_error_count", 0)
                    xcomet_loader._error_count = error_count + 1
                    if error_count < 3:
                        print(f"[Warning] 批量XCOMET终稿评分失败: {str(e)[:100]}")
                    elif error_count == 3:
                        print("[Warning] XCOMET错误过多，后续错误将静默处理...")
                    for final_idx in final_xcomet_indices:
                        if "xcomet_final" not in results[final_idx]:
                            results[final_idx]["xcomet_final"] = {
                                "score": None,
                                "error_spans": [],
                                "error": str(e)[:200]
                                if error_count < 3
                                else "Multiple errors (suppressed)",
                            }
            else:
                print("[XCOMET] 没有需要评分的终稿翻译")
        else:
            print("[XCOMET] 未加载，跳过终稿评分")
    
    else:
        # ========== 基线模式流程 ==========

        # ========== 阶段1：批量生成所有数据的初稿 ==========
        print("\n" + "="*60)
        print("阶段1: 批量生成所有数据的初稿")
        print("="*60)

        all_draft_generated_texts = []
        for i in tqdm(range(0, num_samples, args.batch_size), desc="生成初稿"):
            batch = dataset_df.iloc[i:i + args.batch_size]
            prompts = batch["prompt"].tolist()

            try:
                generated_texts = qwen_generator.generate_draft(
                    prompts,
                    mode="draft",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                if isinstance(generated_texts, str):
                    generated_texts = [generated_texts]
                if len(generated_texts) != len(batch):
                    print(
                        f"[Warning] 生成文本数量({len(generated_texts)})与批次大小({len(batch)})不匹配，使用空字符串填充"
                    )
                    generated_texts = list(generated_texts) + [""] * (
                        len(batch) - len(generated_texts)
                    )
            except Exception as e:
                print(f"\n[Error] Generation failed for batch starting at index {i}: {e}")
                import traceback

                traceback.print_exc()
                generated_texts = [""] * len(batch)

            all_draft_generated_texts.extend(generated_texts)

        # 保存初稿生成结果
        for idx, draft_text in enumerate(all_draft_generated_texts):
            results[idx]["draft_generated_text"] = draft_text

        # ========== 阶段2：对所有初稿进行格式检查 ==========
        print("\n" + "="*60)
        print("阶段2: 对所有初稿进行格式检查")
        print("="*60)

        for idx in tqdm(range(num_samples), desc="检查初稿格式"):
            draft_text = results[idx]["draft_generated_text"]
            format_valid, draft_translation, format_score = check_and_extract_translate_tag(
                draft_text
            )
            results[idx]["draft_format_score"] = format_score
            results[idx]["draft_translation"] = draft_translation if format_valid else None

        # ========== 阶段3：对所有初稿翻译进行XCOMET评分 ==========
        print("\n" + "="*60)
        print("阶段3: 对所有初稿翻译进行XCOMET评分")
        print("="*60)

        if xcomet_loader:
            draft_xcomet_triplets = []
            draft_xcomet_indices = []
            for idx in range(num_samples):
                format_valid = results[idx].get("draft_format_score", 0) == 1
                draft_translation = results[idx].get("draft_translation")
                if format_valid and draft_translation:
                    src_text = (
                        str(results[idx]["src_text"]).strip()
                        if results[idx]["src_text"]
                        else ""
                    )
                    ref_text = (
                        str(results[idx]["tgt_text"]).strip()
                        if results[idx]["tgt_text"]
                        else ""
                    )
                    if src_text and ref_text:
                        draft_xcomet_triplets.append(
                            {"src": src_text, "mt": draft_translation, "ref": ref_text}
                        )
                        draft_xcomet_indices.append(idx)

            if draft_xcomet_triplets:
                try:
                    print(
                        f"[XCOMET] 批量评分 {len(draft_xcomet_triplets)} 个初稿翻译..."
                    )
                    xcomet_results = xcomet_loader.predict(
                        draft_xcomet_triplets,
                        batch_size=args.xcomet_batch_size,
                        return_system_score=True,
                    )
                    for result_idx, xcomet_idx in enumerate(draft_xcomet_indices):
                        if result_idx < len(xcomet_results):
                            xcomet_analysis = xcomet_results[result_idx]
                            results[xcomet_idx]["xcomet_draft"] = xcomet_analysis
                        else:
                            results[xcomet_idx]["xcomet_draft"] = {
                                "score": None,
                                "error_spans": [],
                                "error": "XCOMET result index out of range",
                            }
                except Exception as e:
                    error_count = getattr(xcomet_loader, "_error_count", 0)
                    xcomet_loader._error_count = error_count + 1
                    if error_count < 3:
                        print(f"[Warning] 批量XCOMET评分失败: {str(e)[:100]}")
                    elif error_count == 3:
                        print("[Warning] XCOMET错误过多，后续错误将静默处理...")
                    for xcomet_idx in draft_xcomet_indices:
                        if "xcomet_draft" not in results[xcomet_idx]:
                            results[xcomet_idx]["xcomet_draft"] = {
                                "score": None,
                                "error_spans": [],
                                "error": str(e)[:200]
                                if error_count < 3
                                else "Multiple errors (suppressed)",
                            }
            else:
                print("[XCOMET] 没有需要评分的初稿翻译")

            # 对于没有 xcomet_draft 的样本做兜底说明
            for idx in range(num_samples):
                if "xcomet_draft" not in results[idx]:
                    format_valid = results[idx].get("draft_format_score", 0) == 1
                    draft_translation = results[idx].get("draft_translation")
                    if not format_valid or not draft_translation:
                        results[idx]["xcomet_draft"] = {
                            "score": None,
                            "error_spans": [],
                            "error": "Draft format invalid or translation is empty"
                            if not format_valid
                            else "Draft translation is empty",
                        }
                    else:
                        results[idx]["xcomet_draft"] = {
                            "score": None,
                            "error_spans": [],
                            "error": "XCOMET loader not available",
                        }
        else:
            print("[XCOMET] 未加载，跳过初稿评分")
            # 统一给一个说明
            for idx in range(num_samples):
                if "xcomet_draft" not in results[idx]:
                    results[idx]["xcomet_draft"] = {
                        "score": None,
                        "error_spans": [],
                        "error": "XCOMET loader not available",
                    }

        # ========== 阶段4：批量生成所有数据的终稿（repair） ==========
        print("\n" + "="*60)
        print("阶段4: 批量生成所有数据的终稿（repair）")
        print("="*60)

        # 收集需要repair的样本
        repair_prompts = []
        repair_indices = []

        for idx in range(num_samples):
            format_valid = results[idx].get("draft_format_score", 0) == 1
            draft_translation = results[idx].get("draft_translation")
            xcomet_draft = results[idx].get("xcomet_draft", {})
            error_spans = (
                xcomet_draft.get("error_spans", [])
                if isinstance(xcomet_draft, dict)
                else []
            )

            results[idx]["repair_generated_text"] = None
            results[idx]["repair_prompt"] = None
            results[idx]["repair_format_score"] = 0
            results[idx]["final_translation"] = None

            # 情况1：如果没有初稿，跳过refinement
            if not draft_translation:
                continue

            # 情况2：有初稿但无错误，跳过refinement，直接使用初稿
            elif format_valid and draft_translation and (not error_spans or len(error_spans) == 0):
                results[idx]["final_translation"] = draft_translation

            # 情况3：有初稿且有错误，需要repair
            elif format_valid and draft_translation and error_spans:
                try:
                    repair_example = {
                        "lg": results[idx]["lang_pair"],
                        "src_text": results[idx]["src_text"],
                    }
                    repair_prompt = make_prefix(
                        repair_example,
                        template_type="repair",
                        st_mode="baseline",
                        tokenizer=tokenizer,
                        error_spans=error_spans,
                        draft_translation=draft_translation,
                    )
                    repair_prompts.append(repair_prompt)
                    repair_indices.append(idx)
                    results[idx]["repair_prompt"] = repair_prompt
                except Exception as e:
                    print(
                        f"[Warning] 构建repair prompt失败 for sample {results[idx]['index']}: {str(e)[:100]}"
                    )

        # 批量生成repair翻译
        if repair_prompts:
            all_repair_generated_texts = []
            for i in tqdm(
                range(0, len(repair_prompts), args.batch_size), desc="生成repair翻译"
            ):
                batch_prompts = repair_prompts[i : i + args.batch_size]
                try:
                    repair_texts = qwen_generator.generate_draft(
                        batch_prompts,
                        mode="repair",
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    if isinstance(repair_texts, str):
                        repair_texts = [repair_texts]
                    if len(repair_texts) != len(batch_prompts):
                        print(
                            f"[Warning] Repair生成文本数量({len(repair_texts)})与批次大小({len(batch_prompts)})不匹配"
                        )
                        repair_texts = list(repair_texts) + [""] * (
                            len(batch_prompts) - len(repair_texts)
                        )
                except Exception as e:
                    print(
                        f"\n[Error] Repair generation failed for batch starting at index {i}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    repair_texts = [""] * len(batch_prompts)

                all_repair_generated_texts.extend(repair_texts)

            # 保存repair结果
            for prompt_idx, result_idx in enumerate(repair_indices):
                repair_text = all_repair_generated_texts[prompt_idx]
                results[result_idx]["repair_generated_text"] = repair_text

        # ========== 阶段5：对所有终稿进行格式检查 ==========
        print("\n" + "="*60)
        print("阶段5: 对所有终稿进行格式检查")
        print("="*60)

        for idx in tqdm(range(num_samples), desc="检查终稿格式"):
            # 有初稿、无错误，直接使用初稿，不需要额外格式检查
            if (
                results[idx].get("final_translation") is not None
                and results[idx].get("repair_generated_text") is None
            ):
                results[idx]["repair_format_score"] = 0
                continue

            repair_generated_text = results[idx].get("repair_generated_text")
            if repair_generated_text:
                (
                    repair_format_valid,
                    final_translation,
                    repair_format_score,
                ) = check_and_extract_translate_tag(repair_generated_text)
                results[idx]["repair_format_score"] = repair_format_score
                results[idx]["final_translation"] = (
                    final_translation if repair_format_valid else None
                )

        # ========== 阶段6：对所有终稿翻译进行XCOMET评分 ==========
        print("\n" + "="*60)
        print("阶段6: 对所有终稿翻译进行XCOMET评分")
        print("="*60)

        if xcomet_loader:
            final_xcomet_triplets = []
            final_xcomet_indices = []
            for idx in range(num_samples):
                final_translation = results[idx].get("final_translation")
                if final_translation:
                    src_text = (
                        str(results[idx]["src_text"]).strip()
                        if results[idx]["src_text"]
                        else ""
                    )
                    ref_text = (
                        str(results[idx]["tgt_text"]).strip()
                        if results[idx]["tgt_text"]
                        else ""
                    )
                    if src_text and ref_text:
                        final_xcomet_triplets.append(
                            {"src": src_text, "mt": final_translation, "ref": ref_text}
                        )
                        final_xcomet_indices.append(idx)

            if final_xcomet_triplets:
                try:
                    print(
                        f"[XCOMET] 批量评分 {len(final_xcomet_triplets)} 个终稿翻译..."
                    )
                    xcomet_final_results = xcomet_loader.predict(
                        final_xcomet_triplets,
                        batch_size=args.xcomet_batch_size,
                        return_system_score=True,
                    )
                    for result_idx, final_idx in enumerate(final_xcomet_indices):
                        if result_idx < len(xcomet_final_results):
                            xcomet_analysis_final = xcomet_final_results[result_idx]
                            results[final_idx]["xcomet_final"] = xcomet_analysis_final
                        else:
                            results[final_idx]["xcomet_final"] = {
                                "score": None,
                                "error_spans": [],
                                "error": "XCOMET final result index out of range",
                            }
                except Exception as e:
                    error_count = getattr(xcomet_loader, "_error_count", 0)
                    xcomet_loader._error_count = error_count + 1
                    if error_count < 3:
                        print(f"[Warning] 批量XCOMET终稿评分失败: {str(e)[:100]}")
                    elif error_count == 3:
                        print("[Warning] XCOMET错误过多，后续错误将静默处理...")
                    for final_idx in final_xcomet_indices:
                        if "xcomet_final" not in results[final_idx]:
                            results[final_idx]["xcomet_final"] = {
                                "score": None,
                                "error_spans": [],
                                "error": str(e)[:200]
                                if error_count < 3
                                else "Multiple errors (suppressed)",
                            }
            else:
                print("[XCOMET] 没有需要评分的终稿翻译")
        else:
            print("[XCOMET] 未加载，跳过终稿评分")
            for idx in range(num_samples):
                if "final_translation" in results[idx] and results[idx]["final_translation"]:
                    if "xcomet_final" not in results[idx]:
                        results[idx]["xcomet_final"] = {
                            "score": None,
                            "error_spans": [],
                            "error": "XCOMET loader not available",
                        }

    # ========== 5. 保存结果 ==========
    if args.output_file:
        print("\n" + "="*60)
        print("步骤5: 保存结果")
        print("="*60)
        output_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.output_file
        )
        
        # 保存为JSON
        output_txt = "xcomet_all_stats.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_stats(f"[Output] Results saved to {output_path}", output_file=output_txt)
        
        # 打印统计信息
        draft_format_scores = [r.get('draft_format_score', 0) for r in results]
        repair_format_scores = [r.get('repair_format_score', 0) for r in results]
        draft_valid_count = sum(draft_format_scores)
        repair_valid_count = sum(repair_format_scores)

        
        num_samples = len(results)
        log_stats(
            f"[Stats] Draft格式正确率: {draft_valid_count}/{num_samples} ({100*draft_valid_count/num_samples:.1f}%)",
            output_file=output_txt,
        )
        log_stats(
            f"[Stats] Repair格式正确率: {repair_valid_count}/{num_samples} ({100*repair_valid_count/num_samples:.1f}%)",
            output_file=output_txt,
        )

        if xcomet_loader:
            # XCOMET统计（初稿）
            draft_scores = [
                r.get('xcomet_draft', {}).get('score')
                for r in results
                if r.get('xcomet_draft') and r['xcomet_draft'].get('score') is not None
            ]
            if draft_scores:
                log_stats(
                    "[Stats] XCOMET Draft scores - "
                    f"Mean: {sum(draft_scores)/len(draft_scores):.4f}, "
                    f"Min: {min(draft_scores):.4f}, Max: {max(draft_scores):.4f}",
                    output_file=output_txt,
                )

            draft_error_span_counts = [
                len(r.get('xcomet_draft', {}).get('error_spans', []))
                for r in results
                if r.get('xcomet_draft')
            ]
            if draft_error_span_counts:
                avg_draft_spans = sum(draft_error_span_counts) / len(draft_error_span_counts)
                log_stats(
                    f"[Stats] Avg. error spans per draft sample: {avg_draft_spans:.2f}",
                    output_file=output_txt,
                )

            # 终稿错误spans统计
            final_error_span_counts = [
                len(r.get('xcomet_final', {}).get('error_spans', []))
                for r in results
                if r.get('xcomet_final')
            ]
            if final_error_span_counts:
                avg_final_spans = sum(final_error_span_counts) / len(final_error_span_counts)
                log_stats(
                    f"[Stats] Avg. error spans per final sample: {avg_final_spans:.2f}",
                    output_file=output_txt,
                )

            # XCOMET统计（终稿）
            final_scores = [
                r.get('xcomet_final', {}).get('score')
                for r in results
                if r.get('xcomet_final') and r['xcomet_final'].get('score') is not None
            ]
            if final_scores:
                log_stats(
                    "[Stats] XCOMET Final scores - "
                    f"Mean: {sum(final_scores)/len(final_scores):.4f}, "
                    f"Min: {min(final_scores):.4f}, Max: {max(final_scores):.4f}",
                    output_file=output_txt,
                )

            # 改进统计（注意：这里最好用 paired，而不是 zip 两个 list，
            # 不过你目前 draft/final 是一一对应的也可以先这样用）
            if draft_scores and final_scores:
                improved_count = sum(
                    1 for d, f in zip(draft_scores, final_scores) if f and d and f > d
                )
                log_stats(
                    f"[Stats] 终稿改进初稿的样本数: {improved_count}/{len(final_scores)} "
                    f"({100*improved_count/len(final_scores):.1f}%)",
                    output_file=output_txt,
                )
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

