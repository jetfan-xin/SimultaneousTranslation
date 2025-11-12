#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimultaneousTranslation主执行脚本
实现：
1. 获取数据集（MT_Grpo的data/train）
2. 转换数据集为parquet格式，并根据draft mode模板生成完整的prompt文本（包含instruction、格式说明和用户输入）
3. 在外部加载XCOMET模型，方便后续评分环节直接调用
4. 调用Qwen2.5-7B，使用步骤2生成的prompt，Mode=draft，生成并获取回答
5. 检查格式是否正确，如果格式正确，则抽取其中的<translate>部分，作为初稿，格式分数记为1；如果格式不正确，则不做更多操作，直接将格式分数记为0
6. 对格式正确的翻译调用XCOMET，输出有错误的部分
7. 再次调用Qwen2.5-7B，Mode=repair，prompt中包含：mode、原文、初稿和错误的span，让模型再次思考，获取回答
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

from data.process_data import read_jsonl_files, preprocess_data, make_prefix
from transformers import AutoTokenizer
from xcomet_loader import XCOMETLoader
from qwen_generator import QwenGenerator
from utils import check_and_extract_translate_tag, format_error_spans_for_prompt, split_into_segments


def load_dataset_from_train_files(train_files: List[str], base_dir: str = None):
    """
    从训练文件中加载数据集
    
    Args:
        train_files: 训练文件路径列表
        base_dir: 基础目录，如果文件路径是相对路径则基于此目录
    
    Returns:
        处理后的数据列表
    """
    if base_dir:
        train_files = [os.path.join(base_dir, f) if not os.path.isabs(f) else f for f in train_files]
    
    print(f"[Dataset] Loading training files: {train_files}")
    data = read_jsonl_files(train_files)
    processed_data = preprocess_data(data)
    print(f"[Dataset] Loaded {len(processed_data)} samples")
    return processed_data


def process_data_for_qwen(data: List[Dict], tokenizer_path: str, tokenizer=None, output_file: Optional[str] = None):
    """
    将数据转换为Qwen2.5-7B可以处理的格式，并生成完整的prompt文本
    
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
            "lang_pair": example.get('lg', 'en-zh'),
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
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--train_files', nargs='+', 
                       default=['train/json/train_enzh_6565.jsonl', 'train/json/train_zhen_6565.jsonl'],
                       help='训练JSONL文件路径（相对于data_dir）')
    parser.add_argument('--tokenizer_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
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
    parser.add_argument('--qwen_model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Qwen2.5-7B模型路径或HuggingFace模型ID')
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
    parser.add_argument('--input_results_file', type=str, default=None,
                       help='输入results.json文件路径（如果提供，将从该文件读取已有的draft结果进行refinement）')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件路径（JSON格式，保存生成结果）')
    parser.add_argument('--processed_data_file', type=str, default=None,
                       help='（已弃用）处理后的数据文件路径。现在会自动为每个数据集文件生成对应的parquet文件')
    
    args = parser.parse_args()
    extended_mode = args.pipeline_mode == 'extended'
    
    # ========== 0. GPU环境检测和分配 ==========
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
        
        # 显示GPU分配计划
        original_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        if args.xcomet_cpu:
            print(f"\n[XCOMET] 将使用: CPU（--xcomet_cpu）")
        elif args.xcomet_gpus:
            print(f"\n[XCOMET] 将使用GPU: {args.xcomet_gpus}")
        elif original_cuda_env and not args.qwen_gpus:
            print(f"\n[XCOMET] 将使用GPU: {original_cuda_env}（环境变量CUDA_VISIBLE_DEVICES）")
        else:
            print(f"\n[XCOMET] 将使用: CPU（默认）")
        
        if args.qwen_cpu:
            print(f"[Qwen] 将使用: CPU（--qwen_cpu）")
        elif args.qwen_gpus:
            print(f"[Qwen] 将使用GPU: {args.qwen_gpus}")
        elif original_cuda_env and not args.xcomet_gpus:
            print(f"[Qwen] 将使用GPU: {original_cuda_env}（环境变量CUDA_VISIBLE_DEVICES）")
        else:
            print(f"[Qwen] 将使用: CPU（默认）")
    else:
        print("✗ 未检测到GPU，将使用CPU模式（非常慢）")
    
    # ========== 检查是否从results.json读取数据 ==========
    if args.input_results_file:
        # 从results.json读取已有数据，只进行refinement
        print("\n" + "="*60)
        print("从results.json读取数据，进行refinement处理")
        print("="*60)
        input_path = args.input_results_file if os.path.isabs(args.input_results_file) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.input_results_file
        )
        print(f"[Input] Loading results from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        print(f"[Input] Loaded {len(existing_results)} results from results.json")
        
        # 加载tokenizer（后续repair mode需要使用）
        print(f"[Process] Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        
        # 将results.json转换为DataFrame格式以便后续处理
        dataset_df = pd.DataFrame(existing_results)
        
        # 跳过步骤1-4，直接进入步骤5-8（格式检查和refinement）
        skip_draft_generation = True
    else:
        # 正常流程：从数据集文件读取
        # ========== 1. 获取数据集 ==========
        print("\n" + "="*60)
        print("步骤1: 获取数据集")
        print("="*60)
        data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.data_dir
        )
        train_files = [os.path.join(data_dir, f) for f in args.train_files]
        
        # ========== 2. 转换数据集为Qwen格式并生成prompt ==========
        print("\n" + "="*60)
        print("步骤2: 转换数据集为parquet格式并生成prompt（draft mode，复用base模板）")
        print("="*60)
        
        # 加载tokenizer（用于生成prompt和后续repair mode）
        print(f"[Process] Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        
        # 为每个训练文件分别处理并保存parquet
        all_datasets = []
        parquet_files = []
        
        # 创建parquet保存目录：data/train/parquet
        parquet_dir = Path(args.data_dir) / "train" / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Process] Parquet文件将保存到: {parquet_dir}")
        
        for train_file in train_files:
            # 生成对应的parquet文件路径（保存到data/train/parquet目录，文件名不变，扩展名改为.parquet）
            train_file_path = Path(train_file)
            parquet_filename = train_file_path.stem + '.parquet'  # 保持原文件名，只改扩展名
            parquet_file_path = parquet_dir / parquet_filename
            parquet_files.append(str(parquet_file_path))
            
            # 检查parquet文件是否存在
            if parquet_file_path.exists():
                print(f"[Process] 发现已处理的parquet文件: {parquet_file_path}")
                print(f"[Process] 直接加载，跳过数据转换...")
                dataset = Dataset.from_parquet(str(parquet_file_path))
                print(f"[Process] 成功加载 {len(dataset)} 个样本")
            else:
                # 如果parquet文件不存在，读取原始jsonl文件并转换
                print(f"[Process] 未找到parquet文件，将转换并保存: {parquet_file_path}")
                
                # 读取该文件的原始数据
                file_data = []
                with open(train_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            file_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"[Line {line_num}] JSON parse failed → Line content: {repr(line)}")
                
                # 预处理数据
                processed_file_data = preprocess_data(file_data)
                
                # 转换为Qwen格式并保存
                dataset = process_data_for_qwen(
                    processed_file_data,
                    tokenizer_path=args.tokenizer_path,
                    tokenizer=tokenizer,
                    output_file=str(parquet_file_path)
                )
                print(f"[Process] 成功转换并保存 {len(dataset)} 个样本到 {parquet_file_path}")
            
            # 为每个数据集添加文件来源信息，并重新编号索引
            dataset = dataset.map(lambda example, idx: {**example, "index": idx, "source_file": train_file_path.name}, with_indices=True)
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
        
        skip_draft_generation = False
    
    # ========== 3. 加载XCOMET模型 ==========
    print("\n" + "="*60)
    print("步骤3: 加载XCOMET模型")
    print("="*60)
    xcomet_loader = None
    if args.load_xcomet:
        # 确定XCOMET checkpoint路径
        if args.xcomet_ckpt:
            xcomet_ckpt = args.xcomet_ckpt
        elif os.getenv("WORD_QE_CKPT"):
            xcomet_ckpt = os.getenv("WORD_QE_CKPT")
        else:
            # 默认路径
            default_ckpt = os.path.expanduser("~/models/XCOMET-XL/checkpoints/model.ckpt")
            if os.path.exists(default_ckpt):
                xcomet_ckpt = default_ckpt
            else:
                print(f"[XCOMET] Warning: XCOMET checkpoint not found at default path: {default_ckpt}")
                print("[XCOMET] Skipping XCOMET loading. Set --xcomet_ckpt or WORD_QE_CKPT environment variable.")
                xcomet_ckpt = None
        
        if xcomet_ckpt:
            # 使用统一的XCOMET GPU选择逻辑
            xcomet_gpu_ids, _ = get_xcomet_gpu_setting(args)
            
            if args.xcomet_cpu:
                print("[XCOMET] 强制使用CPU模式（--xcomet_cpu）")
            elif args.xcomet_gpus:
                print(f"[XCOMET] 通过参数指定使用GPU: {xcomet_gpu_ids}")
            elif xcomet_gpu_ids:
                print(f"[XCOMET] 使用环境变量CUDA_VISIBLE_DEVICES={xcomet_gpu_ids}")
            else:
                print("[XCOMET] 默认使用CPU模式")
            
            try:
                xcomet_loader = XCOMETLoader(
                    xcomet_ckpt, 
                    force_cpu=args.xcomet_cpu or (xcomet_gpu_ids is None),
                    gpu_ids=xcomet_gpu_ids
                )
                print("[XCOMET] XCOMET model loaded successfully")
            except Exception as e:
                print(f"[XCOMET] Failed to load XCOMET: {e}")
                xcomet_loader = None
    else:
        print("[XCOMET] Skipping XCOMET loading (--load_xcomet=False)")
    
    # ========== 4. 调用Qwen2.5-7B生成draft翻译 ==========
    if not skip_draft_generation:
        print("\n" + "="*60)
        print("步骤4: 调用Qwen2.5-7B生成draft翻译")
        print("="*60)
        
        print(f"[Qwen] Loading Qwen model from {args.qwen_model_path}...")
        
        # 设置Qwen使用的GPU
        # GPU选择逻辑：
        # 1. 默认：CPU模式
        # 2. 如果设置了 --qwen_cpu，强制CPU模式
        # 3. 如果设置了 --qwen_gpus，使用指定的GPU
        # 4. 如果只设置了CUDA_VISIBLE_DEVICES（环境变量），且没有分别设置qwen_gpus和xcomet_gpus，则使用CUDA_VISIBLE_DEVICES
        # 5. 否则，使用CPU
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        import torch
        if args.qwen_cpu:
            # 强制CPU模式
            qwen_gpu_ids = None
            # 清除CUDA_VISIBLE_DEVICES以确保使用CPU
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print("[Qwen] 强制使用CPU模式（--qwen_cpu）")
        elif args.qwen_gpus:
            # 通过参数指定GPU
            qwen_gpu_ids = args.qwen_gpus
            os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            print(f"[Qwen] 通过参数指定使用GPU: {qwen_gpu_ids}")
        elif original_cuda_visible and not args.xcomet_gpus:
            # 如果只设置了CUDA_VISIBLE_DEVICES，且没有分别设置xcomet_gpus，则两者都使用CUDA_VISIBLE_DEVICES
            qwen_gpu_ids = original_cuda_visible
            os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            print(f"[Qwen] 使用环境变量CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")
        else:
            # 默认CPU模式
            qwen_gpu_ids = None
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print("[Qwen] 默认使用CPU模式")
        
        vllm_extra_kwargs = {}
        if args.use_vllm:
            if args.vllm_max_num_seqs:
                vllm_extra_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
            # 禁用自定义all-reduce避免多PCIe GPU报错
            vllm_extra_kwargs["disable_custom_all_reduce"] = True

        try:
            qwen_generator = QwenGenerator(
                model_path=args.qwen_model_path,
                use_vllm=args.use_vllm if not args.qwen_cpu else False,  # CPU模式下不使用vLLM
                device="cpu" if args.qwen_cpu else None,  # CPU模式下明确指定device
                gpu_memory_utilization=args.gpu_memory_utilization,
                **vllm_extra_kwargs
            )
            print("[Qwen] Qwen model loaded successfully")
        except Exception as e:
            print(f"[Qwen] Failed to load Qwen model: {e}")
            print("[Qwen] Trying to continue with transformers backend...")
            qwen_generator = QwenGenerator(
                model_path=args.qwen_model_path,
                use_vllm=False,
                device="cpu" if args.qwen_cpu else None,  # CPU模式下明确指定device
                gpu_memory_utilization=args.gpu_memory_utilization
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
            draft_prompts: List[str] = []
            for idx in range(num_samples):
                draft_example = {
                    'lg': results[idx]['lang_pair'],
                    'src_text': results[idx]['src_text'],  # 完整原文
                }
                draft_prompt = make_prefix(
                    draft_example,
                    template_type='draft',
                    tokenizer=tokenizer
                )
                draft_prompts.append(draft_prompt)
            
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
                        top_p=args.top_p
                    )
                    if isinstance(generated_texts, str):
                        generated_texts = [generated_texts]
                    if len(generated_texts) != len(batch_prompts):
                        print(f"[Warning] 生成文本数量({len(generated_texts)})与批次大小({len(batch_prompts)})不匹配")
                        generated_texts = list(generated_texts) + [""] * (len(batch_prompts) - len(generated_texts))
                except Exception as e:
                    print(f"\n[Error] Generation failed for batch starting at index {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    generated_texts = [""] * len(batch_prompts)
                
                all_draft_generated_texts.extend(generated_texts)
            
            # 保存每个样本的完整初稿生成结果
            for idx in range(num_samples):
                if idx < len(all_draft_generated_texts):
                    results[idx]['draft_generated_text'] = all_draft_generated_texts[idx]
                else:
                    results[idx]['draft_generated_text'] = ""
            
            # ========== 阶段2：对所有初稿进行格式检查，提取<translate>标签中的内容 ==========
            print("\n" + "="*60)
            print("阶段2: 对所有初稿进行格式检查，提取<translate>标签中的内容")
            print("="*60)
            
            for idx in tqdm(range(num_samples), desc="检查初稿格式"):
                draft_text = results[idx].get('draft_generated_text', '')
                format_valid, draft_translation, format_score = check_and_extract_translate_tag(draft_text)
                results[idx]['draft_format_score'] = format_score
                results[idx]['draft_translation'] = draft_translation if format_valid else None
            
            # ========== 阶段3：把完整的初稿翻译切分为初稿短句 ==========
            print("\n" + "="*60)
            print("阶段3: 把完整的初稿翻译切分为初稿短句")
            print("="*60)
            
            for idx in tqdm(range(num_samples), desc="切分初稿翻译"):
                draft_translation = results[idx].get('draft_translation')
                if draft_translation:
                    draft_segments = split_into_segments(draft_translation)
                    if not draft_segments and draft_translation:
                        draft_segments = [draft_translation.strip()]
                else:
                    draft_segments = []
                
                results[idx]['draft_segments'] = draft_segments
                results[idx]['draft_segment_results'] = [
                    {"score": None, "error_spans": []} for _ in draft_segments
                ]
            
            # ========== 阶段4：对所有初稿短句进行XCOMET评分 ==========
            print("\n" + "="*60)
            print("阶段4: 对所有初稿短句进行XCOMET评分")
            print("="*60)
            
            if xcomet_loader:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 强制使用CPU模式（--xcomet_cpu）")
                elif args.xcomet_gpus:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 通过参数指定使用GPU: {xcomet_gpu_ids}")
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 使用环境变量CUDA_VISIBLE_DEVICES={xcomet_gpu_ids}")
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 默认使用CPU模式")
                
                # 构建三元组：完整原文、初稿短句、完整参考翻译
                draft_segment_triplets = []
                draft_segment_mapping = []
                for idx in range(num_samples):
                    src_text = results[idx]['src_text']  # 完整原文
                    ref_text = results[idx]['tgt_text']  # 完整参考翻译
                    draft_segments = results[idx].get('draft_segments', [])
                    
                    for seg_idx, draft_seg in enumerate(draft_segments):
                        if draft_seg:
                            draft_segment_triplets.append({
                                "src": str(src_text).strip(),
                                "mt": str(draft_seg).strip(),
                                "ref": str(ref_text).strip()
                            })
                            draft_segment_mapping.append((idx, seg_idx))
                
                if draft_segment_triplets:
                    try:
                        print(f"[XCOMET] 批量评分 {len(draft_segment_triplets)} 个初稿短句（使用完整原文和完整参考翻译）...")
                        segment_results = xcomet_loader.predict(
                            draft_segment_triplets,
                            batch_size=args.xcomet_batch_size,
                            return_system_score=True
                        )
                        for result_idx, (sample_idx, seg_idx) in enumerate(draft_segment_mapping):
                            if result_idx < len(segment_results):
                                results[sample_idx]['draft_segment_results'][seg_idx] = segment_results[result_idx]
                            else:
                                results[sample_idx]['draft_segment_results'][seg_idx] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": "XCOMET result index out of range",
                                }
                    except Exception as e:
                        error_count = xcomet_loader._error_count
                        xcomet_loader._error_count = error_count + 1
                        if error_count < 3:
                            print(f"[Warning] 批量XCOMET评分失败: {str(e)[:100]}")
                        elif error_count == 3:
                            print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                        for sample_idx, seg_idx in draft_segment_mapping:
                            if seg_idx < len(results[sample_idx].get('draft_segment_results', [])):
                                if not results[sample_idx]['draft_segment_results'][seg_idx]:
                                    results[sample_idx]['draft_segment_results'][seg_idx] = {
                                        "score": None,
                                        "error_spans": [],
                                        "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                                    }
                else:
                    print("[XCOMET] 没有需要评分的初稿短句")
                
                # 汇总所有短句的评分
                for idx in range(num_samples):
                    segment_results = results[idx].get('draft_segment_results', [])
                    if not segment_results:
                        continue
                    scores = [seg_res.get('score') for seg_res in segment_results if isinstance(seg_res, dict) and seg_res.get('score') is not None]
                    combined_spans = []
                    for seg_res in segment_results:
                        if isinstance(seg_res, dict) and seg_res.get('error_spans'):
                            combined_spans.extend(seg_res['error_spans'])
                    results[idx]['xcomet_draft'] = {
                        "score": (sum(scores) / len(scores)) if scores else None,
                        "error_spans": combined_spans,
                    }
                
                if torch.cuda.is_available() and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                    print(f"[Stage] 恢复Qwen的CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")
            
            # ========== 阶段5：批量生成所有初稿短句的润色短句 ==========
            print("\n" + "="*60)
            print("阶段5: 批量生成所有初稿短句的润色短句")
            print("="*60)
            
            repair_prompts: List[str] = []
            repair_mapping: List[tuple] = []  # (sample_idx, segment_idx)
            
            for idx in range(num_samples):
                src_text = results[idx]['src_text']  # 完整原文
                ref_text = results[idx]['tgt_text']  # 完整参考翻译
                draft_segments = results[idx].get('draft_segments', [])
                segment_results = results[idx].get('draft_segment_results', [])
                
                results[idx]['repair_segment_outputs'] = [None] * len(draft_segments)
                results[idx]['final_segments'] = [None] * len(draft_segments)
                results[idx]['repair_segment_prompts'] = [None] * len(draft_segments)
                results[idx]['repair_segment_format_scores'] = [0] * len(draft_segments)
                
                for seg_idx, draft_seg in enumerate(draft_segments):
                    if not draft_seg:
                        # 如果没有初稿短句，跳过
                        continue
                    
                    segment_errors = []
                    if seg_idx < len(segment_results) and isinstance(segment_results[seg_idx], dict):
                        segment_errors = segment_results[seg_idx].get('error_spans', []) or []
                    
                    # 生成repair prompt：包含完整原文、初稿短句、完整参考翻译、错误片段
                    repair_example = {
                        'lg': results[idx]['lang_pair'],
                        'src_text': src_text,  # 完整原文
                    }
                    repair_prompt = make_prefix(
                        repair_example,
                        template_type='repair',
                        tokenizer=tokenizer,
                        error_spans=segment_errors,
                        draft_translation=draft_seg,  # 初稿短句
                        ref_text=ref_text  # 完整参考翻译
                    )
                    repair_prompts.append(repair_prompt)
                    repair_mapping.append((idx, seg_idx))
                    results[idx]['repair_segment_prompts'][seg_idx] = repair_prompt
            
            if repair_prompts:
                all_repair_generated_texts: List[str] = []
                for i in tqdm(range(0, len(repair_prompts), args.batch_size), desc="生成润色短句"):
                    batch_prompts = repair_prompts[i:i + args.batch_size]
                    try:
                        repair_texts = qwen_generator.generate_draft(
                            batch_prompts,
                            mode="repair",
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p
                        )
                        if isinstance(repair_texts, str):
                            repair_texts = [repair_texts]
                        if len(repair_texts) != len(batch_prompts):
                            print(f"[Warning] Repair生成文本数量({len(repair_texts)})与批次大小({len(batch_prompts)})不匹配")
                            repair_texts = list(repair_texts) + [""] * (len(batch_prompts) - len(repair_texts))
                    except Exception as e:
                        print(f"\n[Error] Repair generation failed for batch starting at index {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        repair_texts = [""] * len(batch_prompts)
                    
                    all_repair_generated_texts.extend(repair_texts)
                
                # 提取润色短句的translate
                for prompt_idx, (sample_idx, segment_idx) in enumerate(repair_mapping):
                    if prompt_idx >= len(all_repair_generated_texts):
                        continue
                    repair_text = all_repair_generated_texts[prompt_idx]
                    results[sample_idx]['repair_segment_outputs'][segment_idx] = repair_text
                    valid, final_text, format_score = check_and_extract_translate_tag(repair_text)
                    if valid and final_text:
                        results[sample_idx]['final_segments'][segment_idx] = final_text
                        results[sample_idx]['repair_segment_format_scores'][segment_idx] = format_score
                    else:
                        # 回退到初稿短句
                        draft_seg = results[sample_idx]['draft_segments'][segment_idx] if segment_idx < len(results[sample_idx]['draft_segments']) else None
                        results[sample_idx]['final_segments'][segment_idx] = draft_seg
                        results[sample_idx]['repair_segment_format_scores'][segment_idx] = 0
            
            # ========== 阶段6：汇总终稿短句 ==========
            print("\n" + "="*60)
            print("阶段6: 汇总终稿短句")
            print("="*60)
            
            for idx in range(num_samples):
                draft_segments = results[idx].get('draft_segments', [])
                final_segments = results[idx].get('final_segments') or []
                
                # 确保所有短句都有最终结果
                for seg_idx in range(len(draft_segments)):
                    if seg_idx >= len(final_segments) or not final_segments[seg_idx]:
                        # 如果没有润色短句，使用初稿短句
                        draft_seg = draft_segments[seg_idx] if seg_idx < len(draft_segments) else None
                        if draft_seg:
                            if seg_idx >= len(final_segments):
                                final_segments.extend([None] * (seg_idx - len(final_segments) + 1))
                            final_segments[seg_idx] = draft_seg
                
                # 检查是否有缺失的初稿短句
                has_all_drafts = all(
                    seg_idx < len(draft_segments) and draft_segments[seg_idx] 
                    for seg_idx in range(len(draft_segments))
                )
                
                if has_all_drafts and final_segments:
                    # 合并所有短句为终稿
                    combined_translation = ' '.join(
                        seg.strip() for seg in final_segments if seg and seg.strip()
                    )
                    results[idx]['final_translation'] = combined_translation if combined_translation else None
                else:
                    # 如果有初稿短句不存在的情况，则没有终稿
                    results[idx]['final_translation'] = None
                
                # 计算repair格式得分
                segment_format_scores = results[idx].get('repair_segment_format_scores', [])
                if segment_format_scores:
                    results[idx]['repair_format_score'] = 1 if all(score == 1 for score in segment_format_scores) else 0
                else:
                    results[idx]['repair_format_score'] = 0
                
                if results[idx].get('repair_segment_outputs'):
                    results[idx]['repair_generated_text'] = ' '.join(filter(None, results[idx]['repair_segment_outputs']))
                else:
                    results[idx]['repair_generated_text'] = None
            
            # ========== 阶段7：对所有终稿翻译进行XCOMET评分 ==========
            print("\n" + "="*60)
            print("阶段7: 对所有终稿翻译进行XCOMET评分")
            print("="*60)
            
            if xcomet_loader:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 强制使用CPU模式（--xcomet_cpu）")
                elif args.xcomet_gpus and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 通过参数指定使用GPU: {xcomet_gpu_ids}")
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 使用环境变量CUDA_VISIBLE_DEVICES={xcomet_gpu_ids}")
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 默认使用CPU模式")
                
                final_xcomet_triplets = []
                final_xcomet_indices = []
                for idx in range(num_samples):
                    final_translation = results[idx].get('final_translation')
                    if final_translation:
                        src_text = str(results[idx]['src_text']).strip() if results[idx]['src_text'] else ""
                        ref_text = str(results[idx]['tgt_text']).strip() if results[idx]['tgt_text'] else ""
                        if src_text and ref_text:
                            final_xcomet_triplets.append({
                                "src": src_text,
                                "mt": final_translation,
                                "ref": ref_text
                            })
                            final_xcomet_indices.append(idx)
                if final_xcomet_triplets:
                    try:
                        print(f"[XCOMET] 批量评分 {len(final_xcomet_triplets)} 个终稿翻译...")
                        xcomet_final_results = xcomet_loader.predict(
                            final_xcomet_triplets,
                            batch_size=args.xcomet_batch_size,
                            return_system_score=True
                        )
                        for result_idx, final_idx in enumerate(final_xcomet_indices):
                            if result_idx < len(xcomet_final_results):
                                xcomet_analysis_final = xcomet_final_results[result_idx]
                                results[final_idx]['xcomet_final'] = xcomet_analysis_final
                            else:
                                results[final_idx]['xcomet_final'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": "XCOMET final result index out of range",
                                }
                    except Exception as e:
                        error_count = xcomet_loader._error_count
                        xcomet_loader._error_count = error_count + 1
                        if error_count < 3:
                            print(f"[Warning] 批量XCOMET终稿评分失败: {str(e)[:100]}")
                        elif error_count == 3:
                            print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                        for final_idx in final_xcomet_indices:
                            if 'xcomet_final' not in results[final_idx]:
                                results[final_idx]['xcomet_final'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                                }
                else:
                    print("[XCOMET] 没有需要评分的终稿翻译")
                
                if torch.cuda.is_available() and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                    print(f"[Stage] 恢复Qwen的CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")
        
        else:
            # ========== 基线模式流程 ==========
            
            # ========== 阶段1：批量生成所有数据的初稿 ==========
            print("\n" + "="*60)
            print("阶段1: 批量生成所有数据的初稿")
            print("="*60)
            
            all_draft_generated_texts = []
            for i in tqdm(range(0, num_samples, args.batch_size), desc="生成初稿"):
                batch = dataset_df.iloc[i:i+args.batch_size]
                prompts = batch['prompt'].tolist()
                
                try:
                    generated_texts = qwen_generator.generate_draft(
                        prompts,
                        mode="draft",
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )
                    if isinstance(generated_texts, str):
                        generated_texts = [generated_texts]
                    if len(generated_texts) != len(batch):
                        print(f"[Warning] 生成文本数量({len(generated_texts)})与批次大小({len(batch)})不匹配，使用空字符串填充")
                        generated_texts = list(generated_texts) + [""] * (len(batch) - len(generated_texts))
                except Exception as e:
                    print(f"\n[Error] Generation failed for batch starting at index {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    generated_texts = [""] * len(batch)
                
                all_draft_generated_texts.extend(generated_texts)
            
            # 保存初稿生成结果
            for idx, draft_text in enumerate(all_draft_generated_texts):
                results[idx]['draft_generated_text'] = draft_text
            
            # ========== 阶段2：对所有初稿进行格式检查 ==========
            print("\n" + "="*60)
            print("阶段2: 对所有初稿进行格式检查")
            print("="*60)
            
            for idx in tqdm(range(num_samples), desc="检查初稿格式"):
                draft_text = results[idx]['draft_generated_text']
                format_valid, draft_translation, format_score = check_and_extract_translate_tag(draft_text)
                results[idx]['draft_format_score'] = format_score
                results[idx]['draft_translation'] = draft_translation if format_valid else None
         
            # ========== 阶段3：对所有初稿翻译进行XCOMET评分 ==========
            print("\n" + "="*60)
            print("阶段3: 对所有初稿翻译进行XCOMET评分")
            print("="*60)
            
            if xcomet_loader:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 强制使用CPU模式（--xcomet_cpu）")
                elif args.xcomet_gpus:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 通过参数指定使用GPU: {xcomet_gpu_ids}")
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 使用环境变量CUDA_VISIBLE_DEVICES={xcomet_gpu_ids}")
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 默认使用CPU模式")

                draft_xcomet_triplets = []
                draft_xcomet_indices = []
                for idx in range(num_samples):
                    format_valid = results[idx].get('draft_format_score', 0) == 1
                    draft_translation = results[idx].get('draft_translation')
                    if format_valid and draft_translation:
                        src_text = str(results[idx]['src_text']).strip() if results[idx]['src_text'] else ""
                        ref_text = str(results[idx]['tgt_text']).strip() if results[idx]['tgt_text'] else ""
                        if src_text and ref_text:
                            draft_xcomet_triplets.append({
                                "src": src_text,
                                "mt": draft_translation,
                                "ref": ref_text
                            })
                            draft_xcomet_indices.append(idx)
                if draft_xcomet_triplets:
                    try:
                        print(f"[XCOMET] 批量评分 {len(draft_xcomet_triplets)} 个初稿翻译...")
                        xcomet_results = xcomet_loader.predict(
                            draft_xcomet_triplets,
                            batch_size=args.xcomet_batch_size,
                            return_system_score=True
                        )
                        for result_idx, xcomet_idx in enumerate(draft_xcomet_indices):
                            if result_idx < len(xcomet_results):
                                xcomet_analysis = xcomet_results[result_idx]
                                results[xcomet_idx]['xcomet_draft'] = xcomet_analysis
                            else:
                                results[xcomet_idx]['xcomet_draft'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": "XCOMET result index out of range",
                                }
                    except Exception as e:
                        error_count = xcomet_loader._error_count
                        xcomet_loader._error_count = error_count + 1
                        if error_count < 3:
                            print(f"[Warning] 批量XCOMET评分失败: {str(e)[:100]}")
                        elif error_count == 3:
                            print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                        for xcomet_idx in draft_xcomet_indices:
                            if 'xcomet_draft' not in results[xcomet_idx]:
                                results[xcomet_idx]['xcomet_draft'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                                }
                else:
                    print("[XCOMET] 没有需要评分的初稿翻译")

                for idx in range(num_samples):
                    if 'xcomet_draft' not in results[idx]:
                        format_valid = results[idx].get('draft_format_score', 0) == 1
                        draft_translation = results[idx].get('draft_translation')
                        if not format_valid or not draft_translation:
                            results[idx]['xcomet_draft'] = {
                                "score": None,
                                "error_spans": [],
                                "error": "Draft format invalid or translation is empty" if not format_valid else "Draft translation is empty",
                            }
                        else:
                            results[idx]['xcomet_draft'] = {
                                "score": None,
                                "error_spans": [],
                                "error": "XCOMET loader not available",
                            }

                if torch.cuda.is_available() and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                    print(f"[Stage] 恢复Qwen的CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")

            # ========== 阶段4：批量生成所有数据的终稿（repair） ==========
            print("\n" + "="*60)
            print("阶段4: 批量生成所有数据的终稿（repair）")
            print("="*60)
            
            # 收集需要repair的样本
            repair_prompts = []
            repair_indices = []
            
            for idx in range(num_samples):
                format_valid = results[idx].get('draft_format_score', 0) == 1
                draft_translation = results[idx].get('draft_translation')
                xcomet_draft = results[idx].get('xcomet_draft', {})
                error_spans = xcomet_draft.get('error_spans', []) if isinstance(xcomet_draft, dict) else []
                
                # 情况1：如果没有初稿，跳过refinement
                if not draft_translation:
                    results[idx]['repair_generated_text'] = None
                    results[idx]['repair_prompt'] = None
                    results[idx]['repair_format_score'] = 0
                    results[idx]['final_translation'] = None
                
                # 情况2：如果有初稿但无错误，跳过refinement，直接使用初稿
                elif format_valid and draft_translation and (not error_spans or len(error_spans) == 0):
                    results[idx]['repair_generated_text'] = None
                    results[idx]['repair_prompt'] = None
                    results[idx]['repair_format_score'] = 0
                    results[idx]['final_translation'] = draft_translation
                
                # 情况3：有初稿且有错误，需要repair
                elif format_valid and draft_translation and error_spans:
                    try:
                        repair_example = {
                            'lg': results[idx]['lang_pair'],
                            'src_text': results[idx]['src_text'],
                        }
                        repair_prompt = make_prefix(
                            repair_example,
                            template_type='repair',
                            tokenizer=tokenizer,
                            error_spans=error_spans,
                            draft_translation=draft_translation
                        )
                        repair_prompts.append(repair_prompt)
                        repair_indices.append(idx)
                    except Exception as e:
                        print(f"[Warning] 构建repair prompt失败 for sample {results[idx]['index']}: {str(e)[:100]}")
                        results[idx]['repair_generated_text'] = None
                        results[idx]['repair_prompt'] = None
                        results[idx]['repair_format_score'] = 0
                        results[idx]['final_translation'] = None
            
            # 批量生成repair翻译
            if repair_prompts:
                all_repair_generated_texts = []
                for i in tqdm(range(0, len(repair_prompts), args.batch_size), desc="生成repair翻译"):
                    batch_prompts = repair_prompts[i:i+args.batch_size]
                    try:
                        repair_texts = qwen_generator.generate_draft(
                            batch_prompts,
                            mode="repair",
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p
                        )
                        if isinstance(repair_texts, str):
                            repair_texts = [repair_texts]
                        if len(repair_texts) != len(batch_prompts):
                            print(f"[Warning] Repair生成文本数量({len(repair_texts)})与批次大小({len(batch_prompts)})不匹配")
                            repair_texts = list(repair_texts) + [""] * (len(batch_prompts) - len(repair_texts))
                    except Exception as e:
                        print(f"\n[Error] Repair generation failed for batch starting at index {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        repair_texts = [""] * len(batch_prompts)
                    
                    all_repair_generated_texts.extend(repair_texts)
                
                # 保存repair结果
                for prompt_idx, result_idx in enumerate(repair_indices):
                    if prompt_idx < len(all_repair_generated_texts):
                        repair_text = all_repair_generated_texts[prompt_idx]
                        results[result_idx]['repair_generated_text'] = repair_text
                        repair_example = {
                            'lg': results[result_idx]['lang_pair'],
                            'src_text': results[result_idx]['src_text'],
                        }
                        xcomet_draft = results[result_idx].get('xcomet_draft', {})
                        error_spans = xcomet_draft.get('error_spans', []) if isinstance(xcomet_draft, dict) else []
                        draft_translation = results[result_idx].get('draft_translation')
                        results[result_idx]['repair_prompt'] = make_prefix(
                            repair_example,
                            template_type='repair',
                            tokenizer=tokenizer,
                            error_spans=error_spans,
                            draft_translation=draft_translation
                        )
                    else:
                        results[result_idx]['repair_generated_text'] = None
                        results[result_idx]['repair_prompt'] = None
                        results[result_idx]['repair_format_score'] = 0
                        results[result_idx]['final_translation'] = None
            
            # ========== 阶段5：对所有终稿进行格式检查 ==========
            print("\n" + "="*60)
            print("阶段5: 对所有终稿进行格式检查")
            print("="*60)
            
            for idx in tqdm(range(num_samples), desc="检查终稿格式"):
                if results[idx].get('final_translation') is not None and results[idx].get('repair_generated_text') is None:
                    results[idx]['repair_format_score'] = 0
                    continue
                repair_generated_text = results[idx].get('repair_generated_text')
                if repair_generated_text:
                    repair_format_valid, final_translation, repair_format_score = check_and_extract_translate_tag(repair_generated_text)
                    results[idx]['repair_format_score'] = repair_format_score
                    results[idx]['final_translation'] = final_translation if repair_format_valid else None
                else:
                    if 'repair_format_score' not in results[idx]:
                        results[idx]['repair_format_score'] = 0
                    if 'final_translation' not in results[idx]:
                        results[idx]['final_translation'] = None
            
            # ========== 阶段6：对所有终稿翻译进行XCOMET评分 ==========
            print("\n" + "="*60)
            print("阶段6: 对所有终稿翻译进行XCOMET评分")
            print("="*60)
            
            if xcomet_loader:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 强制使用CPU模式（--xcomet_cpu）")
                elif args.xcomet_gpus and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 通过参数指定使用GPU: {xcomet_gpu_ids}")
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                    print(f"[XCOMET] 使用环境变量CUDA_VISIBLE_DEVICES={xcomet_gpu_ids}")
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                    print("[XCOMET] 默认使用CPU模式")
                
                final_xcomet_triplets = []
                final_xcomet_indices = []
                for idx in range(num_samples):
                    final_translation = results[idx].get('final_translation')
                    if final_translation:
                        src_text = str(results[idx]['src_text']).strip() if results[idx]['src_text'] else ""
                        ref_text = str(results[idx]['tgt_text']).strip() if results[idx]['tgt_text'] else ""
                        if src_text and ref_text:
                            final_xcomet_triplets.append({
                                "src": src_text,
                                "mt": final_translation,
                                "ref": ref_text
                            })
                            final_xcomet_indices.append(idx)
                if final_xcomet_triplets:
                    try:
                        print(f"[XCOMET] 批量评分 {len(final_xcomet_triplets)} 个终稿翻译...")
                        xcomet_final_results = xcomet_loader.predict(
                            final_xcomet_triplets,
                            batch_size=args.xcomet_batch_size,
                            return_system_score=True
                        )
                        for result_idx, final_idx in enumerate(final_xcomet_indices):
                            if result_idx < len(xcomet_final_results):
                                xcomet_analysis_final = xcomet_final_results[result_idx]
                                results[final_idx]['xcomet_final'] = xcomet_analysis_final
                            else:
                                results[final_idx]['xcomet_final'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": "XCOMET final result index out of range",
                                }
                    except Exception as e:
                        error_count = xcomet_loader._error_count
                        xcomet_loader._error_count = error_count + 1
                        if error_count < 3:
                            print(f"[Warning] 批量XCOMET终稿评分失败: {str(e)[:100]}")
                        elif error_count == 3:
                            print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                        for final_idx in final_xcomet_indices:
                            if 'xcomet_final' not in results[final_idx]:
                                results[final_idx]['xcomet_final'] = {
                                    "score": None,
                                    "error_spans": [],
                                    "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                                }
                else:
                    print("[XCOMET] 没有需要评分的终稿翻译")
                
                if torch.cuda.is_available():
                    if qwen_gpu_ids:
                        os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                        print(f"[Stage] 恢复Qwen的CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")
                    else:
                        # 如果没有指定qwen_gpu_ids，清除CUDA_VISIBLE_DEVICES（使用CPU）
                        if "CUDA_VISIBLE_DEVICES" in os.environ:
                            del os.environ["CUDA_VISIBLE_DEVICES"]
                        print(f"[Stage] 清除CUDA_VISIBLE_DEVICES（Qwen使用CPU）")
    
    # ========== 从results.json读取数据并进行refinement的处理 ==========
    if skip_draft_generation:
        # 需要加载Qwen和XCOMET模型
        print("\n" + "="*60)
        print("加载模型进行refinement")
        print("="*60)
        
        # 加载XCOMET（如果需要）
        if args.load_xcomet:
            if args.xcomet_ckpt:
                xcomet_ckpt = args.xcomet_ckpt
            elif os.getenv("WORD_QE_CKPT"):
                xcomet_ckpt = os.getenv("WORD_QE_CKPT")
            else:
                default_ckpt = os.path.expanduser("~/models/XCOMET-XL/checkpoints/model.ckpt")
                xcomet_ckpt = default_ckpt if os.path.exists(default_ckpt) else None
            
            if xcomet_ckpt:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, _ = get_xcomet_gpu_setting(args)
                
                try:
                    xcomet_loader = XCOMETLoader(
                        xcomet_ckpt,
                        force_cpu=args.xcomet_cpu or (xcomet_gpu_ids is None),
                        gpu_ids=xcomet_gpu_ids
                    )
                    print("[XCOMET] XCOMET model loaded successfully")
                except Exception as e:
                    print(f"[XCOMET] Failed to load XCOMET: {e}")
                    xcomet_loader = None
            else:
                xcomet_loader = None
        else:
            xcomet_loader = None
        
        # 加载Qwen模型
        print(f"[Qwen] Loading Qwen model from {args.qwen_model_path}...")
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        # GPU选择逻辑（与主流程一致）：
        # 1. 默认：CPU模式
        # 2. 如果设置了 --qwen_cpu，强制CPU模式
        # 3. 如果设置了 --qwen_gpus，使用指定的GPU
        # 4. 如果只设置了CUDA_VISIBLE_DEVICES，且没有分别设置qwen_gpus和xcomet_gpus，则使用CUDA_VISIBLE_DEVICES
        # 5. 否则，使用CPU
        import torch
        if args.qwen_cpu:
            # 强制CPU模式
            qwen_gpu_ids = None
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print("[Qwen] 强制使用CPU模式（--qwen_cpu）")
        elif args.qwen_gpus:
            # 通过参数指定GPU
            qwen_gpu_ids = args.qwen_gpus
            os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            print(f"[Qwen] 通过参数指定使用GPU: {qwen_gpu_ids}")
        elif original_cuda_visible and not args.xcomet_gpus:
            # 如果只设置了CUDA_VISIBLE_DEVICES，且没有分别设置xcomet_gpus，则两者都使用CUDA_VISIBLE_DEVICES
            qwen_gpu_ids = original_cuda_visible
            os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            print(f"[Qwen] 使用环境变量CUDA_VISIBLE_DEVICES={qwen_gpu_ids}")
        else:
            # 默认CPU模式
            qwen_gpu_ids = None
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print("[Qwen] 默认使用CPU模式")
        
        vllm_extra_kwargs = {}
        if args.use_vllm:
            if args.vllm_max_num_seqs:
                vllm_extra_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
            vllm_extra_kwargs["disable_custom_all_reduce"] = True

        try:
            qwen_generator = QwenGenerator(
                model_path=args.qwen_model_path,
                use_vllm=args.use_vllm if not args.qwen_cpu else False,  # CPU模式下不使用vLLM
                device="cpu" if args.qwen_cpu else None,  # CPU模式下明确指定device
                gpu_memory_utilization=args.gpu_memory_utilization,
                **vllm_extra_kwargs
            )
            print("[Qwen] Qwen model loaded successfully")
        except Exception as e:
            print(f"[Qwen] Failed to load Qwen model: {e}")
            qwen_generator = QwenGenerator(
                model_path=args.qwen_model_path,
                use_vllm=False,
                device="cpu" if args.qwen_cpu else None,  # CPU模式下明确指定device
                gpu_memory_utilization=args.gpu_memory_utilization
            )
        
        # 处理每个已有结果
        results = []
        
        # ========== 步骤5：先处理格式检查，收集需要XCOMET评分的样本 ==========
        batch_results = []
        draft_xcomet_triplets = []
        draft_xcomet_indices = []
        
        for idx, (_, row) in enumerate(dataset_df.iterrows()):
            result = row.to_dict()
            
            # 从results.json中读取draft_generated_text
            draft_text = result.get('draft_generated_text', '')
            if not draft_text:
                # 如果没有draft_generated_text，尝试使用generated_text
                draft_text = result.get('generated_text', '')
            
            # 检查draft格式，提取<translate>部分作为初稿
            format_valid, draft_translation, format_score = check_and_extract_translate_tag(draft_text)
            
            # 更新结果
            result['draft_format_score'] = format_score
            result['draft_translation'] = draft_translation if format_valid else None
            
            # 如果results.json中已有xcomet_draft数据，使用它
            if result.get('xcomet_draft') and isinstance(result['xcomet_draft'], dict):
                # 已有XCOMET结果，直接使用
                pass
            elif format_valid and draft_translation and xcomet_loader:
                # 需要XCOMET评分，收集样本
                src_text = str(result.get('src_text', '')).strip()
                ref_text = str(result.get('tgt_text', '')).strip()
                if src_text and ref_text:
                    draft_xcomet_triplets.append({
                        "src": src_text,
                        "mt": draft_translation,
                        "ref": ref_text
                    })
                    draft_xcomet_indices.append(idx)
            else:
                result['xcomet_draft'] = {
                    "score": None,
                    "error_spans": [],
                    "error": "XCOMET not available or format invalid",
                }
            
            batch_results.append(result)
        
        # ========== 步骤6：批量调用XCOMET对draft进行评分 ==========
        if draft_xcomet_triplets and xcomet_loader:
            try:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                xcomet_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                elif args.xcomet_gpus and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                
                # 批量调用XCOMET
                xcomet_results = xcomet_loader.predict(
                    draft_xcomet_triplets,
                    batch_size=args.xcomet_batch_size,
                    return_system_score=True
                )
                
                # 将结果映射回每个样本
                for result_idx, xcomet_idx in enumerate(draft_xcomet_indices):
                    if result_idx < len(xcomet_results):
                        xcomet_analysis = xcomet_results[result_idx]
                        batch_results[xcomet_idx]['xcomet_draft'] = xcomet_analysis
                    else:
                        batch_results[xcomet_idx]['xcomet_draft'] = {
                            "score": None,
                            "error_spans": [],
                            "error": "XCOMET result index out of range",
                        }
                
                # 恢复Qwen的CUDA_VISIBLE_DEVICES
                if xcomet_cuda_visible is not None and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            except Exception as e:
                # 恢复Qwen的CUDA_VISIBLE_DEVICES
                if xcomet_cuda_visible is not None and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                
                error_count = xcomet_loader._error_count
                xcomet_loader._error_count = error_count + 1
                if error_count < 3:
                    print(f"[Warning] 批量XCOMET评分失败: {str(e)[:100]}")
                elif error_count == 3:
                    print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                
                # 为所有需要评分的样本设置错误信息
                for xcomet_idx in draft_xcomet_indices:
                    if 'xcomet_draft' not in batch_results[xcomet_idx] or not isinstance(batch_results[xcomet_idx].get('xcomet_draft'), dict):
                        batch_results[xcomet_idx]['xcomet_draft'] = {
                            "score": None,
                            "error_spans": [],
                            "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                        }
        
        # ========== 步骤7-8：处理refinement ==========
        for idx, result in enumerate(tqdm(batch_results, desc="Processing refinement")):
            format_valid = result.get('draft_format_score', 0) == 1
            draft_translation = result.get('draft_translation')
            xcomet_draft = result.get('xcomet_draft', {})
            error_spans = xcomet_draft.get('error_spans', []) if isinstance(xcomet_draft, dict) else []
            
            # ========== 步骤7-8：处理repair和final_translation ==========
            repair_generated_text = None
            repair_prompt = None
            repair_format_valid = False
            repair_format_score = 0
            final_translation = None
            
            # 情况1：如果没有初稿，跳过refinement，所有repair参数和final_translation都为空
            if not draft_translation:
                print(f"[Info] 样本 {result.get('index', '?')} 无初稿，跳过refinement")
                # 所有字段保持为None/False/0，已在上面初始化
            
            # 情况2：如果有初稿，检查是否有错误spans
            elif format_valid and draft_translation:
                if not error_spans or len(error_spans) == 0:
                    # 如果没有错误spans，说明初稿完美，跳过refinement
                    # repair所有参数为空，但final_translation直接设置为初稿翻译本身
                    # 注意：这种情况下final_translation仍会被XCOMET评分（在后续批量评分阶段）
                    print(f"[Info] 样本 {result.get('index', '?')} 初稿无错误，跳过refinement，直接使用初稿作为终稿（将对终稿进行XCOMET评分）")
                    repair_generated_text = None
                    repair_prompt = None
                    repair_format_valid = False
                    repair_format_score = 0
                    final_translation = draft_translation
                else:
                    # 有错误spans，进行repair
                    try:
                        # 构建repair prompt（直接传递error_spans列表）
                        repair_example = {
                            'lg': result.get('lang_pair', ''),
                            'src_text': result.get('src_text', ''),
                        }
                        repair_prompt = make_prefix(
                            repair_example,
                            template_type='repair',
                            tokenizer=tokenizer,
                            error_spans=error_spans,  # 直接传递error_spans列表
                            draft_translation=draft_translation
                        )
                        repair_texts = qwen_generator.generate_draft(
                            repair_prompt,
                            mode="repair",
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p
                        )
                        repair_generated_text = repair_texts if isinstance(repair_texts, str) else repair_texts[0] if repair_texts else None
                        
                        # 检查repair格式，提取<translate>部分作为终稿
                        if repair_generated_text:
                            repair_format_valid, final_translation, repair_format_score = check_and_extract_translate_tag(repair_generated_text)
                    
                    except Exception as e:
                        print(f"[Warning] Repair generation failed for sample {result.get('index', '?')}: {str(e)[:100]}")
                        repair_generated_text = None
                        repair_prompt = None
                        repair_format_valid = False
                        repair_format_score = 0
                        final_translation = None
            
            # 保存结果
            result['repair_generated_text'] = repair_generated_text
            result['repair_prompt'] = repair_prompt
            result['repair_format_score'] = repair_format_score
            result['final_translation'] = final_translation
            
            batch_results[idx] = result
        
        # ========== 批量对终稿调用XCOMET评分（可选） ==========
        final_xcomet_triplets = []
        final_xcomet_indices = []
        
        for idx, result in enumerate(batch_results):
            # 注意：现在final_translation可能不是通过repair_format_valid判断的
            # 如果没有错误spans，final_translation会直接设置为初稿，但repair_format_score为0
            # 这种情况下也需要对final_translation进行XCOMET评分
            final_translation = result.get('final_translation')
            # 只要有final_translation就可以进行终稿评分（包括：repair后的终稿、无错误时直接使用初稿作为终稿）
            if final_translation and xcomet_loader:
                src_text = str(result.get('src_text', '')).strip()
                ref_text = str(result.get('tgt_text', '')).strip()
                if src_text and ref_text:
                    final_xcomet_triplets.append({
                        "src": src_text,
                        "mt": final_translation,
                        "ref": ref_text
                    })
                    final_xcomet_indices.append(idx)
        
        if final_xcomet_triplets and xcomet_loader:
            try:
                # 使用统一的XCOMET GPU选择逻辑
                xcomet_gpu_ids, should_set_env = get_xcomet_gpu_setting(args)
                xcomet_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                
                if args.xcomet_cpu:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                elif args.xcomet_gpus and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                elif xcomet_gpu_ids and should_set_env:
                    os.environ["CUDA_VISIBLE_DEVICES"] = xcomet_gpu_ids
                else:
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                
                # 批量调用XCOMET对终稿评分
                xcomet_final_results = xcomet_loader.predict(
                    final_xcomet_triplets,
                    batch_size=args.xcomet_batch_size,
                    return_system_score=True
                )
                
                # 将结果映射回每个样本
                for result_idx, final_idx in enumerate(final_xcomet_indices):
                    if result_idx < len(xcomet_final_results):
                        xcomet_analysis_final = xcomet_final_results[result_idx]
                        batch_results[final_idx]['xcomet_final'] = xcomet_analysis_final
                    else:
                        batch_results[final_idx]['xcomet_final'] = {
                            "score": None,
                            "error_spans": [],
                            "error": "XCOMET final result index out of range",
                        }
                
                # 恢复Qwen的CUDA_VISIBLE_DEVICES
                if xcomet_cuda_visible is not None and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
            except Exception as e:
                # 恢复Qwen的CUDA_VISIBLE_DEVICES
                if xcomet_cuda_visible is not None and qwen_gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = qwen_gpu_ids
                
                error_count = xcomet_loader._error_count
                xcomet_loader._error_count = error_count + 1
                if error_count < 3:
                    print(f"[Warning] 批量XCOMET终稿评分失败: {str(e)[:100]}")
                elif error_count == 3:
                    print(f"[Warning] XCOMET错误过多，后续错误将静默处理...")
                
                # 为所有需要评分的样本设置错误信息
                for final_idx in final_xcomet_indices:
                    if 'xcomet_final' not in batch_results[final_idx]:
                        batch_results[final_idx]['xcomet_final'] = {
                            "score": None,
                            "error_spans": [],
                            "error": str(e)[:200] if error_count < 3 else "Multiple errors (suppressed)",
                        }
        
        results = batch_results
    
    print(f"\n[Generation] Generated {len(results)} translations")
    
    # ========== 5. 保存结果 ==========
    if args.output_file:
        print("\n" + "="*60)
        print("步骤5: 保存结果")
        print("="*60)
        output_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.output_file
        )
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Output] Results saved to {output_path}")
        
        # 打印统计信息
        # 格式统计
        draft_format_scores = [r.get('draft_format_score', 0) for r in results]
        repair_format_scores = [r.get('repair_format_score', 0) for r in results]
        draft_valid_count = sum(draft_format_scores)
        repair_valid_count = sum(repair_format_scores)
        print(f"[Stats] Draft格式正确率: {draft_valid_count}/{len(results)} ({100*draft_valid_count/len(results):.1f}%)")
        print(f"[Stats] Repair格式正确率: {repair_valid_count}/{len(results)} ({100*repair_valid_count/len(results):.1f}%)")
        
        # XCOMET统计（初稿）
        if xcomet_loader:
            draft_scores = [
                r.get('xcomet_draft', {}).get('score')
                for r in results
                if r.get('xcomet_draft') and r['xcomet_draft'].get('score') is not None
            ]
            if draft_scores:
                print(
                    "[Stats] XCOMET Draft scores - "
                    f"Mean: {sum(draft_scores)/len(draft_scores):.4f}, "
                    f"Min: {min(draft_scores):.4f}, Max: {max(draft_scores):.4f}"
                )
            
            draft_error_span_counts = [
                len(r.get('xcomet_draft', {}).get('error_spans', []))
                for r in results
                if r.get('xcomet_draft')
            ]
            if draft_error_span_counts:
                avg_draft_spans = sum(draft_error_span_counts) / len(draft_error_span_counts)
                print(f"[Stats] Avg. error spans per draft sample: {avg_draft_spans:.2f}")
            
            # 终稿错误spans统计
            final_error_span_counts = [
                len(r.get('xcomet_final', {}).get('error_spans', []))
                for r in results
                if r.get('xcomet_final')
            ]
            if final_error_span_counts:
                avg_final_spans = sum(final_error_span_counts) / len(final_error_span_counts)
                print(f"[Stats] Avg. error spans per final sample: {avg_final_spans:.2f}")
            
            # XCOMET统计（终稿）
            final_scores = [
                r.get('xcomet_final', {}).get('score')
                for r in results
                if r.get('xcomet_final') and r['xcomet_final'].get('score') is not None
            ]
            if final_scores:
                print(
                    "[Stats] XCOMET Final scores - "
                    f"Mean: {sum(final_scores)/len(final_scores):.4f}, "
                    f"Min: {min(final_scores):.4f}, Max: {max(final_scores):.4f}"
                )
            
            # 改进统计
            if draft_scores and final_scores:
                improved_count = sum(1 for d, f in zip(draft_scores, final_scores) if f and d and f > d)
                print(f"[Stats] 终稿改进初稿的样本数: {improved_count}/{len(final_scores)} ({100*improved_count/len(final_scores):.1f}%)")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

