# -*- coding: utf-8 -*-
"""
XCOMET模型加载器
复用MT_Grpo/verl/comet_reward_batch_with_ray.py中的实现
"""

import os
import torch
from typing import List, Dict, Optional, Any
from comet import load_from_checkpoint

# ================== 配置 ==================
_WORD_LEVEL_BATCH = int(os.getenv("WORD_LEVEL_BATCH", "32"))  # xcomet batch size
_WORD_QE_CKPT = os.getenv("WORD_QE_CKPT")


class XCOMETLoader:
    """XCOMET模型加载器，复用MT_Grpo的实现方式"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None, force_cpu: bool = False, gpu_ids: Optional[str] = None):
        """
        初始化XCOMET加载器
        
        Args:
            checkpoint_path: XCOMET模型checkpoint路径，如果为None则从环境变量WORD_QE_CKPT获取
            device: 设备 ('cuda' or 'cpu')，如果为None则自动选择
            force_cpu: 强制使用CPU（用于避免CUDA错误）
            gpu_ids: 指定使用的GPU编号，如"0,1"（仅在device='cuda'时有效）
        """
        self.checkpoint_path = checkpoint_path or _WORD_QE_CKPT
        if not self.checkpoint_path:
            raise ValueError(
                "XCOMET checkpoint path not provided. "
                "Please set --xcomet_ckpt argument or WORD_QE_CKPT environment variable."
            )
        self.force_cpu = force_cpu
        if force_cpu:
            self.device = "cpu"
            print("[WORD-QE] 强制使用CPU模式（避免CUDA错误）")
        else:
            if gpu_ids:
                # 解析GPU编号，使用第一个GPU作为主设备
                gpu_list = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
                if gpu_list:
                    self.device = f"cuda:{gpu_list[0]}"
                    print(f"[WORD-QE] 指定使用GPU: {gpu_list}，主设备: {self.device}")
                else:
                    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = gpu_ids
        self.model = None
        self._error_count = 0
        self._load_model()
    
    def _load_model(self):
        """加载XCOMET模型，复用MT_Grpo的加载方式"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"XCOMET checkpoint not found: {self.checkpoint_path}\n"
                f"Please download it first using download_comet_ckpts.py"
            )
        
        print(f"[WORD-QE] Loading XCOMET model from {self.checkpoint_path}...")
        try:
            # 复用MT_Grpo中的加载方式
            # XCOMET使用指定设备（可以是任何GPU）
            model_device = self.device
            self.model = load_from_checkpoint(self.checkpoint_path).to(model_device)
            self.model.eval()
            print(f"[WORD-QE] XCOMET model loaded on {model_device}")
        except Exception as e:
            print(f"[WORD-QE] Load failed: {e}")
            raise
    
    def predict(
        self,
        triplets: List[Dict[str, str]],
        batch_size: Optional[int] = None,
        return_system_score: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        使用XCOMET对翻译进行评分，并返回错误片段信息。

        Args:
            triplets: 三元组列表，每个元素包含 {"src": source_text, "mt": translation, "ref": reference}
            batch_size: 批处理大小，如果为None则使用默认值
            return_system_score: 是否在每条结果中包含system-level评分

        Returns:
            列表，每个元素包含：
                - score: 句子级别得分（浮点数）
                - error_spans: 错误片段信息（若模型提供）
                - metadata: 预测返回的原始metadata（若存在）
                - system_score: system-level得分（可选）
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the loader first.")
        
        if not triplets:
            return []
        
        batch_size = batch_size or _WORD_LEVEL_BATCH
        
        # XCOMET使用GPU进行推理
        use_gpu = 1 if (self.device.startswith("cuda") and torch.cuda.is_available()) else 0
        
        # 验证数据：检查是否有空字符串或无效数据
        # 注意：ref字段是可选的，如果不存在或为空，仍然可以评分（用于扩展模式）
        valid_triplets = []
        valid_indices = []
        for idx, triplet in enumerate(triplets):
            src = triplet.get("src", "").strip()
            mt = triplet.get("mt", "").strip()
            ref = triplet.get("ref", "").strip() if "ref" in triplet else None
            
            # src和mt必须存在，ref是可选的
            if src and mt:
                valid_triplets.append(triplet)
                # print(f"[WORD-QE] Valid sample: {triplet} ")
                valid_indices.append(idx)
            else:
                print(f"[WORD-QE] Warning: 跳过无效样本 {idx} (src或mt为空)")
        
        if not valid_triplets:
            print("[WORD-QE] Warning: 所有样本都无效，返回空结果")
            return []
        
        try:
            # 运行XCOMET推理
            # 注意：对于推理任务，使用 gpus=1 避免分布式训练（多进程）导致的重复加载
            # 如果需要多GPU加速，应该使用 DataParallel 而不是 DDP
            if self.force_cpu or not use_gpu:
                num_gpus_to_use = 0
                print(f"[WORD-QE] 使用CPU模式进行评分（batch_size={len(valid_triplets)}）")
            else:
                # 推理任务只使用单GPU，避免分布式训练导致的多进程重复加载
                num_gpus_to_use = 1
                print(f"[WORD-QE] 使用GPU模式进行评分（batch_size={len(valid_triplets)}，使用单GPU避免多进程重复）")
            output = self.model.predict(valid_triplets, batch_size=batch_size, gpus=num_gpus_to_use)
            results: List[Dict[str, Any]] = []

            # 提取句子级得分和元数据
            if isinstance(output, dict):
                scores = output.get("scores", [])
                metadata = output.get("metadata", None)
                system_score = output.get("system_score", None)
            else:
                scores = list(output.scores) if hasattr(output, "scores") else []
                metadata = getattr(output, "metadata", None)
                system_score = getattr(output, "system_score", None)

            # 提取错误片段
            error_spans = None
            if metadata is not None:
                # metadata 可能是dict或对象
                if isinstance(metadata, dict):
                    error_spans = metadata.get("error_spans")
                else:
                    error_spans = getattr(metadata, "error_spans", None)

            # 构建完整结果列表（包括被跳过的无效样本）
            all_results = []
            valid_result_idx = 0
            for original_idx in range(len(triplets)):
                if original_idx in valid_indices:
                    # 这是有效样本，使用XCOMET的结果
                    item = {
                        "score": scores[valid_result_idx],
                        "error_spans": error_spans[valid_result_idx] if error_spans and valid_result_idx < len(error_spans) else [],
                    }
                    if return_system_score and system_score is not None:
                        item["system_score"] = system_score
                    all_results.append(item)
                    valid_result_idx += 1
                else:
                    # 这是无效样本，返回默认值
                    item = {
                        "score": 0.0,
                        "error_spans": [],
                    }
                    if return_system_score:
                        item["system_score"] = None
                    all_results.append(item)
            
            return all_results
        except RuntimeError as e:
            error_msg = str(e)
            # CUDA错误处理
            if "CUDA" in error_msg or "device-side assert" in error_msg:
                print(f"[WORD-QE] CUDA错误（可能由数据问题引起）: {error_msg[:200]}")
                print("[WORD-QE] 尝试使用CPU模式或检查数据质量")
                # 返回默认结果而不是抛出异常
                return [{"score": 0.0, "error_spans": []} for _ in triplets]
            else:
                print(f"[WORD-QE] Scoring failed: {error_msg[:200]}")
                # 返回默认结果
                return [{"score": 0.0, "error_spans": []} for _ in triplets]
        except Exception as e:
            error_msg = str(e)
            print(f"[WORD-QE] Scoring failed: {error_msg[:200]}")
            # 返回默认结果而不是抛出异常
            return [{"score": 0.0, "error_spans": []} for _ in triplets]
    
    def score(
        self,
        triplets: List[Dict[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """兼容旧接口，仅返回句子级得分。"""
        results = self.predict(triplets, batch_size=batch_size, return_system_score=False)
        return [item.get("score", 0.0) for item in results]

    def analyze_single(
        self, src: str, mt: str, ref: str, include_system_score: bool = False
    ) -> Dict[str, Any]:
        """对单个样本进行分析，返回得分与错误片段。"""
        results = self.predict(
            [{"src": src, "mt": mt, "ref": ref}],
            batch_size=1,
            return_system_score=include_system_score,
        )
        return results[0] if results else {"score": 0.0, "error_spans": []}

    def score_single(self, src: str, mt: str, ref: str) -> float:
        """兼容旧接口，仅返回得分。"""
        return self.analyze_single(src, mt, ref).get("score", 0.0)

