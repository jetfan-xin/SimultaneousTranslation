# -*- coding: utf-8 -*-
"""
Qwen2.5-3B模型调用模块
复用MT_Grpo中vllm的使用方式
"""

import os
import logging
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入vllm，如果不可用则设置为None
try:
    from vllm import LLM, SamplingParams
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("vllm.engine").setLevel(logging.WARNING)
    logging.getLogger("vllm.executor").setLevel(logging.WARNING)
    logging.getLogger("vllm.worker").setLevel(logging.WARNING)
    VLLM_AVAILABLE = True
    print("[INFO] vLLM imported successfully")
except ImportError as e:
    VLLM_AVAILABLE = False
    print(f"[Warning] vLLM import failed: {e}")
    print("[Warning] Will use transformers backend instead")
except Exception as e:
    VLLM_AVAILABLE = False
    print(f"[Warning] vLLM import error: {type(e).__name__}: {e}")
    print("[Warning] Will use transformers backend instead")

class QwenGenerator:
    """Qwen2.5-3B模型生成器，支持draft mode"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        use_vllm: bool = True,
        device: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,  # test_time: 0.85
        **kwargs
    ):
        """
        初始化Qwen生成器（参考 test_time/vllm_infer.py 的配置）
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            use_vllm: 是否使用vllm进行推理（推荐，速度更快）
            device: 设备 ('cuda' or 'cpu')，如果为None则自动选择
            gpu_memory_utilization: vLLM的GPU内存使用率（默认0.85，test_time: 0.85）
            **kwargs: 其他vllm参数（tensor_parallel_size等）
        """
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # 警告：CPU模式会很慢
        if self.device == "cpu":
            print("[WARNING] 使用CPU模式，推理会非常慢！")
            print("[WARNING] 建议：1) 使用GPU 2) 安装vllm 3) 减少max_tokens")
        else:
            # 根据CUDA_VISIBLE_DEVICES确定GPU数量
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                try:
                    device_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                    num_gpus = len(device_ids)
                    print(f"[INFO] 根据CUDA_VISIBLE_DEVICES检测到 {num_gpus} 个GPU: {device_ids}")
                except ValueError:
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    if num_gpus > 1:
                        print(f"[INFO] 检测到 {num_gpus} 个GPU，将充分利用GPU资源")
            else:
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if num_gpus > 1:
                    print(f"[INFO] 检测到 {num_gpus} 个GPU，将充分利用GPU资源")
        
        self.tokenizer = None
        self.model = None
        self.vllm_engine = None
        self._load_model(**kwargs)
    
    def _load_model(self, **kwargs):
        """加载模型和tokenizer，复用MT_Grpo中vllm的使用方式"""
        print(f"[Qwen] Loading model from {self.model_path}...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        if self.use_vllm and VLLM_AVAILABLE:
            # 检查是否有GPU可用，vLLM需要GPU
            if not torch.cuda.is_available():
                print("[Qwen] 未检测到GPU，vLLM需要GPU，将使用transformers后端")
                self.use_vllm = False
            else:
                try:
                    # 检查CUDA_VISIBLE_DEVICES格式（vLLM对格式很敏感）
                    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if cuda_visible:
                        # 检查是否有尾随逗号或其他格式问题
                        if cuda_visible.endswith(","):
                            print(f"[Warning] CUDA_VISIBLE_DEVICES末尾有多余逗号: '{cuda_visible}'")
                            print(f"[Warning] 这可能导致vLLM解析错误，建议移除末尾逗号")
                            # 自动修复
                            cuda_visible = cuda_visible.rstrip(",")
                            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
                            print(f"[Info] 已自动修复为: '{cuda_visible}'")
                        
                        # 验证格式是否有效
                        try:
                            device_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                            print(f"[Info] CUDA_VISIBLE_DEVICES解析为: {device_ids}")
                        except ValueError as e:
                            print(f"[Warning] CUDA_VISIBLE_DEVICES格式无效: '{cuda_visible}'")
                            print(f"[Warning] 这可能导致vLLM初始化失败")
                    
                    # 根据 test_time/vllm_infer.py 的设置，默认使用 tensor_parallel_size=1
                    # 如果用户通过 kwargs 指定了 tensor_parallel_size，则使用用户指定的值
                    if "tensor_parallel_size" not in kwargs:
                        kwargs["tensor_parallel_size"] = 1  # test_time: 1
                        print(f"[Qwen] 使用默认 tensor_parallel_size=1（test_time 配置）")
                    
                    # 自动检测并选择内存最充足的 GPU
                    # 保存原始的 CUDA_VISIBLE_DEVICES
                    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    selected_physical_gpu_id = None
                    
                    if self.device.startswith("cuda") and torch.cuda.is_available():
                        try:
                            requested_gpu_memory_util = kwargs.get("gpu_memory_utilization", self.gpu_memory_utilization)
                            num_visible_gpus = torch.cuda.device_count()
                            
                            # 解析原始的物理 GPU ID
                            original_physical_gpu_ids = []
                            if original_cuda_visible:
                                try:
                                    original_physical_gpu_ids = [int(x.strip()) for x in original_cuda_visible.split(",") if x.strip()]
                                except ValueError:
                                    pass
                            
                            # 检测所有可见 GPU 的内存
                            # 使用 nvidia-smi 获取更准确的内存信息（包括其他进程占用的内存）
                            gpu_memory_info = []
                            for visible_gpu_id in range(num_visible_gpus):
                                # 获取对应的物理 GPU ID
                                physical_gpu_id = original_physical_gpu_ids[visible_gpu_id] if visible_gpu_id < len(original_physical_gpu_ids) else visible_gpu_id
                                
                                total_memory = torch.cuda.get_device_properties(visible_gpu_id).total_memory / 1024**3  # GB
                                reserved_memory = torch.cuda.memory_reserved(visible_gpu_id) / 1024**3  # GB
                                
                                # 尝试使用 nvidia-smi 获取更准确的内存使用情况（包括其他进程）
                                try:
                                    import subprocess
                                    # 使用 nvidia-smi 查询实际内存使用
                                    result = subprocess.run(
                                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", 
                                         "--format=csv,noheader,nounits", f"--id={physical_gpu_id}"],
                                        capture_output=True, text=True, timeout=2
                                    )
                                    if result.returncode == 0:
                                        parts = result.stdout.strip().split(", ")
                                        if len(parts) >= 3:
                                            used_memory = float(parts[1]) / 1024  # MB to GB
                                            total_memory_smi = float(parts[2]) / 1024  # MB to GB
                                            # 使用 nvidia-smi 的数据更准确
                                            free_memory = total_memory_smi - used_memory
                                            reserved_memory = used_memory  # 更新为实际使用
                                        else:
                                            # 回退到 torch 的检测
                                            free_memory = total_memory - reserved_memory
                                    else:
                                        # 回退到 torch 的检测
                                        free_memory = total_memory - reserved_memory
                                except Exception:
                                    # 如果 nvidia-smi 失败，使用 torch 的检测
                                    free_memory = total_memory - reserved_memory
                                
                                required_memory = total_memory * requested_gpu_memory_util
                                
                                gpu_memory_info.append({
                                    "visible_gpu_id": visible_gpu_id,
                                    "physical_gpu_id": physical_gpu_id,
                                    "total": total_memory,
                                    "reserved": reserved_memory,
                                    "free": free_memory,
                                    "required": required_memory,
                                    "sufficient": free_memory >= required_memory
                                })
                            
                            # 打印所有 GPU 的内存信息
                            print(f"[Qwen] 检测 {num_visible_gpus} 个可见 GPU 的内存状态：")
                            for info in gpu_memory_info:
                                status = "✓ 充足" if info["sufficient"] else "✗ 不足"
                                print(f"[Qwen]   可见 GPU {info['visible_gpu_id']} (物理 GPU {info['physical_gpu_id']}): "
                                      f"总内存={info['total']:.2f} GB, 已用={info['reserved']:.2f} GB, "
                                      f"可用={info['free']:.2f} GB, 需要={info['required']:.2f} GB {status}")
                            
                            # 优先选择内存充足的 GPU，如果都充足则选择可用内存最多的
                            sufficient_gpus = [info for info in gpu_memory_info if info["sufficient"]]
                            if sufficient_gpus:
                                # 选择可用内存最多的 GPU
                                selected_gpu_info = max(sufficient_gpus, key=lambda x: x["free"])
                                selected_physical_gpu_id = selected_gpu_info["physical_gpu_id"]
                                print(f"[Qwen] 自动选择物理 GPU {selected_physical_gpu_id}（可见 GPU {selected_gpu_info['visible_gpu_id']}，"
                                      f"可用内存最多: {selected_gpu_info['free']:.2f} GB）")
                            else:
                                # 如果没有充足的 GPU，选择可用内存最多的
                                selected_gpu_info = max(gpu_memory_info, key=lambda x: x["free"])
                                selected_physical_gpu_id = selected_gpu_info["physical_gpu_id"]
                                # 自动降低 gpu_memory_utilization
                                actual_util = max(0.3, (selected_gpu_info["free"] - 1.0) / selected_gpu_info["total"])
                                kwargs["gpu_memory_utilization"] = actual_util
                                print(f"[Qwen] 所有 GPU 内存都不足，选择物理 GPU {selected_physical_gpu_id}（可见 GPU {selected_gpu_info['visible_gpu_id']}，"
                                      f"可用内存最多: {selected_gpu_info['free']:.2f} GB）")
                                print(f"[Qwen] 自动降低 gpu_memory_utilization: {requested_gpu_memory_util} -> {actual_util:.2f}")
                            
                            # 更新 CUDA_VISIBLE_DEVICES 以只使用选中的 GPU
                            # vLLM 使用 tensor_parallel_size=1，只需要一个 GPU
                            if selected_physical_gpu_id is not None:
                                os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_physical_gpu_id)
                                print(f"[Qwen] 更新 CUDA_VISIBLE_DEVICES={selected_physical_gpu_id}（vLLM 将使用此 GPU）")
                        except Exception as mem_check_err:
                            print(f"[Qwen] 无法检测 GPU 内存，使用默认 GPU 0: {mem_check_err}")
                            import traceback
                            traceback.print_exc()
                    
                    if "dtype" not in kwargs:
                        preferred_dtype = "float16"
                        try:
                            target_device_index = 0
                            device_name = torch.cuda.get_device_name(target_device_index)
                            major, minor = torch.cuda.get_device_capability(target_device_index)
                            if "A100" in device_name or "H100" in device_name or major >= 9:
                                preferred_dtype = "bfloat16"
                            else:
                                preferred_dtype = "float16"
                            if preferred_dtype == "float16":
                                print(f"[Qwen] 检测到设备 {device_name} (cc {major}.{minor})，自动使用float16以避免bfloat16导致的数值问题")
                            else:
                                print(f"[Qwen] 检测到设备 {device_name} 支持bfloat16，将使用bfloat16")
                        except Exception as auto_dtype_err:
                            print(f"[Qwen] 自动检测dtype失败，原因: {auto_dtype_err}".rstrip())
                            preferred_dtype = "auto"
                        kwargs["dtype"] = preferred_dtype

                    # 根据 test_time/vllm_infer.py 的设置来配置 vLLM
                    # max_model_len: test_time 中使用 16384
                    default_max_model_len = kwargs.get("max_model_len", 16384)
                    if default_max_model_len > 32768:
                        print(f"[Qwen] Warning: max_model_len={default_max_model_len} 可能过大，将限制到 16384")
                        default_max_model_len = 16384
                    
                    # 默认启用 disable_custom_all_reduce（test_time 中设置）
                    disable_custom_all_reduce = kwargs.get("disable_custom_all_reduce", True)
                    
                    vllm_kwargs = {
                        "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
                        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", self.gpu_memory_utilization),
                        "trust_remote_code": True,
                        "max_model_len": default_max_model_len,
                        "dtype": kwargs.get("dtype", "auto"),
                        "max_num_seqs": kwargs.get("max_num_seqs", None),
                        "disable_custom_all_reduce": disable_custom_all_reduce,
                        "enforce_eager": kwargs.get("enforce_eager", True),  # test_time 中设置，强制使用 eager 模式
                    }
                    # 移除None值
                    vllm_kwargs = {k: v for k, v in vllm_kwargs.items() if v is not None}
                    
                    print(f"[Qwen] Loading vLLM with tensor_parallel_size={vllm_kwargs.get('tensor_parallel_size')}, gpu_memory_utilization={vllm_kwargs.get('gpu_memory_utilization')}")
                    self.vllm_engine = LLM(model=self.model_path, **vllm_kwargs)
                    print(f"[Qwen] Model loaded with vLLM successfully")
                    
                    # 恢复原始的 CUDA_VISIBLE_DEVICES（不影响其他组件如 XCOMET）
                    if original_cuda_visible and selected_physical_gpu_id is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        print(f"[Qwen] 恢复 CUDA_VISIBLE_DEVICES={original_cuda_visible}")
                except Exception as e:
                    # 恢复原始的 CUDA_VISIBLE_DEVICES（即使初始化失败）
                    if original_cuda_visible and selected_physical_gpu_id is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                        print(f"[Qwen] 恢复 CUDA_VISIBLE_DEVICES={original_cuda_visible}")
                    
                    error_msg = str(e)
                    print(f"[Qwen] Failed to load with vLLM: {error_msg}")
                    if "No available memory for the cache blocks" in error_msg or "KV cache" in error_msg:
                        print(f"[Qwen] KV cache 内存不足。建议：")
                        print(f"[Qwen]   1. 减少 GPU 数量（如使用 2 个 GPU：--qwen_gpus 0,1）")
                        print(f"[Qwen]   2. 降低 gpu_memory_utilization（当前: {kwargs.get('gpu_memory_utilization', self.gpu_memory_utilization)}，建议: 0.7）")
                        print(f"[Qwen]   3. 或者使用 transformers 后端（较慢但更稳定）")
                    elif "Free memory on device" in error_msg:
                        print(f"[Qwen] GPU 内存不足。建议：")
                        print(f"[Qwen]   1. 清理 GPU 上的其他进程（如 XCOMET）")
                        print(f"[Qwen]   2. 降低 gpu_memory_utilization（当前: {kwargs.get('gpu_memory_utilization', self.gpu_memory_utilization)}，建议: 0.7）")
                        print(f"[Qwen]   3. 使用空闲的 GPU（如 --qwen_gpus 4）")
                        print(f"[Qwen]   4. 或者使用 transformers 后端（较慢但更稳定）")
                    elif "ValueError" in error_msg and ("invalid literal" in error_msg or "CUDA_VISIBLE_DEVICES" in error_msg):
                        print(f"[Qwen] 这可能是CUDA_VISIBLE_DEVICES格式问题，请检查环境变量格式")
                        print(f"[Qwen] 当前CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
                    print(f"[Qwen] Falling back to transformers backend")
                    self.use_vllm = False
        elif self.use_vllm and not VLLM_AVAILABLE:
            print("[Qwen] vllm not available, using transformers backend")
            self.use_vllm = False
        
        if not self.use_vllm:
            # 使用transformers进行推理（只用单卡，不再自动多卡）
            print(f"[Qwen] Loading model with transformers on {self.device}...")
            if self.device.startswith("cuda"):
                # 单卡 GPU：显式加载到当前 device，不用 device_map="auto"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=None,          # 不启用自动多卡切分
                    trust_remote_code=True,
                )
                # 移到指定 GPU（通常是 cuda 或 cuda:0）
                self.model.to(self.device)

                print(f"[Qwen] Model loaded with transformers on {self.device}")
            else:
                # CPU 模式
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                )
                self.model = self.model.to(self.device)
                print(f"[Qwen] Model loaded with transformers on {self.device}")

            self.model.eval()
    
    def generate_draft(
        self,
        prompts: Union[str, List[str]],
        mode: str = "draft",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        生成draft翻译，复用MT_Grpo中vllm的生成方式
        
        Args:
            prompts: 单个prompt字符串或prompt列表
            mode: 生成模式（固定为"draft"）
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus sampling参数
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本（字符串或字符串列表）
        """
        # 首先验证和规范化参数，避免后续错误
        raw_temperature = float(temperature)
        raw_top_p = float(top_p)
        if raw_temperature < 0:
            raw_temperature = 0.0
        if raw_top_p <= 0:
            raw_top_p = 0.0
        raw_top_p = min(raw_top_p, 1.0)

        # 根据温度/采样配置决定是否启用采样
        explicit_do_sample = kwargs.pop("do_sample", None)
        should_sample = explicit_do_sample if explicit_do_sample is not None else (raw_temperature > 0.0 and raw_top_p < 1.0)

        # 正规化温度和top_p，只有在启用采样时才使用
        temperature = max(min(raw_temperature, 2.0), 0.0)
        top_p = raw_top_p if raw_top_p > 0.0 else 1.0
        if not should_sample:
            temperature = 0.0
            top_p = 1.0
        elif temperature < 1e-4:
            # 防止过小温度导致数值不稳定
            temperature = 1e-4
        elif top_p < 1e-4:
            top_p = 1e-4
        max_tokens = max(int(max_tokens), 1)
        
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        if self.use_vllm and self.vllm_engine:
            # 使用vllm生成，参考 test_time/vllm_infer.py 的设置
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                skip_special_tokens=False,  # test_time 中设置，保持与工作版本一致
                **kwargs
            )
            outputs = self.vllm_engine.generate(prompts, sampling_params)
            results = [output.outputs[0].text for output in outputs]
        else:
            # 使用transformers生成
            results = []
            # CPU模式下，限制max_tokens以避免过长时间等待
            is_cpu = self.device == "cpu"
            if is_cpu and max_tokens > 512:
                print(f"[WARNING] CPU模式：将max_tokens从{max_tokens}减少到512以加速生成")
                max_tokens = 512
            
            for idx, prompt in enumerate(prompts):
                print(f"[生成进度] {idx+1}/{len(prompts)}")
                # 构建chat格式
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                # Generate
                is_cpu = self.device == "cpu"
                device_info = "CPU模式" if is_cpu else f"GPU模式 ({self.device})"
                if is_cpu:
                    print(f"[生成中] 这可能需要几分钟（{device_info}）...")
                else:
                    print(f"[生成中] 使用{device_info}...")
                
                # 确定是否使用采样（CPU 模式下强制使用 greedy decoding）
                use_sampling = should_sample and not is_cpu
                
                # 预先清理 kwargs 中的采样参数（避免警告）
                sampling_params = {"temperature", "top_p", "top_k", "do_sample"}
                cleaned_kwargs = {}
                for k, v in kwargs.items():
                    if k not in sampling_params:
                        cleaned_kwargs[k] = v
                    # 采样参数会在后面根据 use_sampling 决定是否添加
                    
                with torch.no_grad():
                    generate_kwargs = {
                        **inputs,
                        "max_new_tokens": max_tokens,
                        "do_sample": use_sampling,
                    }

                    if use_sampling:
                        # 使用采样：传递 temperature 和 top_p
                        generate_kwargs.update({
                            "temperature": temperature,
                            "top_p": top_p,
                        })
                        # 如果 kwargs 中有采样参数，也添加（允许覆盖）
                        for k in ["temperature", "top_p", "top_k"]:
                            if k in kwargs:
                                generate_kwargs[k] = kwargs[k]
                    else:
                        # 使用 greedy decoding：不传递任何采样参数，避免警告
                        generate_kwargs.update({
                            "num_beams": 1,
                        })
                    
                    # 添加其他非采样参数
                    generate_kwargs.update(cleaned_kwargs)
                    
                    # 尝试生成，捕获各种可能的错误
                    try:
                        outputs = self.model.generate(**generate_kwargs)
                        # 同步CUDA操作，确保错误能及时捕获
                        if not is_cpu:
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass  # 如果CUDA上下文已损坏，忽略同步错误
                    except (RuntimeError, ValueError, Exception) as e:
                        error_msg = str(e).lower()
                        # 检查是否是数值问题（inf/nan/probability）或CUDA错误
                        if any(keyword in error_msg for keyword in ["inf", "nan", "probability", "cuda error", "device-side assert", "accelerator"]):
                            print(f"[Warning] 生成时遇到数值/CUDA问题，切换到greedy decoding重试...")
                            # 注意：如果CUDA上下文已损坏，不要调用empty_cache，会导致更多错误
                            # 直接尝试重新生成
                            
                            # 使用greedy decoding重试（更稳定）
                            generate_kwargs_retry = {
                                **inputs,
                                "max_new_tokens": max_tokens,
                                "do_sample": False,
                                "num_beams": 1,
                            }
                            try:
                                outputs = self.model.generate(**generate_kwargs_retry)
                                # 不调用synchronize，因为CUDA上下文可能已损坏
                            except Exception as e2:
                                print(f"[Error] 重试后仍然失败: {e2}")
                                # 返回空字符串而不是崩溃
                                generated_text = ""
                                results.append(generated_text)
                                print(f"[完成] 样本 {idx+1}/{len(prompts)} (生成失败，返回空字符串)")
                                continue
                        else:
                            # 其他错误直接抛出
                            raise
                    
                    # Decode（只有在没有continue的情况下才会执行到这里）
                    try:
                        generated_text = self.tokenizer.decode(
                            outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True
                        )
                    except Exception as e3:
                        print(f"[Error] 解码失败: {e3}")
                        generated_text = ""
                    
                    results.append(generated_text)
                    print(f"[完成] 样本 {idx+1}/{len(prompts)}")
        
        return results[0] if is_single else results
    
    def generate_from_source(
        self,
        source_texts: Union[str, List[str]],
        lang_pair: Union[str, List[str]] = "en-zh",
        mode: str = "draft",
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        从源文本直接生成翻译（自动构建prompt，使用rl模板）
        
        Args:
            source_texts: 源文本（字符串或列表）
            lang_pair: 语言对，如 "en-zh"（字符串或列表）
            mode: 生成模式（默认"draft"）
            **generation_kwargs: 传递给generate_draft的参数
        
        Returns:
            生成的翻译（字符串或列表）
        """
        from data.process_data import make_prefix, language_map
        
        is_single = isinstance(source_texts, str)
        if is_single:
            source_texts = [source_texts]
            lang_pair = [lang_pair]
        elif isinstance(lang_pair, str):
            lang_pair = [lang_pair] * len(source_texts)
        
        prompts = []
        for src, lg in zip(source_texts, lang_pair):
            source_lang, target_lang = lg.split('-')
            src_lang_name = language_map.get(source_lang, source_lang.capitalize())
            tgt_lang_name = language_map.get(target_lang, target_lang.capitalize())
            
            # 使用rl模板构建prompt（draft mode复用rl模板）
            example = {
                'lg': lg,
                'src_text': src,
            }
            prompt = make_prefix(example, template_type='draft', tokenizer=self.tokenizer)
            prompts.append(prompt)
        
        return self.generate_draft(prompts, mode=mode, **generation_kwargs)

