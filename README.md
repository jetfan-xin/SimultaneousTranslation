# SimultaneousTranslation

基于MT_Grpo实现的同声传译项目，支持使用Qwen2.5-7B生成draft翻译，并使用XCOMET进行评分。

## 功能特性

1. **数据集获取**：从MT_Grpo的data/train文件夹加载训练数据
2. **数据转换和Prompt生成**：将JSONL数据转换为parquet格式，同时根据draft mode模板生成完整的prompt文本（包含instruction、格式说明和用户输入）
3. **Parquet缓存**：自动保存处理后的数据到 `data/train/parquet` 目录（文件名与原始JSONL相同，扩展名改为.parquet），下次运行时会自动加载，避免重复转换
4. **XCOMET模型加载**：在外部加载XCOMET-XL模型，复用MT_Grpo中的实现方式，输出句子得分与错误片段（含严重等级）
   - 使用单GPU推理，避免多进程重复加载
   - 支持CPU模式（`--xcomet_cpu`）
5. **Qwen2.5-7B生成**：调用Qwen2.5-7B模型，使用draft mode，复用MT_Grpo中vllm的使用方式
   - 支持vLLM和transformers两种后端
   - 自动检测GPU类型并选择合适的dtype
   - 配置与 `test_time/vllm_infer.py` 完全对齐
6. **自动GPU选择**：自动检测所有可见GPU的内存状态，选择最充足的GPU
   - 使用 `nvidia-smi` 获取准确的内存信息（包括其他进程占用的内存）
   - 自动避开已被XCOMET占用的GPU
   - 如果所有GPU内存都不足，自动降低 `gpu_memory_utilization`
7. **扩展模式（可选）**：支持 `--pipeline_mode extended`，不切分原文，先对完整原文生成初稿，然后切分初稿翻译为短句，针对短句调用XCOMET获取错误片段（使用完整原文和完整参考翻译），并逐句修复后合并为终稿
8. **阶段式处理**：将整个流程分为多个阶段，每个阶段独立运行，便于调试和监控

## 安装依赖

#### 1. 创建conda环境

```bash
conda create -n ST python=3.10 -y
conda activate ST
```

#### 2. 更新pip

```bash
pip install --upgrade pip
```

#### 3. 安装PyTorch

**如果有GPU（推荐）：**

```bash
# CUDA 11.8或更高版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**如果只有CPU：**

```bash
pip install torch torchvision torchaudio
```

#### 4. 安装其他依赖

```bash
cd /ltstorage/home/4xin/SimultaneousTranslation
pip install -r requirements.txt
```

注意：`requirements.txt`中包含vllm，如果GPU不可用或不想安装vllm，可以手动安装其他依赖：

```bash
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install pandas>=1.5.0
pip install tqdm>=4.65.0
pip install huggingface-hub>=0.17.0
pip install unbabel-comet>=2.0.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0
```

#### 5. 安装vllm（可选，推荐用于GPU加速）

```bash
pip install vllm
```

### 验证安装

```bash
conda activate ST
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import comet; print('COMET已安装')"
python -c "import datasets; print('Datasets已安装')"
```

如果安装了vllm，验证：

```bash
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
```

### 环境信息

- **环境名称**: ST
- **Python版本**: 3.10
- **主要依赖**:
  - PyTorch >= 2.0.0
  - Transformers >= 4.35.0
  - Datasets >= 2.14.0
  - unbabel-comet >= 2.0.0 (用于XCOMET)
  - vllm >= 0.2.0 (可选，用于加速推理)

### 常见问题

#### 1. vllm安装失败

如果vllm安装失败，不用担心，程序会自动回退到transformers后端。vllm主要用于加速推理，不是必需的。

#### 2. CUDA版本不匹配

如果PyTorch的CUDA版本与系统CUDA版本不匹配，可能会导致问题。检查CUDA版本：

```bash
nvidia-smi
```

然后安装对应版本的PyTorch。

#### 3. 内存不足

如果GPU内存不足（<16GB），可以考虑：
- 不使用vllm（使用transformers后端）
- 减小batch_size
- 使用量化模型

#### 4. 激活环境

每次使用前需要激活环境：

```bash
conda activate ST
```

或者添加到.bashrc：

```bash
echo "conda activate ST" >> ~/.bashrc
```

## 使用步骤

### 1. 下载XCOMET模型（可选）

如果需要使用XCOMET进行评分，需要先下载模型：

```bash
python download_xcomet.py --output_dir ~/models/XCOMET-XL
```
（需要先登陆Huggingface，并在XCOMET-XL网页(https://huggingface.co/Unbabel/XCOMET-XL)上授权下载！！！）

或者设置环境变量：

```bash
export WORD_QE_CKPT=~/models/XCOMET-XL/checkpoints/model.ckpt
```

### 2. 运行主脚本

## 快速开始示例

### 基线模式（Baseline Mode）

基线模式按整句处理，包含4个阶段：生成初稿 → 格式检查 → XCOMET评分 → Repair修复。

#### CPU上运行：

```bash
python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_cpu \
    --qwen_cpu \
    --num_samples 5 \
    --output_file test_5_baseline_cpu.json
```

**输出结果示例**：
```
============================================================
步骤5: 保存结果
============================================================
[Output] Results saved to test_5_baseline_cpu.json
[Stats] Draft格式正确率: 5/5 (100.0%)
[Stats] Repair格式正确率: 5/5 (100.0%)
[Stats] XCOMET Draft scores - Mean: 0.8251, Min: 0.6145, Max: 0.9898
[Stats] Avg. error spans per draft sample: 2.80
[Stats] Avg. error spans per final sample: 2.80
[Stats] XCOMET Final scores - Mean: 0.8547, Min: 0.6833, Max: 0.9875
[Stats] 终稿改进初稿的样本数: 2/5 (40.0%)
```

**统计信息说明**：
- **Draft格式正确率**：初稿中成功提取 `<translate>` 标签的比例
- **Repair格式正确率**：修复后终稿中成功提取 `<translate>` 标签的比例
- **XCOMET Draft scores**：初稿翻译的XCOMET质量得分（范围通常0-1，越高越好）
- **Avg. error spans per draft sample**：每个初稿样本平均的错误片段数量
- **Avg. error spans per final sample**：每个终稿样本平均的错误片段数量
- **XCOMET Final scores**：终稿翻译的XCOMET质量得分
- **终稿改进初稿的样本数**：终稿XCOMET得分高于初稿的样本数量（说明repair有效）

#### GPU上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --num_samples 5 \
    --output_file test_5_baseline_gpu.json
```

**说明**：
- 只设置 `CUDA_VISIBLE_DEVICES` 时，Qwen 和 XCOMET 都会使用这些 GPU（共享）
- 默认使用 GPU 模式（如果 GPU 可用）
- 可以顺利跑通少量数据

**输出结果示例**：
```
============================================================
步骤5: 保存结果
============================================================
[Output] Results saved to test_5_baseline_gpu.json
[Stats] Draft格式正确率: 5/5 (100.0%)
[Stats] Repair格式正确率: 5/5 (100.0%)
[Stats] XCOMET Draft scores - Mean: 0.8251, Min: 0.6145, Max: 0.9898
[Stats] Avg. error spans per draft sample: 2.80
[Stats] Avg. error spans per final sample: 2.80
[Stats] XCOMET Final scores - Mean: 0.8547, Min: 0.6833, Max: 0.9875
[Stats] 终稿改进初稿的样本数: 2/5 (40.0%)
```

---

### 扩展模式（Extended Mode）

扩展模式不切分原文，先对完整原文生成初稿，然后切分初稿翻译为短句进行处理，包含7个阶段：生成完整初稿 → 格式检查 → 切分初稿翻译 → 短句XCOMET评分 → 润色短句 → 汇总终稿 → 终稿XCOMET评分。

#### GPU上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --pipeline_mode extended \
    --num_samples 3 \
    --output_file test_5_extended_gpu.json
```

**说明**：
- 扩展模式不切分原文，先对完整原文生成初稿
- 然后切分初稿翻译为短句
- 短句级别的XCOMET评分使用完整原文和完整参考翻译（三元组：完整原文、初稿短句、完整参考翻译）
- Repair prompt包含完整原文和完整参考翻译，提供更多上下文
- 只对有错误的短句进行润色
- 如果任何初稿短句缺失，则没有终稿

**输出结果示例**：
```
============================================================
步骤5: 保存结果
============================================================
[Output] Results saved to test_5_extended_gpu.json
[Stats] Draft格式正确率: 0/3 (0.0%)
[Stats] Repair格式正确率: 3/3 (100.0%)
[Stats] XCOMET Draft scores - Mean: 0.8943, Min: 0.8078, Max: 0.9890
[Stats] Avg. error spans per draft sample: 3.00
[Stats] Avg. error spans per final sample: 1.67
[Stats] XCOMET Final scores - Mean: 0.7252, Min: 0.4366, Max: 0.9898
[Stats] 终稿改进初稿的样本数: 1/3 (33.3%)
```

**统计信息说明**：
- **Draft格式正确率**：初稿短句中成功提取 `<translate>` 标签的比例（扩展模式下可能较低，因为短句较多）
- **Repair格式正确率**：润色短句中成功提取 `<translate>` 标签的比例
- **XCOMET Draft scores**：初稿短句的平均XCOMET得分（不使用参考翻译）
- **Avg. error spans per draft sample**：每个初稿样本平均的错误片段数量
- **Avg. error spans per final sample**：每个终稿样本平均的错误片段数量（通常比初稿少，说明润色有效）
- **XCOMET Final scores**：终稿翻译的XCOMET质量得分（使用参考翻译）
- **终稿改进初稿的样本数**：终稿XCOMET得分高于初稿的样本数量

---

### GPU选择规则

**默认行为**：
- **默认使用 CPU 模式**（不设置任何GPU参数时）
- 如果只设置了 `CUDA_VISIBLE_DEVICES` 环境变量，且没有分别设置 `--qwen_gpus` 和 `--xcomet_gpus`，则 Qwen 和 XCOMET 都会使用 `CUDA_VISIBLE_DEVICES` 指定的 GPU（共享）
- 如果分别设置了 `--qwen_gpus` 和 `--xcomet_gpus`，则各自使用指定的 GPU

**优先级（从高到低）**：
1. `--qwen_cpu` / `--xcomet_cpu`：强制CPU模式
2. `--qwen_gpus` / `--xcomet_gpus`：分别指定GPU
3. `CUDA_VISIBLE_DEVICES` 环境变量：共享GPU（如果未分别指定）
4. 默认：CPU模式

**示例**：

1. **共享GPU**（只设置环境变量）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --num_samples 5 \
    --output_file test_shared.json
```
→ Qwen 和 XCOMET 都使用 GPU 0,1,2,4

2. **分别指定GPU**：
```bash
python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_gpus 0,1 \
    --qwen_gpus 2,3,4 \
    --num_samples 5 \
    --output_file test_separate.json
```
→ XCOMET 使用 GPU 0,1，Qwen 使用 GPU 2,3,4

3. **混合模式**（XCOMET用CPU，Qwen用GPU）：
```bash
python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_cpu \
    --qwen_gpus 0,1,2,3 \
    --num_samples 5 \
    --output_file test_mixed.json
```
→ XCOMET 使用 CPU，Qwen 使用 GPU 0,1,2,3

### 其他用法示例

#### 手动指定GPU（分别指定）：

```bash
# XCOMET使用GPU 0，Qwen使用GPU 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_gpus 0 \
    --qwen_gpus 1 \
    --pipeline_mode baseline \
    --output_file results.json \
    --num_samples 10
```

#### 如果遇到XCOMET的CUDA错误，可以使用CPU模式：

```bash
python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_cpu \
    --qwen_gpus 0 \
    --pipeline_mode baseline \
    --output_file results.json \
    --num_samples 10
```

#### 如果GPU内存不足，降低gpu_memory_utilization：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_dir data \
    --train_files train/json/train_enzh_6565.jsonl \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_gpus 0 \
    --qwen_gpus 1 \
    --gpu_memory_utilization 0.6 \
    --pipeline_mode baseline \
    --output_file results.json \
    --num_samples 10
```

#### 从results.json读取数据进行refinement：

如果需要从已有的results.json文件读取draft结果并进行refinement：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --input_results_file results.json \
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
    --xcomet_gpus 0,1 \
    --qwen_gpus 2,3 \
    --gpu_memory_utilization 0.7 \
    --output_file results_refined.json \
    --use_vllm
```

#### 完整参数说明：

```bash
# 指定使用GPU设备（根据实际需求调整）
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py \
    --data_dir data \                          # 数据目录
    --train_files train/json/xxx.jsonl \      # 训练文件（相对data_dir，可指定多个）
    --tokenizer_path Qwen/Qwen2.5-7B-Instruct \  # Tokenizer路径（默认：Qwen/Qwen2.5-7B-Instruct）
    --qwen_model_path Qwen/Qwen2.5-7B-Instruct \ # Qwen模型路径（默认：Qwen/Qwen2.5-7B-Instruct）
    --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \  # XCOMET checkpoint（必需）
    --use_vllm \                                # 使用vllm（推荐，更快，默认：True）
    --max_tokens 2048 \                         # 最大生成token数（默认：2048，与test_time对齐）
    --temperature 0.2 \                         # 采样温度（默认：0.2，与test_time对齐）
    --top_p 0.95 \                              # nucleus sampling（默认：0.95，与test_time对齐）
    --batch_size 16 \                           # 批处理大小（默认：16，与test_time对齐）
    --gpu_memory_utilization 0.85 \            # GPU内存使用率（默认：0.85，与test_time对齐）
    --num_samples 100 \                         # 处理的样本数量（默认：全部）
    --pipeline_mode baseline \                  # 流程模式：baseline（整句）或extended（短句修复）
    --xcomet_gpus 0 \                           # XCOMET使用的GPU（可选，默认CPU）
    --qwen_gpus 1 \                             # Qwen使用的GPU（可选，默认CPU）
    --xcomet_cpu \                              # 强制XCOMET使用CPU（可选）
    --qwen_cpu \                                # 强制Qwen使用CPU（可选）
    --input_results_file results.json \         # 输入results.json文件（用于refinement）
    --output_file results.json                  # 输出文件（必需）
    # 注意：parquet文件会自动保存到 data/train/parquet 目录，无需手动指定
```

**重要参数说明**：

1. **GPU选择**（优先级从高到低）：
   - **默认**：CPU模式（不设置任何GPU参数时）
   - `CUDA_VISIBLE_DEVICES=0,1,2,4`：如果只设置环境变量，且没有分别设置 `--qwen_gpus` 和 `--xcomet_gpus`，则 Qwen 和 XCOMET 都会使用这些 GPU（共享）
   - `--qwen_gpus` / `--xcomet_gpus`：分别指定 Qwen 和 XCOMET 使用的 GPU
   - `--qwen_cpu` / `--xcomet_cpu`：强制使用 CPU 模式
   - XCOMET 使用单GPU进行推理（避免多进程重复加载）
   - Qwen 使用单GPU（tensor_parallel_size=1，与test_time对齐）

2. **默认参数（与test_time/vllm_infer.py对齐）**：
   - `max_tokens=2048`
   - `temperature=0.2`
   - `top_p=0.95`
   - `batch_size=16`
   - `gpu_memory_utilization=0.85`
   - `tensor_parallel_size=1`（单GPU）
   - `max_model_len=16384`
   - `enforce_eager=True`
   - `disable_custom_all_reduce=True`

3. **流程模式**：
   - `baseline`：按整句处理，生成初稿后直接进行XCOMET评分，然后进行repair
   - `extended`：不切分原文，先对完整原文生成初稿，然后切分初稿翻译为短句，逐句使用XCOMET获取错误span（使用完整原文和完整参考翻译），逐句修复后合并为终稿

4. **自动功能**：
   - 默认使用CPU模式（不设置GPU参数时）
   - 如果设置了GPU参数，自动检测GPU类型并选择合适的dtype（A6000使用float16，A100使用bfloat16）
   - 如果设置了GPU参数，自动检测GPU内存并选择最充足的GPU（Qwen）
   - 自动处理GPU内存不足的情况（降低utilization或回退到CPU）

## 详细流程说明

### 基线模式（Baseline Mode）

基线模式按整句处理，包含4个阶段：

#### 阶段1：批量生成所有数据的初稿
- **输入**：原始数据（`src_text`、`tgt_text`、`lang_pair`）
- **处理**：
  1. 为每个样本生成 draft prompt（使用 `draft` 模板）
  2. 批量调用 Qwen2.5-7B 生成初稿翻译
  3. 保存生成的原始文本到 `draft_generated_text`
- **输出**：`draft_generated_text`（包含 `<think>` 和 `<translate>` 标签）

#### 阶段2：对所有初稿进行格式检查
- **处理**：
  1. 对每个 `draft_generated_text` 进行格式检查
  2. 提取 `<translate>` 标签中的内容作为 `draft_translation`
  3. 计算格式得分 `draft_format_score`（1=正确，0=错误）
- **输出**：
  - `draft_translation`：提取的初稿翻译
  - `draft_format_score`：格式得分

#### 阶段3：对所有初稿翻译进行XCOMET评分
- **处理**：
  1. 为每个格式正确的初稿构建三元组（`src`、`mt`、`ref`）
  2. 批量调用 XCOMET 进行评分
  3. 获取句子得分和错误片段（error spans）
- **输出**：
  - `xcomet_draft`：包含 `score`、`error_spans`、`system_score`

#### 阶段4：批量生成所有数据的终稿（repair）
- **处理**：
  1. 识别需要repair的样本（有错误片段的初稿）
  2. 为每个需要repair的样本生成 repair prompt（包含原文、初稿、错误片段）
  3. 批量调用 Qwen 生成修复后的翻译
  4. 提取修复后的翻译，失败则回退到初稿
- **输出**：
  - `repair_generated_text`：修复生成的原始文本
  - `final_translation`：最终翻译（修复后的翻译或初稿）
  - `repair_format_score`：修复格式得分

**特点**：
- 整句级别的处理
- 使用参考翻译进行XCOMET评分
- 只对有错误的初稿进行repair

---

### 扩展模式（Extended Mode）

扩展模式不切分原文，先对完整原文生成初稿，然后切分初稿翻译为短句进行处理，包含7个阶段：

#### 阶段1：批量生成完整原文的初稿翻译
- **输入**：原始数据中的完整 `src_text`（不切分）
- **处理**：
  1. 为每个完整原文生成 draft prompt（使用 `draft` 模板）
  2. 批量调用 Qwen2.5-7B 生成完整原文的初稿翻译
  3. 保存生成的原始文本到 `draft_generated_text`
- **输出**：
  - `draft_generated_text`：每个样本的完整初稿生成文本（包含 `<think>` 和 `<translate>` 标签）

**关键特性**：
- **不切分原文**：直接对完整原文生成初稿，保持原文的完整上下文
- 整句级别的初稿生成

#### 阶段2：对所有初稿进行格式检查，提取<translate>标签中的内容
- **处理**：
  1. 对每个 `draft_generated_text` 进行格式检查
  2. 提取 `<translate>` 标签中的内容作为完整初稿翻译 `draft_translation`
  3. 计算格式得分 `draft_format_score`（1=正确，0=错误）
- **输出**：
  - `draft_translation`：提取的完整初稿翻译
  - `draft_format_score`：格式得分

#### 阶段3：把完整的初稿翻译切分为初稿短句
- **输入**：完整初稿翻译 `draft_translation`
- **处理**：
  1. 使用 `split_into_segments` 函数对完整初稿翻译进行**同传式切块**
  2. **硬边界**（强制切分）：
     - 句末终结符：`。！？!?`（保留在短句中）
     - 空行分隔
  3. **软边界**（基于长度和结构的启发式切分）：
     - 软标点：`，,、；;：:…‥`
     - 当短句超过理想长度（默认40字符）时，在软标点处切分
     - 如果软标点后出现连接词（如"但是"、"however"等），更容易触发切分
  4. **避免切分的位置**：
     - 括号、引号等成对符号内部
     - 连接词结构中间（如"因为...所以..."）
  5. 保留标点符号在短句中
- **输出**：
  - `draft_segments`：初稿短句列表
  - 初始化 `draft_segment_results` 等字段

**同传式切块特点**：
- 每个短句是一个完整但不冗长的意义单元
- 不强行拆开紧密依赖前文的结构
- 符合同传的习惯：一口气说得出口，但又不至于太长
- 默认理想长度40字符，绝对最大长度80字符（超过则强制切分）

#### 阶段4：对所有初稿短句进行XCOMET评分
- **处理**：
  1. 为每个初稿短句构建三元组（`src`、`mt`、`ref`）：
     - `src`：**完整原文**（`src_text`）
     - `mt`：初稿短句（`draft_segment`）
     - `ref`：**完整参考翻译**（`tgt_text`）
  2. 批量调用 XCOMET 进行评分（使用完整原文和完整参考翻译）
  3. 获取每个短句的得分和错误片段
  4. 汇总所有短句的评分（平均得分和合并的错误片段）
- **输出**：
  - `draft_segment_results`：每个短句的XCOMET评分结果
  - `xcomet_draft`：整体初稿的汇总评分

**关键特性**：
- **使用完整原文和完整参考翻译**：XCOMET评分时使用完整原文（而非短句原文）和完整参考翻译，提供更多上下文信息
- 短句级别的错误检测

#### 阶段5：批量生成所有初稿短句的润色短句
- **处理**：
  1. 为每个有错误的初稿短句生成 repair prompt
  2. Repair prompt 包含：
     - `User: {src_text}` - **完整原文**
     - `Draft Translation Segment: {draft_translation_segment}` - 初稿短句
     - `Draft Source Segment: {ref_text}` - **完整参考翻译**
     - `Error Evaluation: {error_spans_json}` - 错误片段（JSON格式）
  3. 批量调用 Qwen 生成润色后的短句
  4. 提取润色短句的 `<translate>` 标签，失败则回退到初稿短句
- **输出**：
  - `repair_segment_outputs`：每个短句的润色生成文本
  - `final_segments`：每个短句的最终翻译
  - `repair_segment_format_scores`：每个短句的格式得分

**Repair Prompt 格式**：
```
A conversation between User and Assistant. The User asks for polishing a draft translation segment from {src_lang} to {tgt_lang}. The draft translation segment is part of a translation for a source text segment. The Assistant needs to polish this draft translation segment based on the source text segment, the draft translation segment, the original source text segment that the draft was translated from, and error evaluation. The Assistant first thinks about the reasoning process in the mind and then provides the user with the polished translation segment. The reasoning process and polished translation segment are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think> <translate> polished translation segment here </translate>. 

User: {src_text}
Draft Translation Segment: {draft_translation_segment}
Draft Source Segment: {ref_text}
Error Evaluation: {error_spans_json}
Assistant:
```

**关键特性**：
- **使用完整原文和完整参考翻译**：Repair prompt 包含完整原文和完整参考翻译，为模型提供更多上下文信息，有助于生成更准确的润色结果

#### 阶段6：汇总终稿短句
- **处理**：
  1. 确保所有短句都有最终结果（使用润色短句或初稿短句）
  2. 检查是否有缺失的初稿短句
  3. 如果有缺失，则没有终稿（`final_translation = None`）
  4. 否则，合并所有短句为终稿（使用空格连接）
- **输出**：
  - `final_translation`：合并后的终稿翻译（如果有所有短句）
  - `repair_format_score`：终稿格式得分

**合并规则**：
- 优先使用润色短句的 translate
- 如果润色短句格式无效，回退到初稿短句
- 如果任何初稿短句不存在，则没有终稿

#### 阶段7：对所有终稿翻译进行XCOMET评分
- **处理**：
  1. 为每个有终稿的样本构建三元组（`src`、`mt`、`ref`）
  2. 批量调用 XCOMET 进行评分（使用参考翻译）
  3. 获取终稿的得分和错误片段
- **输出**：
  - `xcomet_final`：终稿的XCOMET评分结果

**特点**：
- 不切分原文，先对完整原文生成初稿
- 切分初稿翻译为短句，进行短句级别的处理
- 初稿短句XCOMET评分使用完整原文和完整参考翻译（提供更多上下文）
- 终稿XCOMET评分使用参考翻译
- Repair prompt包含完整原文和完整参考翻译（提供更多上下文）
- 只对有错误的短句进行润色
- 如果任何初稿短句缺失，则没有终稿

---

### 两种模式的对比

| 特性 | 基线模式 | 扩展模式 |
|------|---------|---------|
| **原文处理** | 不切分 | 不切分 |
| **初稿生成** | 整句生成 | 整句生成（不切分原文） |
| **初稿处理** | 直接使用 | 切分为短句 |
| **XCOMET评分** | 使用参考翻译 | 初稿短句使用完整原文和完整参考翻译，终稿使用参考翻译 |
| **错误检测** | 整句级别 | 短句级别 |
| **修复生成** | 整句修复 | 短句修复（Repair prompt包含完整原文和完整参考翻译） |
| **终稿合并** | 直接使用修复结果 | 合并所有短句 |
| **失败处理** | 回退到初稿 | 回退到初稿短句，如果任何短句缺失则没有终稿 |

### 3. 单独使用各个模块

#### 使用XCOMET加载器

```python
from xcomet_loader import XCOMETLoader

loader = XCOMETLoader("~/models/XCOMET-XL/checkpoints/model.ckpt")
score = loader.score_single(
    src="Hello world",
    mt="你好世界",
    ref="你好世界"
)
print(f"XCOMET score: {score}")
```

#### 使用Qwen生成器

```python
from qwen_generator import QwenGenerator
generator = QwenGenerator("Qwen/Qwen2.5-7B-Instruct", use_vllm=True)
translation = generator.generate_from_source(
    source_texts="Hello world",
    lang_pair="en-zh",
    mode="draft"
)
print(f"Translation: {translation}")
```

## 短句切分策略（同传式切块）

扩展模式使用**同传式切块**策略，将长文本切分为适合翻译的短句。该策略模拟同声传译的工作方式，确保每个短句是完整的意义单元。

### 切分规则

#### 1. 硬边界（强制切分）
遇到以下情况，直接结束当前短句：
- **句末终结符**：`。！？!?`（保留在短句中）
- **空行**：原文中的空行分隔

#### 2. 软边界（启发式切分）
对于 `，,、；;：:…‥` 等软标点，基于以下条件判断是否切分：
- **长度限制**：当前短句超过理想长度（默认40字符）时，在软标点处切分
- **连接词检测**：如果软标点后出现连接词，更容易触发切分
  - 中文：但是、但、然而、不过、所以、因此、于是、然后、同时、之后、接着、相比之下
  - 英文：but, however, therefore, so, then, for example, for instance, in addition, on the other hand, meanwhile, furthermore, moreover
- **成对符号保护**：不在括号、引号等成对符号内部切分

#### 3. 避免切分的位置
以下位置不会切分，以保持意义完整性：
- **成对结构**：括号 `() [] {} （）【】`、引号 `"" '' 《》` 内部
- **连接词结构**：如"因为...所以..."、"not only...but also..."等
- **短句合并**：如果切分后产生过短的片段（<理想长度的30%），会与上一段合并

#### 4. 长度控制
- **理想长度**：默认40字符（可通过参数调整）
- **绝对最大长度**：默认80字符，超过则强制切分（可能在单词中间）

### 使用示例

```python
from utils import split_into_segments

text = "类比迁移，对于文本的翻译，我们能不能像同声传译一样，把长句划分为一段段可以单独输出的短句？比如这一句，如果太长，就可以在合适的逗号后面切开，而不是只在句号处切分。"

segments = split_into_segments(text)
# 结果：
# ['类比迁移，对于文本的翻译，我们能不能像同声传译一样，把长句划分为一段段可以单独输出的短句？',
#  '比如这一句，如果太长，就可以在合适的逗号后面切开，而不是只在句号处切分。']

# 自定义长度限制
segments = split_into_segments(text, max_len=30, hard_max_len=60)
```

### 参数说明

- `text` (str): 待切分的文本
- `max_len` (int, 默认40): 理想短句长度（字符数）
- `hard_max_len` (int, 默认80): 绝对最大长度，超过则强制切分

## 数据格式

输入数据格式（JSONL）：

```json
{"data_source": "train", "lg": "en-zh", "en": "Hello world", "zh": "你好世界"}
```

输出数据格式（JSON）：

```json
{
  "index": 0,
  "data_source": "train_en-zh",
  "lang_pair": "en-zh",
  "src_text": "Hello world",
  "tgt_text": "你好世界",
  "generated_text": "生成的翻译...",
  "prompt": "A conversation between User and Assistant...",
  "mode": "draft",
  "xcomet": {
    "score": 0.85,
    "error_spans": [
      {
        "text": "my food",
        "severity": "minor",
        "start": 13,
        "end": 21,
        "confidence": 0.41
      }
    ],
    "system_score": 0.90
  }
}
```

## 项目结构

```
SimultaneousTranslation/
├── data/                        # 数据目录
│   ├── process_data.py         # 数据预处理脚本（支持draft mode，复用base模板）
│   └── train/                  # 训练数据
│       ├── json/              # JSONL原始数据文件
│       │   ├── train_enzh_6565.jsonl
│       │   └── train_zhen_6565.jsonl
│       └── parquet/           # 转换后的parquet缓存文件（自动生成）
│           ├── train_enzh_6565.parquet
│           └── train_zhen_6565.parquet
├── main.py                     # 主执行脚本（必需）
├── qwen_generator.py           # Qwen2.5-7B生成器（必需）
├── xcomet_loader.py            # XCOMET模型加载器（必需）
├── utils.py                    # 工具函数（格式检查、短句切分等，必需）
├── download_xcomet.py          # XCOMET模型下载脚本（可选）
├── requirements.txt            # Python依赖列表（必需）
├── README.md                   # 项目文档（必需）
├── .gitignore                  # Git忽略文件（GitHub上传时排除不需要的文件）
└── *.json                      # 测试输出文件
```

### 核心文件说明

**必需文件**（项目运行必需）：
- `main.py` - 主执行脚本
- `qwen_generator.py` - Qwen模型生成器
- `xcomet_loader.py` - XCOMET评分器
- `utils.py` - 工具函数（格式检查、短句切分）
- `data/process_data.py` - 数据预处理
- `requirements.txt` - 依赖列表
- `README.md` - 项目文档

**可选文件**（辅助工具）：
- `download_xcomet.py` - 下载XCOMET模型

**自动生成的文件/目录**：
- `data/train/parquet/` - Parquet缓存文件（自动生成）
- `data/test/parquet/` - 测试Parquet缓存（自动生成）
- `__pycache__/` - Python缓存目录（可删除）
- `*.json` - 输出结果文件（可删除，用于测试）


## 注意事项

1. **GPU设备分配（重要）**：
   
   **默认行为**：
   - 默认使用 CPU 模式（不设置任何GPU参数时）
   - 如果只设置了 `CUDA_VISIBLE_DEVICES` 环境变量，Qwen 和 XCOMET 都会使用这些 GPU（共享）
   
   **共享GPU（推荐，简单）**：
   ```bash
   # Qwen 和 XCOMET 都使用 GPU 0,1,2,4
   CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py \
       --data_dir data \
       --train_files train/json/train_enzh_6565.jsonl \
       --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
       --output_file results.json
   ```
   
   **分别指定GPU**：
   ```bash
   # XCOMET使用GPU 0，Qwen使用GPU 1
   python main.py \
       --data_dir data \
       --train_files train/json/train_enzh_6565.jsonl \
       --xcomet_ckpt ~/models/XCOMET-XL/checkpoints/model.ckpt \
       --xcomet_gpus 0 \
       --qwen_gpus 1 \
       --output_file results.json
   ```
   
   **GPU选择优先级**：
   - `--qwen_cpu` / `--xcomet_cpu`：强制CPU模式（最高优先级）
   - `--qwen_gpus` / `--xcomet_gpus`：分别指定GPU
   - `CUDA_VISIBLE_DEVICES` 环境变量：共享GPU（如果未分别指定）
   - 默认：CPU模式

2. **GPU内存管理**：
   - `--gpu_memory_utilization`：vLLM的GPU内存使用率（0.0-1.0），默认0.85（与test_time对齐）
   - 如果遇到OOM，程序会自动降低utilization或选择其他GPU
   - Qwen2.5-7B模型较大，建议每个GPU至少16GB内存
   - 如果内存不足，可以：
     - 减小batch_size（默认16）
     - 降低gpu_memory_utilization（默认0.85）
     - 使用transformers后端（不使用vllm）
     - 为XCOMET和Qwen分配不同的GPU

3. **XCOMET单GPU推理**：
   - XCOMET 使用单GPU进行推理（`gpus=1`），避免分布式训练导致的多进程重复加载
   - 这不会影响性能，因为推理任务主要是数据并行，单GPU + 批处理足够
   - 如果指定了 `--xcomet_gpus 0,1`，程序会使用第一个GPU（0）

4. **vllm安装**：如果使用vllm进行加速推理，需要单独安装。如果vllm不可用，程序会自动回退到transformers后端。

5. **XCOMET模型**：
   - XCOMET模型较大（约几GB），首次下载需要一些时间。下载后可以重复使用。
   - 如果遇到CUDA错误（"device-side assert triggered"），可以使用 `--xcomet_cpu` 参数强制XCOMET使用CPU模式
   - CPU模式虽然较慢，但更稳定，适合批量处理
   - 使用 `--xcomet_gpus` 可以为XCOMET指定特定的GPU（如"0"）

6. **默认参数（与test_time对齐）**：
   - 所有默认参数已与 `test_time/vllm_infer.py` 对齐
   - `max_tokens=2048`, `temperature=0.2`, `top_p=0.95`, `batch_size=16`
   - `gpu_memory_utilization=0.85`, `tensor_parallel_size=1`
   - `max_model_len=16384`, `enforce_eager=True`, `disable_custom_all_reduce=True`

7. **数据路径**：确保data目录下包含train/json/文件夹和相应的JSONL文件。

8. **Parquet缓存**：程序会自动将处理后的数据保存为parquet格式到 `data/train/parquet` 目录，文件名与原始JSONL文件相同（扩展名改为.parquet）。下次运行时会自动加载，跳过数据转换步骤。

9. **Draft模式**：draft mode的prompt直接复用base模板，无需额外配置。

10. **CPU模式警告**：如果在CPU上运行，推理会非常慢。强烈建议使用GPU。如果必须使用CPU，请减少 `max_tokens` 和 `num_samples`。

11. **从results.json读取数据**：使用 `--input_results_file` 参数可以从已有的results.json文件读取draft结果，然后进行refinement处理。这对于只需要对已有结果进行refinement的场景很有用。

12. **GPU类型自动检测**：
    - 程序会自动检测GPU类型并选择合适的dtype
    - RTX A6000（compute capability 8.6）：自动使用 `float16`（避免bfloat16数值问题）
    - A100/H100（compute capability >= 9.0）：自动使用 `bfloat16`

## 故障排除

### 进程残留占用GPU

**症状**：
- `nvidia-smi` 显示GPU被占用，但找不到对应进程

**解决方案**：
```bash
# 查找占用GPU的进程
nvidia-smi

# 杀死所有vllm相关进程
pkill -f vllm

# 杀死所有python main.py进程
pkill -f "python main.py"

# 如果上述方法无效，使用kill -9强制杀死
kill -9 <PID>
```

## 代码复用说明

本项目尽可能复用MT_Grpo中的相关代码：

- **XCOMET加载**：复用`MT_Grpo/verl/comet_reward_batch_with_ray.py`中的模型加载和评分方式
- **vllm使用**：复用`MT_Grpo/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`中的vllm使用方式
- **数据下载**：复用`MT_Grpo/scripts/download_comet_ckpts.py`中的下载方式
- **配置对齐**：与`test_time/vllm_infer.py`的配置完全对齐

## 参考

本项目参考了MT_Grpo的实现方式：
- 数据预处理：`MT_Grpo/data/process_data.py`
- XCOMET评分：`MT_Grpo/verl/comet_reward_batch_with_ray.py`
- vllm调用：`MT_Grpo/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- 模型下载：`MT_Grpo/scripts/download_comet_ckpts.py`
- vLLM配置：`SimultaneousTranslation/test_time/vllm_infer.py`

