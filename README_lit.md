# SimultaneousTranslation

以同声传译为灵感，增强 LLM 的翻译质量：Qwen 生成初稿 → XCOMET 定位错误 → 基于错误 span 修复。支持 **baseline（整句）** 与 **extended（短句修复）** 两套流程，自动管理 GPU 映射与显存。

---

## 流程总览
1) **数据准备**：读取 `data/test/used` 下的 JSONL，缺省转换为 Parquet 缓存。  
2) **Draft 生成**：基于模板构造 prompt 调用 Qwen（vLLM/transformers）。  
3) **格式校验**：提取 `<translate>` 作为初稿，记录格式分。  
4) **XCOMET 评分**：句子级（baseline）或短句级（extended）打分并返回错误 span。  
5) **Repair 生成**：携带原文/参考 + 初稿 + 错误 span 进行二次生成。  
6) **统计与日志**：保存结果与汇总统计到 `xcomet_all_stats.txt`。

### Baseline vs Extended
- **baseline**：整句生成 → 评分 → 整句修复。  
- **extended**：整段初稿 → 同传式短句切分 → 短句评分与修复 → 合并终稿 → 终稿再评。

---

## 主要组件
- `main.py`：流水线主控（数据加载、缓存、GPU 映射、分阶段执行、日志）。  
- `data/process_data.py`：JSONL 读取、prompt 生成、Parquet 缓存；`data/test/unify_test_data.py` 统一多源数据。  
- `qwen_generator.py`：Qwen 推理封装，vLLM 优先，自动选择 dtype / GPU，显存不足时降占比或回退 CPU/transformers。  
- `xcomet_loader.py`：XCOMET-XL 加载与评分，支持 GPU/CPU、错误 span 输出。  
- `utils.py`：`<translate>` 提取、错误 span 格式化、同传式切块（硬/软边界）与 wtpsplit 分句。  
- `experiments/`：XCOMET span 策略对比脚本与结果。  
- `utils/*.py`：结果清洗、修复早期标签、补跑 XCOMET、跨目录拷贝 repair 文本。  
- `runs/*.sh`：批量运行示例。  

### 目录结构
```shell
SimultaneousTranslation/
├─ main.py
├─ qwen_generator.py
├─ xcomet_loader.py
├─ utils.py
├─ download_xcomet.py
├─ data/
│  ├─ process_data.py
│  ├─ test/
│  │  ├─ used/               # 统一后的测试集（jsonl/parquet）
│  │  ├─ unify_test_data.py   # 将原始多源数据转成统一格式
│  │  ├─ metrics.py, merge.py # 评测与汇总
│  │  └─ ...                  # 原始数据源
├─ experiments/
│  ├─ xcomet_1.0/
│  └─ xcomet_2.0/
├─ runs/
├─ utils/                     # 各种清洗/修复脚本
└─ results/                   # 已跑结果与指标
```

---

## 安装依赖
1. 创建环境
```bash
conda create -n st python=3.10 -y
conda activate st
pip install --upgrade pip
```
2. 安装 PyTorch（示例：CUDA 11.8）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. 其他依赖
```bash
pip install -r requirements.txt   # 如不装 vLLM，可手动跳过
```

### 下载 XCOMET-XL
```bash
python download_xcomet.py --output_dir ~/models/XCOMET-XL
# 或直接设置
export WORD_QE_CKPT=~/models/XCOMET-XL/checkpoints/model.ckpt
```

---

## 数据格式与准备
- 统一 JSONL：
```json
{"data_source": "culturemt", "lg": "en-zh", "src_text": "...", "tgt_text": "..."}
```
- 已整理数据位于 `data/test/used/`（CultureMT、CommonMT、DRT、FLORES101、RTT 等多语向）。  
- 若需从原始数据生成：运行 `data/test/unify_test_data.py`，产物放回 `data/test/used/`。  
- 首次运行会生成同名 `.parquet` 缓存，后续自动加载。  

---

## 快速开始

### 基线模式（整句 → 评分 → Repair）
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --data_dir data/test/used \
  --test_files wmt23_zh-en.jsonl \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --xcomet_gpus 0 \
  --qwen_gpus 1 \
  --pipeline_mode baseline \
  --num_samples 5 \
  --output_file results/test_baseline_demo.json
```

### 扩展模式（整段初稿 → 短句修复 → 终稿再评）
```bash
CUDA_VISIBLE_DEVICES=1,2,3 python main.py \
  --data_dir data/test/used \
  --test_files wmt23_zh-en.jsonl \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --xcomet_gpus 1 \
  --qwen_gpus 2 \
  --pipeline_mode extended \
  --num_samples 3 \
  --output_file results/test_extended_demo.json
```

### GPU/CPU 选择规则（优先级）
1. `--xcomet_cpu` / `--qwen_cpu`。  
2. `--xcomet_gpus` / `--qwen_gpus`（物理编号，内部映射到逻辑 id）。  
3. 仅设 `CUDA_VISIBLE_DEVICES`：两者共享该列表。  
4. 未指定则回退 CPU（极慢，仅调试）。  

### 关键参数
- `--pipeline_mode baseline|extended`  
- `--max_tokens_draft / --max_tokens_repair`（默认 2048 / 4096）  
- `--batch_size`（生成批大小）、`--xcomet_batch_size`  
- `--gpu_memory_utilization`（vLLM 显存占比，默认 0.85）  
- `--vllm_max_num_seqs` 控制 warmup 显存占用  
- `--num_samples` 只跑前 N 条  

---

## 详细流程

### 基线模式
1. **初稿生成**：批量调用 Qwen（draft 模式），保存 `draft_generated_text`。  
2. **格式检查**：提取 `<translate>` → `draft_translation`，记录 `draft_format_score`。  
3. **XCOMET 评分**：对格式正确的初稿用参考翻译打分，取 `score` 与 `error_spans`。  
4. **Repair**：若存在错误 spans，构造 repair prompt（原文+初稿+spans），生成 `repair_generated_text`，提取终稿。  
5. **终稿评分**：对 `final_translation` 再跑 XCOMET。  

### 扩展模式
1. **完整初稿生成**：整段原文生成一次初稿。  
2. **格式检查**：提取 `<translate>`。  
3. **短句切分**：`split_into_segments`（硬边界：句末标点/空行；软边界：逗号等 + 长度 + 连接词；避免括号/引号内部切分）。  
4. **短句评分**：每个初稿短句用“完整原文 + 完整参考”跑 XCOMET，聚合短句得分与 spans。  
5. **短句修复**：仅对有错误的短句生成润色；提取 `<translate>`，失败回退初稿短句。  
6. **合并终稿**：若所有短句存在，拼接得到 `final_translation`，否则终稿缺失。  
7. **终稿评分**：对终稿再次跑 XCOMET。  

---

## 短句切分策略（同传式）
- **硬边界**：`。！？!?`，空行。  
- **软边界**：`，,、；;：:…‥`，当长度超阈或遇到连接词（例如“但是/然而/所以/then/however”）更易切分。  
- **避免切分**：括号/引号内部，连接结构中间。  
- **长度控制**：默认理想 100 字符（extended 默认），绝对最大 150，过长强切；末尾碎片会与前段合并。  
- 另有 `split_into_segments_wtpsplit` 使用 SaT-3l-sm（ONNX）分句。  

---

## 数据与输出格式
- 输入 JSONL 字段：`data_source`, `lg`, `src_text`, `tgt_text`。  
- 主脚本输出 JSON（列表）包含：  
  - `draft_prompt`, `draft_generated_text`, `draft_translation`, `draft_format_score`  
  - `xcomet_draft`（score, error_spans, system_score）  
  - `repair_prompt`/`repair_generated_text`（baseline 或短句级列表）、`repair_format_score`  
  - `final_translation`, `xcomet_final`  
- 统计同时写入 `xcomet_all_stats.txt`。  

---

## 评价与实验
- `data/test/metrics.py`、`utils/metrics-part*.py`：对 draft/final 计算 BLEU、COMET-DA、COMETKiwi，输出 `_each.csv` 与 `_total.csv`。  
- `results/`：包含 Qwen2.5/3/3-8B、多个数据集的 baseline/extended 结果与合并表。  
- **XCOMET span 策略对比**  
  - `experiments/xcomet_1.0`（494 短句）：S_seg+MT_seg+Ref (2.2) 准确率 **53.4%**，无参考 37.7%；S_full+MT_seg+Ref (3.2) 31.0%，无参考 15.0%。  
  - `experiments/xcomet_2.0`（100 构造案例，IoU≥0.5）：S_full+MT_full+Ref (1.2) 准确率 **39.5%**，无参考 10.5%。  
  - 结论：携带参考显著提升错误 span 覆盖，extended 流程修复时应包含完整 ref。  

---

## 常用脚本
- `download_xcomet.py`：下载 XCOMET-XL。  
- `runs/run_all-part*.sh`, `runs/run_wmt_enzh_zhen.sh`：批量跑多数据集示例。  
- `utils/clean_results.py`：清理字段、重命名 extended→baseline。  
- `utils/fix_repair_translations.py` / `fix_repair_translations-2.py`：修正早期 `<translate>` 闭合错误并重评 XCOMET。  
- `utils/copy_repair_texts.py`：跨目录补写 `repair_generated_text`。  
- `utils/fix_xcomet_final.py`：终稿缺失时补跑 XCOMET。  

---

## 注意事项 / 排障
- vLLM 对 `CUDA_VISIBLE_DEVICES` 格式敏感，尾逗号会导致失败；内部会尝试自动修复。  
- 显存不足：调低 `gpu_memory_utilization` / `batch_size`，或切换 transformers/CPU；XCOMET 可 `--xcomet_cpu`。  
- XCOMET 默认单 GPU 推理，避免多进程重复加载。  
- CPU 模式极慢，仅建议小样本调试。  
- wtpsplit 依赖 onnxruntime，已在 `wtpsplit[onnx-gpu]` 里声明。  

---

## 最小调试示例（CPU）
```bash
python main.py \
  --data_dir data/test/used \
  --test_files commonmt_lexical_ambiguity_zh-en.jsonl \
  --xcomet_ckpt /ltstorage/home/4xin/models/XCOMET-XL/checkpoints/model.ckpt \
  --xcomet_cpu --qwen_cpu \
  --num_samples 1 \
  --output_file /tmp/debug.json
```
输出 JSON 含 prompt、初稿/终稿翻译及 XCOMET spans，可直接喂给评测脚本。  
