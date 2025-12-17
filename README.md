<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# SIMULTANEOUSTRANSLATION

<em>Two-stage MT (draft → repair) with Qwen + XCOMET for simultaneous translation research</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/jetfan-xin/SimultaneousTranslation?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/jetfan-xin/SimultaneousTranslation?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/jetfan-xin/SimultaneousTranslation?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/jetfan-xin/SimultaneousTranslation?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

SimultaneousTranslation is a research pipeline for building and evaluating two-stage machine translation (draft → repair) with large language models. It prepares heterogeneous test sets, generates translations with Qwen, diagnoses errors with XCOMET, and triggers targeted refinements before re-scoring. The code is tuned for GPU inference (optionally vLLM) and supports both full-sentence and sentence-segment workflows. It is designed to be reproducible for lab members, with utilities for dataset unification, metric computation, and result cleanup.

---

## Features

- Two-stage translation: initial draft generation and error-guided repair.
- Plug-in quality estimation via XCOMET with error span extraction.
- Optional sentence segmentation for finer-grained refinement (extended mode).
- vLLM-backed Qwen inference with automatic GPU selection and fallbacks.
- Dataset preparation utilities for multiple benchmarks (WMT, FLORES, CultureMT, DRT, RTT, CommonMT).
- Metric scripts for BLEU and COMET/COMET-Kiwi on produced results.
- Helpers to patch historical results and maintain consistent JSON outputs.

---

## Project Structure

```sh
└── SimultaneousTranslation/
    ├── README.md
    ├── data
    │   ├── process_data.py
    │   └── test
    ├── download_xcomet.py
    ├── examples
    │   ├── test_3_extended_gpu.json
    │   └── test_baseline_wmt24_en-zh_3.json
    ├── experiments
    │   ├── xcomet_1.0
    │   └── xcomet_2.0
    ├── main.py
    ├── playground
    │   └── test.py
    ├── qwen_generator.py
    ├── requirements.txt
    ├── runs
    │   ├── run_all-part1.sh
    │   ├── run_all-part2.sh
    │   └── run_wmt_enzh_zhen.sh
    ├── utils
    │   ├── clean_results.py
    │   ├── copy_repair_texts.py
    │   ├── fix_repair_translations-2.py
    │   ├── fix_repair_translations.py
    │   ├── fix_xcomet_final.py
    │   ├── main_wo_trans_tag.py
    │   ├── metrics-part1.py
    │   └── metrics-part2.py
    ├── utils.py
    ├── xcomet_all_stats.txt
    └── xcomet_loader.py
```

### Project Index

<details open>
	<summary><b><code>SIMULTANEOUSTRANSLATION/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/xcomet_loader.py'>xcomet_loader.py</a></b></td>
					<td style='padding: 8px;'>XCOMET checkpoint loader with CPU/GPU selection, batch scoring, error spans.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/xcomet_all_stats.txt'>xcomet_all_stats.txt</a></b></td>
					<td style='padding: 8px;'>Aggregated run statistics (format accuracy, XCOMET means, improvement counts) across datasets.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>Format checks, `<translate>` extraction, segmentation heuristics with optional wtpsplit.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>Core dependencies (torch/transformers/vllm, datasets, unbabel-comet, wtpsplit).</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/qwen_generator.py'>qwen_generator.py</a></b></td>
					<td style='padding: 8px;'>Qwen wrapper with vLLM first, memory-aware GPU picking, thinking-mode toggle.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>End-to-end driver (data loading, prompt building, Qwen draft/repair, XCOMET scoring, result saving) with baseline/extended modes.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/download_xcomet.py'>download_xcomet.py</a></b></td>
					<td style='padding: 8px;'>Helper to fetch Unbabel/XCOMET-XL checkpoint (requires HF auth).</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- utils Submodule -->
	<details>
		<summary><b>utils</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ utils</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/metrics-part2.py'>metrics-part2.py</a></b></td>
					<td style='padding: 8px;'>BLEU + COMET-DA + COMET-Kiwi scoring for result JSONs (part2 datasets).</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/metrics-part1.py'>metrics-part1.py</a></b></td>
					<td style='padding: 8px;'>BLEU + COMET-DA + COMET-Kiwi scoring for result JSONs (part1 datasets).</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/main_wo_trans_tag.py'>main_wo_trans_tag.py</a></b></td>
					<td style='padding: 8px;'>Older pipeline variant without `<translate>` tagging (debug/reference).</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/fix_xcomet_final.py'>fix_xcomet_final.py</a></b></td>
					<td style='padding: 8px;'>Adds placeholder `xcomet_final` for entries with missing drafts.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/fix_repair_translations.py'>fix_repair_translations.py</a></b></td>
					<td style='padding: 8px;'>Repairs malformed `<translate>` tags in historical repairs and re-scores with XCOMET.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/fix_repair_translations-2.py'>fix_repair_translations-2.py</a></b></td>
					<td style='padding: 8px;'>Same as above for additional result batches.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/copy_repair_texts.py'>copy_repair_texts.py</a></b></td>
					<td style='padding: 8px;'>Copies `repair_generated_text` between runs to sync outputs.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/utils/clean_results.py'>clean_results.py</a></b></td>
					<td style='padding: 8px;'>Drops extended-only keys and renames files (extended → baseline) for clarity.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- runs Submodule -->
	<details>
		<summary><b>runs</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ runs</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/runs/run_wmt_enzh_zhen.sh'>run_wmt_enzh_zhen.sh</a></b></td>
					<td style='padding: 8px;'>Example batch runs for WMT/CultureMT/DRT subsets with preset GPUs and num_samples.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/runs/run_all-part2.sh'>run_all-part2.sh</a></b></td>
					<td style='padding: 8px;'>Iterates over data/test/used-part2 jsonl files, baseline mode, GPU mapping example.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/runs/run_all-part1.sh'>run_all-part1.sh</a></b></td>
					<td style='padding: 8px;'>Iterates over data/test/used-part1 jsonl files, baseline mode, GPU mapping example.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- playground Submodule -->
	<details>
		<summary><b>playground</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ playground</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/playground/test.py'>test.py</a></b></td>
					<td style='padding: 8px;'>Scratchpad for Qwen chat template and repair prompt experiments.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- experiments Submodule -->
	<details>
		<summary><b>experiments</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ experiments</b></code>
			<!-- xcomet_2.0 Submodule -->
			<details>
				<summary><b>xcomet_2.0</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ experiments.xcomet_2.0</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/experiments/xcomet_2.0/xcomet_compare_s1_gt_100.py'>xcomet_compare_s1_gt_100.py</a></b></td>
							<td style='padding: 8px;'>Compare XCOMET strategies on >100 cases using new segmentation assumptions.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/experiments/xcomet_2.0/xcomet_build_cases.py'>xcomet_build_cases.py</a></b></td>
							<td style='padding: 8px;'>Builds evaluation cases for XCOMET 2.0 experiments.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- xcomet_1.0 Submodule -->
			<details>
				<summary><b>xcomet_1.0</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ experiments.xcomet_1.0</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/experiments/xcomet_1.0/xcomet_compare_strategies_3.py'>xcomet_compare_strategies_3.py</a></b></td>
							<td style='padding: 8px;'>Manual case study: compares whole-sentence vs segmented XCOMET scoring (3 cases).</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/experiments/xcomet_1.0/xcomet_compare_strategies_100.py'>xcomet_compare_strategies_100.py</a></b></td>
							<td style='padding: 8px;'>Expands comparison to 100 samples for robustness checks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/experiments/xcomet_1.0/xcomet_build_cases.py'>xcomet_build_cases.py</a></b></td>
							<td style='padding: 8px;'>Constructs synthetic/real cases for XCOMET 1.0 experiments.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- examples Submodule -->
	<details>
		<summary><b>examples</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ examples</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/examples/test_baseline_wmt24_en-zh_3.json'>test_baseline_wmt24_en-zh_3.json</a></b></td>
					<td style='padding: 8px;'>Sample baseline outputs with prompts, draft/repair translations, and XCOMET scores.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jetfan-xin/SimultaneousTranslation/blob/master/examples/test_3_extended_gpu.json'>test_3_extended_gpu.json</a></b></td>
					<td style='padding: 8px;'>Sample extended-mode outputs (segment repair) with error spans.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

- Python ≥3.10 recommended (PyTorch 2.x).
- GPU with CUDA for practical throughput (CPU works but slow).
- Hugging Face access for Qwen and XCOMET checkpoints.

### Installation

Build SimultaneousTranslation from the source and install dependencies:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/jetfan-xin/SimultaneousTranslation
    cd SimultaneousTranslation
    ```

2. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **(Optional) Download XCOMET-XL checkpoint:**

    ```sh
    python download_xcomet.py --output_dir /path/to/models/XCOMET-XL
    export WORD_QE_CKPT=/path/to/models/XCOMET-XL/checkpoints/model.ckpt
    ```

### Usage

Run the project with:

```sh
python main.py \
  --data_dir data/test/used \
  --test_files wmt24_en-zh.jsonl \
  --qwen_model_path Qwen/Qwen3-8B \
  --xcomet_ckpt /path/to/XCOMET-XL/checkpoints/model.ckpt \
  --xcomet_gpus 0 \
  --qwen_gpus 1 \
  --pipeline_mode baseline \
  --output_file results/wmt24_en-zh_baseline.json
```

- Baseline: full-sentence draft → XCOMET spans → optional repair if errors; final XCOMET rescoring.
- Extended: add `--pipeline_mode extended` to segment drafts (`utils.split_into_segments` / wtpsplit) and repair only errorful segments.
- GPU mapping: physical IDs are mapped to logical IDs (`map_physical_to_logical`); vLLM auto-picks a GPU with free memory and restores `CUDA_VISIBLE_DEVICES`.

Common flags (`main.py`):
- Generation: `--max_tokens_draft`, `--max_tokens_repair`, `--temperature`, `--top_p`, `--batch_size`.
- Devices: `--use_vllm/--no_use_vllm`, `--gpu_memory_utilization`, `--vllm_max_num_seqs`, `--xcomet_cpu`, `--qwen_cpu`.
- Data: `--num_samples` for quick smoke tests; Parquet is cached alongside JSONL.

### Testing

- No automated tests. Validate by running `main.py` on a small subset and evaluating with `utils/metrics-part1.py` or `utils/metrics-part2.py` (adjust COMET checkpoint paths inside).

---

## Roadmap

- [x] Two-stage pipeline (draft + repair) with Qwen and XCOMET scoring.
- [x] Unified evaluation data and Parquet prompt caching.
- [x] Maintenance scripts for historical results and metric computation.
- [ ] Stabilize extended-mode prompts/formatting and re-run failing extended sets (see `xcomet_all_stats.txt` zeros).
- [ ] Centralize model/checkpoint paths (Qwen defaults, COMET paths in metric scripts).
- [ ] Add lightweight regression checks on sampled subsets.

---

## Open Questions / TODO (need confirmation)
- 拓展模式实际未完成，后续需要修改 transformer。
- 改为 Qwen3-4B-Instruct-2507，增加 prompt 分支，重新设计为主动输出 thinking 标签 `[think][/think]`，在所需数据集上运行并评估翻译效果。
- 再次分析基线模式结果。
- 如果此次合理，则用润色翻译确定扩展模式配置：XCOMET 策略测评。

---

## Contributing

- **💬 Discussions**: Internal lab sync; document decisions in repo when changing prompts or checkpoints.
- **🐛 Issues**: Track GPU failures (vLLM init), format regressions, or extended-mode gaps.
- **💡 PRs**: Keep `<translate>` format stable; update metric paths/configs rather than hard-coding new absolutes.

---

## License

Distributed under the repository’s stated license (see badge).

---

## Acknowledgments

- Qwen models via Hugging Face (`qwen_generator.py`).
- Unbabel XCOMET (`download_xcomet.py`, `xcomet_loader.py`).
- Sentence segmentation with `wtpsplit[onnx-gpu]` (`utils.py`).
