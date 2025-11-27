import pandas as pd
from pathlib import Path
import re

# 结果文件所在目录
ROOT = Path("/ltstorage/home/4xin/SimultaneousTranslation/results_Qwen2.5-3B_100")
all_data = pd.DataFrame()


def parse_info_from_filename(filename: str):
    """
    从结果 csv 文件名中解析出：
      - dataset: commonmt / drt / flores101 / rtt / culturemt / unknown
      - subset:  对于 commonmt: contextless/contextual/lexical
                 对于 drt: MetaphorTrans
      - lang_pair: en-zh / zh-en / de-en ...（如果缺失，部分数据集有默认）
      - method: baseline / 其它（来自 test_XXX_*）
    """
    name = filename.lower()

    # ---------- 0.基线还是扩展 ----------
    # 先从文件名里直接正则抓 en-zh / zh-en 等
    mode = "baseline" if "baseline" in name else "extended"

    # ---------- 1. 语言对 ----------
    # 先从文件名里直接正则抓 en-zh / zh-en 等
    lang_match = re.search(r"([a-z]{2,3})-([a-z]{2,3})", name)
    lang_pair = lang_match.group(0) if lang_match else "unknown"

    # ---------- 2. 数据集 + 子集 ----------
    dataset = "unknown"
    subset = None

    # commonMT 三个子集
    if "commonmt" in name:
        dataset = "commonmt"
        if "contextless" in name:
            subset = "contextless"
        elif "contextual" in name:
            subset = "contextual"
        elif "lexical" in name:
            subset = "lexical"

        # commonmt 目前你只跑了 zh-en，如果文件名里没写，就默认 zh-en
        if lang_pair == "unknown":
            lang_pair = "zh-en"

    # DRT（MetaphorTrans）
    elif "drt" in name:
        dataset = "drt"

    # FLORES-101
    elif "flores101" in name:
        dataset = "flores101"

    # RTT
    elif "rtt" in name:
        dataset = "rtt"

    # CultureMT
    elif "culturemt" in name:
        dataset = "culturemt"
    # wmt
    elif "wmt24" in name:
        dataset = "wmt24"

    elif "wmt23" in name:
        dataset = "wmt23"

    # ---------- 3. method ----------
    # 约定：文件名形如 test_baseline_xxx_total.csv
    # 用 test_ 和下一个下划线之间的部分作为 method
    method_match = re.search(r"test_([^_]+)_", name)
    method = method_match.group(1) if method_match else "unknown"

    return mode, dataset, subset, lang_pair, method


for f in ROOT.rglob("*_total.csv"):
    df = pd.read_csv(f)

    mode, dataset, subset, lang_pair, method = parse_info_from_filename(f.name)

    df["mode"] = mode
    df["dataset"] = dataset
    df["subset"] = subset  # 对于非 commonmt/drt 可能是 None
    df["lang_pair"] = lang_pair
    df["method"] = method
    df["file"] = f.name
    df["folder"] = f.parent.name

    all_data = pd.concat([all_data, df], ignore_index=True)

output_file = ROOT / "merged.csv"
all_data.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"所有 CSV 文件已合并到: {output_file}")
print(f"总行数: {len(all_data)}")