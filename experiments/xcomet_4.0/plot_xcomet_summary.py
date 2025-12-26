import json
import pandas as pd
from pathlib import Path

summary_path = Path("/ltstorage/home/4xin/SimultaneousTranslation/experiments/xcomet_4.0/XCOMET-XL_eval_outputs/xcomet_metrics/xcomet_metrics_summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))

STRATEGY_ORDER = ["1.1", "1.2"]
METRIC_ORDER = [
    "total",
    "accurate",
    "accuracy_percent",
    "avg_iou",
    "avg_precision",
    "avg_recall",
    "avg_f1",
]


def build_row(group_type, group, strategies, num_cases=None):
    row = {
        "group_type": group_type,
        "group": group,
    }
    if num_cases is not None:
        row["num_cases"] = num_cases
    for sid in STRATEGY_ORDER:
        if sid not in strategies:
            continue
        stats = strategies.get(sid, {})
        overall = stats.get("overall", {})
        for key in METRIC_ORDER:
            if key in overall:
                row[f"{key}_{sid}"] = overall[key]
    return row


rows = []
dataset_summaries = data.get("datasets", {})


def normalize_num_cases(value):
    if isinstance(value, (int, float)):
        return int(value)
    return 0


overall_cases = 0
cases_by_type = {}
cases_by_lang = {}
for summary in dataset_summaries.values():
    num_cases = normalize_num_cases(summary.get("num_cases"))
    overall_cases += num_cases
    dataset_type = summary.get("dataset_type")
    if dataset_type:
        cases_by_type[dataset_type] = cases_by_type.get(dataset_type, 0) + num_cases
    language_pair = summary.get("language_pair")
    if language_pair:
        cases_by_lang[language_pair] = cases_by_lang.get(language_pair, 0) + num_cases


rows.append(build_row("overall", "overall", data.get("overall", {}), overall_cases))

for group, strategies in data["by_dataset_type"].items():
    rows.append(build_row("dataset_type", group, strategies, cases_by_type.get(group, 0)))

for group, strategies in data["by_language_pair"].items():
    rows.append(build_row("language_pair", group, strategies, cases_by_lang.get(group, 0)))

for dataset_name in sorted(dataset_summaries.keys()):
    summary = dataset_summaries[dataset_name]
    strategies = summary.get("strategies", {})
    rows.append(build_row("dataset", dataset_name, strategies, normalize_num_cases(summary.get("num_cases"))))

df = pd.DataFrame(rows)
ordered_cols = ["group_type", "group", "num_cases"]
for key in METRIC_ORDER:
    for sid in STRATEGY_ORDER:
        ordered_cols.append(f"{key}_{sid}")
existing_cols = [c for c in ordered_cols if c in df.columns]
remaining_cols = [c for c in df.columns if c not in existing_cols]
df = df[existing_cols + remaining_cols]
out = summary_path.parent / "xcomet_summary.csv"
df.to_csv(out, index=False)
print(out)
