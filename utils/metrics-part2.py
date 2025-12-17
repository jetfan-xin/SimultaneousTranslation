import json
from pathlib import Path
import pandas as pd
import os
import sys

import sacrebleu
from comet import load_from_checkpoint

# ===================== 模型加载 =====================

print("load COMET-Kiwi")
# 修改成你自己的 kiwi ckpt 路径
comet_kiwi_model = load_from_checkpoint(
    "/ltstorage/home/4xin/models/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt"
)

print("load COMET-DA (xcomet-style sentence scorer)")
# 修改成你自己的 COMET-DA ckpt 路径
comet_model = load_from_checkpoint(
    "/mnt/data1/users/4xin/hf/hub/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
)

# ===================== BLEU 计算 =====================

def bleu_score(predict, answer, lang, is_sent=False):
    """
    predict: sys translation(s)
      - is_sent=False: List[str]
      - is_sent=True:  str
    answer:  reference(s)
      - is_sent=False: [List[str]]
      - is_sent=True:  str
    lang: target language code, e.g. 'zh', 'en', 'de'
    """
    tokenize_map = {
        "zh": "zh",           # 官方中文 tokenizer
        "ja": "ja-mecab",     # 日语专用
        "ko": "ko-mecab",     # 如果你以后测韩语
        "th": "none",         # 泰语（FLORES 已经字级切分）
        "ar": "none",         # 阿拉伯语
        "hi": "none",         # 印地语
        "ru": "none",         # 俄语（空格分词可靠）
        "tr": "none",         # 土耳其语（空格分词）
        "cs": "none",         # 捷克语（空格分词）
        "vi": "none",         # 越南语
        "te": "none",         # Telugu
        "ta": "none",         # Tamil

        # 西方语言使用 intl 会更稳定
        "fr": "intl",         # 法语
        "es": "intl",         # 西班牙语
        "it": "intl",         # 意大利语
        "pt": "intl",         # 葡萄牙语
        "de": "intl",         # 德语
        "nl": "intl",         # 荷兰语
        "en": "intl",         # 英语

        # 其他未列出的默认使用 flores101 方案
    }
    tokenize = tokenize_map.get(lang, "13a")

    if is_sent:
        bleu = sacrebleu.sentence_bleu(
            predict, [answer], lowercase=True, tokenize=tokenize
        )
    else:
        bleu = sacrebleu.corpus_bleu(
            predict, answer, lowercase=True, tokenize=tokenize
        )
    return bleu.score


# ===================== 单个 JSON 评测函数 =====================

def eval_results_file(result_file: str):
    """
    读取 main.py 生成的结果 json：
    [
      {
        "index": 0,
        "lang_pair": "en-zh" / "lg": "en-zh",
        "src_text": "...",
        "tgt_text": "...",
        "draft_translation": "..."/null,
        "final_translation": "..."/null,
        ...
      },
      ...
    ]

    分别在 draft_translation 和 final_translation 上计算：
      - corpus BLEU
      - sentence BLEU
      - COMET-DA (with ref): sentence scores + system_score
      - COMET-Kiwi (QE): sentence scores + system_score

    输出：
      - <result_file_without_ext>_total.csv
      - <result_file_without_ext>_each.csv
    """
    print(f"\n>>> Evaluating file: {result_file}")
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("  [Warn] Empty json, skip.")
        return

    # ---------- 解析语言方向 ----------
    first = data[0]
    lang_pair = first.get("lang_pair")
    src_lang, tgt_lang = lang_pair.split("-")

    print(f"Detected language pair: {lang_pair}")

    # ---------- 仅收集 final（润色后）数据 ----------
    base_rows = []
    final_ids, final_srcs, final_refs, final_mts = [], [], [], []

    for ex in data:
        idx = ex.get("index")
        src = (ex.get("src_text") or "").strip()
        ref = (ex.get("tgt_text") or "").strip()
        draft = ex.get("draft_translation")
        final = ex.get("final_translation")

        if isinstance(draft, str):
            draft = draft.strip()
        else:
            draft = None

        if isinstance(final, str):
            final = final.strip()
        else:
            final = None

        base_rows.append(
            {
                "index": idx,
                "src": src,
                "ref": ref,
                "draft_translation": draft,
                "final_translation": final,
            }
        )

        # final 用于评测
        if src and ref and final:
            final_ids.append(idx)
            final_srcs.append(src)
            final_refs.append(ref)
            final_mts.append(final)

    # ---------- 仅评测 final ----------
    final_corpus_bleu = None
    final_sentence_bleus = {}
    final_comet_scores = {}
    final_comet_kiwi_scores = {}
    final_sys_comet = None
    final_sys_kiwi = None

    if final_mts:
        print(f"  [Final] #valid samples: {len(final_mts)}")

        final_corpus_bleu = bleu_score(final_mts, [final_refs], tgt_lang, is_sent=False)

        comet_inputs = [
            {"src": s, "mt": m, "ref": r}
            for s, m, r in zip(final_srcs, final_mts, final_refs)
        ]
        model_output = comet_model.predict(comet_inputs, batch_size=8, gpus=1)
        comet_sent = list(model_output.scores)
        final_sys_comet = model_output.system_score

        kiwi_inputs = [{"src": s, "mt": m} for s, m in zip(final_srcs, final_mts)]
        kiwi_output = comet_kiwi_model.predict(kiwi_inputs, batch_size=8, gpus=1)
        kiwi_sent = list(kiwi_output.scores)
        final_sys_kiwi = kiwi_output.system_score

        bleu_sent_list = []
        for m, r in zip(final_mts, final_refs):
            b = bleu_score(m, r, tgt_lang, is_sent=True)
            bleu_sent_list.append(b)

        for i, idx in enumerate(final_ids):
            final_sentence_bleus[idx] = bleu_sent_list[i]
            final_comet_scores[idx] = comet_sent[i]
            final_comet_kiwi_scores[idx] = kiwi_sent[i]
    else:
        print("  [Final] No valid final translations, skip metric computation.")

    # ---------- 保存总分 CSV ----------
    base = result_file.rsplit(".json", 1)[0]
    total_path = base + "_total.csv"

    # ---------- 保存总分 CSV（保留已有 draft 列） ----------
    if os.path.exists(total_path):
        existing = pd.read_csv(total_path).iloc[0].to_dict()
    else:
        existing = {}

    total_row = existing.copy()
    total_row.update(
        {
            "lang_pair": lang_pair,
            "num_final": len(final_mts),
            "BLEU_final": final_corpus_bleu,
            "COMET_final_sys": final_sys_comet,
            "COMET_KIWI_final_sys": final_sys_kiwi,
        }
    )

    pd.DataFrame([total_row]).to_csv(
        total_path, index=False, encoding="utf-8-sig"
    )
    print(f"  [Saved] total metrics -> {total_path}")

    # ---------- 保存逐句 CSV（保留已有 draft 列） ----------
    each_path = base + "_each.csv"
    existing_rows = {}
    if os.path.exists(each_path):
        df_exist = pd.read_csv(each_path)
        for _, r in df_exist.iterrows():
            existing_rows[int(r["index"])] = r.to_dict()

    each_rows = []
    for row in base_rows:
        idx = int(row["index"])
        base_dict = existing_rows.get(idx, {})
        base_dict.update(
            {
                "index": idx,
                "src": row["src"],
                "ref": row["ref"],
                "final_translation": row["final_translation"],
                "BLEU_final": final_sentence_bleus.get(idx),
                "COMET_final": final_comet_scores.get(idx),
                "COMET_KIWI_final": final_comet_kiwi_scores.get(idx),
            }
        )
        each_rows.append(base_dict)

    pd.DataFrame(each_rows).to_csv(each_path, index=False, encoding="utf-8-sig")
    print(f"  [Saved] sentence-level metrics -> {each_path}")


# ===================== main：遍历一个目录下所有结果 json =====================

if __name__ == "__main__":
    """
    用法示例：

        CUDA_VISIBLE_DEVICES=0 \
        python metrics.py /ltstorage/home/4xin/SimultaneousTranslation

    会在给定目录下递归查找所有 .json 文件，
    对每个 main.py 的结果 json 运行 eval_results_file。
    """

    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = f"/ltstorage/home/4xin/SimultaneousTranslation/results_Qwen3-4B-part2"

    base_path = os.path.abspath(base_path)
    print(f"Searching for result jsons under: {base_path}")

    for data_file in Path(base_path).rglob("*.json"):
        name = os.path.basename(data_file)
        if not name.endswith(".json"):
            continue

        try:
            eval_results_file(str(data_file))
        except Exception as e:
            print(f"  [Error] Failed to evaluate {data_file}: {e}")
