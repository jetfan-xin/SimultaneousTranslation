# -*- coding: utf-8 -*-
"""
Compare different XCOMET usage strategies for simultaneous translation scenario
on three cases.

Case 1 (synthetic):
    Remote workers example（手工构造 GOOD / WRONG），用于 sanity check。

Case 2 (from data, index=1):
    But the victim's brother says he can't think of anyone who would want to hurt him...
    我们手动定义：
      - mt_full_good: 忠实翻译
      - mt_full_bad: 仅前半句出错（人物关系/态度错），后半句保持“好起来了”正确含义
      - 并给出 src_segs / mt_segs_* 保证对齐

Case 3 (from data, index=2):
    The body found at the Westfield Mall Wednesday morning...
    我们手动定义：
      - mt_full_good: 忠实翻译（Westfield + 旧金山法医 + 正确信息）
      - mt_full_bad: 仅前半句出错（时间/地点错），后半句保持正确身份信息
      - 同样提供 src_segs / mt_segs_* 保证对齐

Strategies:

1) Strategy 1 (baseline-style whole sentence):
   - WITHOUT ref: XCOMET(S_full, MT_full_good/bad)
   - WITH ref:    XCOMET(S_full, MT_full_good/bad, REF_full)

2) Strategy 2 (source-side segmentation, original project style):
   - XCOMET(S_seg, MT_seg) using manually aligned src_segs & mt_segs_good/bad

3) Strategy 3 (P!K0 style: full source, segmented outputs):
   - WITHOUT ref: XCOMET(S_full, MT_seg_good/bad)
   - WITH ref:    XCOMET(S_full, MT_seg_good/bad, REF_full)

All segmentation is manually specified in build_cases().
No split_into_segments is used.
"""

import os
import sys
from typing import List, Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xcomet_loader import XCOMETLoader  # noqa: E402


# ================= Utils =================

def pretty(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_results(samples: List[Dict[str, str]], results: List[Dict[str, Any]], indent: str = "    "):
    for i, (s, r) in enumerate(zip(samples, results), 1):
        mt = s["mt"]
        score = r.get("score")
        spans = r.get("error_spans", []) or []

        print(f"\n[{i}] MT: {mt}")
        print(f"{indent}Score: {score}")

        if spans:
            span_str = ", ".join(
                f"{span.get('text', '')}({span.get('severity', '')})"
                for span in spans
                if span.get("text")
            )
            print(f"{indent}Error spans: {span_str}")
        else:
            print(f"{indent}Error spans: []")


# ================= Build Cases =================

def build_cases() -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}

    # ----- Case 1: synthetic remote workers -----
    src1 = (
        "The company announced a new policy to support remote workers during the winter, "
        "and promised additional financial assistance for employees living in colder regions."
    )
    ref1 = (
        "公司宣布了一项新政策，在冬季为远程办公员工提供支持，"
        "并承诺为居住在更寒冷地区的员工提供额外的经济补助。"
    )

    # 手动对齐的源端分句
    src1_segs = [
        "The company announced a new policy to support remote workers during the winter, ",
        "and promised additional financial assistance for employees living in colder regions.",
    ]

    # GOOD: 忠实翻译，两段
    mt1_segs_good = [
        "公司宣布了一项新政策，在冬季为远程办公员工提供支持。",
        "并承诺为居住在更寒冷地区的员工提供额外的经济补助。",
    ]
    # BAD: 第二段语义反向
    mt1_segs_bad = [
        "公司宣布了一项新政策，在冬季为远程办公员工提供支持。",
        "但也表示将减少对居住在更寒冷地区员工的帮助。",
    ]

    mt1_full_good = "".join(mt1_segs_good)
    mt1_full_bad = "".join(mt1_segs_bad)

    cases["Case1"] = {
        "label": "Case 1 (synthetic remote workers)",
        "src_full": src1,
        "ref_full": ref1,
        "src_segs": src1_segs,
        "mt_full_good": mt1_full_good,
        "mt_full_bad": mt1_full_bad,
        "mt_segs_good": mt1_segs_good,
        "mt_segs_bad": mt1_segs_bad,
    }

    # ----- Case 2: index=1 -----
    s2_src = (
        "But the victim's brother says he can't think of anyone who would want to hurt him, "
        "saying, \"Things were finally going well for him.\""
    )
    s2_ref = "但受害人的哥哥表示想不出有谁会想要加害于他，并称“一切终于好起来了。”"

    # 源端分句（对齐思路：前半陈述 + 引号部分）
    s2_src_segs = [
        "But the victim's brother says he can't think of anyone who would want to hurt him, ",
        "saying, \"Things were finally going well for him.\"",
    ]

    # GOOD：忠实翻译
    s2_mt_segs_good = [
        "但受害人的哥哥表示想不出有谁会想要加害于他，",
        "并称“一切终于好起来了。”",
    ]
    s2_mt_full_good = "".join(s2_mt_segs_good)

    # BAD：只在前半句动手（朋友 + 觉得很多人想伤害他），后半句保持原“好起来了”
    s2_mt_segs_bad = [
        "但是受害者的朋友说他一直觉得很多人想要伤害他，",
        "并称“一切终于好起来了。”",
    ]
    s2_mt_full_bad = "".join(s2_mt_segs_bad)

    cases["Case2"] = {
        "label": "Case 2 (victim's brother, things going well)",
        "src_full": s2_src,
        "ref_full": s2_ref,
        "src_segs": s2_src_segs,
        "mt_full_good": s2_mt_full_good,
        "mt_full_bad": s2_mt_full_bad,
        "mt_segs_good": s2_mt_segs_good,
        "mt_segs_bad": s2_mt_segs_bad,
    }

    # ----- Case 3: index=2 -----
    s3_src = (
        "The body found at the Westfield Mall Wednesday morning was identified as 28-year-old "
        "San Francisco resident Frank Galicia, the San Francisco Medical Examiner's Office said."
    )
    s3_ref = "旧金山验尸官办公室表示，周三早上于西田购物中心发现的尸体确认为28岁旧金山居民 Frank Galicia。"

    # 源端分句（前半“办公室表示”，后半具体信息）
    s3_src_segs = [
        "The San Francisco Medical Examiner's Office said ",
        "the body found at the Westfield Mall Wednesday morning was identified as "
        "28-year-old San Francisco resident Frank Galicia.",
    ]

    # GOOD：忠实翻译
    s3_mt_segs_good = [
        "旧金山验尸官办公室表示，",
        "周三早上于西田购物中心发现的尸体确认为28岁旧金山居民 Frank Galicia。",
    ]
    s3_mt_full_good = "".join(s3_mt_segs_good)

    # BAD：前半句地点/时间错，后半句身份信息保持一致
    s3_mt_segs_bad = [
        "旧金山验尸官办公室表示，",
        "周二晚上在一处购物中心发现的一具尸体仍未确认身份。",
    ]
    s3_mt_full_bad = "".join(s3_mt_segs_bad)

    cases["Case3"] = {
        "label": "Case 3 (Westfield Mall, Frank Galicia)",
        "src_full": s3_src,
        "ref_full": s3_ref,
        "src_segs": s3_src_segs,
        "mt_full_good": s3_mt_full_good,
        "mt_full_bad": s3_mt_full_bad,
        "mt_segs_good": s3_mt_segs_good,
        "mt_segs_bad": s3_mt_segs_bad,
    }

    return cases


# ================= Strategies =================

def run_strategy_1(xcomet: XCOMETLoader, case: Dict[str, Any]):
    """
    Strategy 1: baseline-style whole-sentence evaluation.

    - [1.1] WITHOUT ref: XCOMET(S_full, MT_full_good/bad)
    - [1.2] WITH ref:    XCOMET(S_full, MT_full_good/bad, REF_full)
    """
    label = case["label"]
    src = case["src_full"]
    ref = case["ref_full"]
    mt_good = case["mt_full_good"]
    mt_bad = case["mt_full_bad"]

    pretty(f"Strategy 1: {label}  |  S_full + (GOOD vs BAD) MT_full")

    # 1.1 no ref
    print("\n[1.1] WITHOUT ref")
    triplets = [{"src": src, "mt": mt_good}, {"src": src, "mt": mt_bad}]
    res = xcomet.predict(triplets)
    print_results(triplets, res)

    # 1.2 with ref
    print("\n[1.2] WITH ref")
    triplets_ref = [
        {"src": src, "mt": mt_good, "ref": ref},
        {"src": src, "mt": mt_bad, "ref": ref},
    ]
    res_ref = xcomet.predict(triplets_ref)
    print_results(triplets_ref, res_ref)

    print("\n[Note] 这里检查整句级：GOOD 是否明显高于 BAD。")


def run_strategy_2(xcomet: XCOMETLoader, case: Dict[str, Any]):
    """
    Strategy 2: source-side segmentation (project original idea)

    - XCOMET(S_seg, MT_seg) for GOOD and BAD, using manually aligned src_segs & mt_segs.
    """
    label = case["label"]
    src_segs = case["src_segs"]
    mt_segs_good = case["mt_segs_good"]
    mt_segs_bad = case["mt_segs_bad"]

    pretty(f"Strategy 2: {label}  |  S_seg + MT_seg (aligned)")

    # GOOD
    triplets_good = [{"src": s, "mt": m} for s, m in zip(src_segs, mt_segs_good)]
    print("\n[2A] GOOD segments")
    res_good = xcomet.predict(triplets_good)
    print_results(triplets_good, res_good)

    # BAD
    triplets_bad = [{"src": s, "mt": m} for s, m in zip(src_segs, mt_segs_bad)]
    print("\n[2B] BAD segments")
    res_bad = xcomet.predict(triplets_bad)
    print_results(triplets_bad, res_bad)

    print("\n[Note] 看每一对应分句上，GOOD vs BAD 是否被 XCOMET 拉开，"
          "这对应“先切 S 再翻译”的评估可行性。")


def run_strategy_3(xcomet: XCOMETLoader, case: Dict[str, Any]):
    """
    Strategy 3: full source + segmented MT (P!K0 style)

    - 不切 S，用 S_full
    - 对 mt_segs_good / mt_segs_bad 的每个片段：
        3.1 WITHOUT ref: XCOMET(S_full, MT_seg)
        3.2 WITH ref:    XCOMET(S_full, MT_seg, REF_full)
    """
    label = case["label"]
    src = case["src_full"]
    ref = case["ref_full"]
    mt_segs_good = case["mt_segs_good"]
    mt_segs_bad = case["mt_segs_bad"]

    pretty(f"Strategy 3: {label}  |  S_full + MT_seg")

    # GOOD
    print("\n[3.1-GOOD] WITHOUT ref")
    triplets_g = [{"src": src, "mt": m} for m in mt_segs_good]
    res_g = xcomet.predict(triplets_g)
    print_results(triplets_g, res_g)

    print("\n[3.2-GOOD] WITH ref")
    triplets_g_r = [{"src": src, "mt": m, "ref": ref} for m in mt_segs_good]
    res_g_r = xcomet.predict(triplets_g_r)
    print_results(triplets_g_r, res_g_r)

    # BAD
    print("\n[3.1-BAD] WITHOUT ref")
    triplets_b = [{"src": src, "mt": m} for m in mt_segs_bad]
    res_b = xcomet.predict(triplets_b)
    print_results(triplets_b, res_b)

    print("\n[3.2-BAD] WITH ref")
    triplets_b_r = [{"src": src, "mt": m, "ref": ref} for m in mt_segs_bad]
    res_b_r = xcomet.predict(triplets_b_r)
    print_results(triplets_b_r, res_b_r)

    print("\n[Note] 这里看：在完整 S 条件下，包含前半错误的 BAD 片段是否被打低分，"
          "以及这种用法是否适合作为同传边界/质检信号。")


# ================= Main =================

def main():
    xcomet = XCOMETLoader()
    cases = build_cases()

    for case in cases.values():
        print("\n" + "#" * 80)
        print(f"Running all strategies on {case['label']}")
        print("#" * 80)

        run_strategy_1(xcomet, case)
        run_strategy_2(xcomet, case)
        run_strategy_3(xcomet, case)


if __name__ == "__main__":
    main()