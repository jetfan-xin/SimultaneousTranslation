# -*- coding: utf-8 -*-
"""
工具函数：格式检查和内容提取
"""

from typing import Tuple, Optional, List
import re
import torch
from wtpsplit import SaT

# ============================================================
# SaT-3l-sm 句子切分模型（ONNX + GPU 优先）
# ============================================================

# 检查是否有 CUDA
_HAS_CUDA = torch.cuda.is_available()

if _HAS_CUDA:
    # ONNX 推理，优先使用 GPU，再回退 CPU
    _sat_model = SaT(
        "sat-3l-sm",
        ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
else:
    # 只有 CPU 的情况：只用 CPUExecutionProvider
    _sat_model = SaT(
        "sat-3l-sm",
        ort_providers=["CPUExecutionProvider"],
    )

# SaT 默认就是 ONNX 推理（因为我们传了 ort_providers），
# 不需要再 .half().to("cuda")，那是 PyTorch 模型的写法。


def check_and_extract_translate_tag(text: str) -> Tuple[bool, Optional[str], int]:
    """
    检查文本中是否包含<translate>标签，如果包含则提取内容
    
    Args:
        text: 要检查的文本
    
    Returns:
        Tuple[bool, Optional[str], int]:
            - 格式是否正确（是否包含完整的<translate>标签）
            - 提取的翻译内容（如果格式正确）
            - 格式分数（1表示正确，0表示错误）
    """
    if not text or not isinstance(text, str):
        return False, None, 0
    
    # 使用字符串查找来获取最后一对<translate>...</translate>
    # 这样可以避免在<think>中出现未闭合的<translate>时被提前截断
    start_idx = text.rfind('<translate>')
    end_idx = text.rfind('</translate>')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        translate_content = text[start_idx + len('<translate>'):end_idx].strip()
        if translate_content:
            return True, translate_content, 1
    
    # 如果没有找到完整的标签，返回格式错误
    return False, None, 0


def format_error_spans_for_prompt(error_spans: list) -> str:
    """
    将错误spans格式化为用于prompt的字符串
    
    Args:
        error_spans: 错误spans列表，每个元素是dict，包含：
            - text: 错误文本
            - start: 起始位置
            - end: 结束位置
            - severity: 错误严重程度
            - confidence: 置信度
    
    Returns:
        格式化的字符串
    """
    if not error_spans:
        return "No errors detected"
    
    formatted_spans = []
    for i, span in enumerate(error_spans, 1):
        if isinstance(span, dict):
            text = span.get('text', '')
            severity = span.get('severity', 'unknown')
            confidence = span.get('confidence', 0.0)
            formatted_spans.append(f"{i}. Text: \"{text}\" (Severity: {severity}, Confidence: {confidence:.2f})")
        else:
            formatted_spans.append(f"{i}. {span}")
    
    return "\n".join(formatted_spans)


# 终结性标点：强制切分（保留在前一段）
HARD_PUNCT_PATTERN = r'([。！？!?]+)'

# 软边界标点：有条件切分（逗号、顿号、分号、冒号等）
SOFT_PUNCT = set(list('，,、；;：:…‥'))

# 成对符号，用于避免在内部切分
LEFT_BRACKETS = set('([{（【《「『')
RIGHT_BRACKETS = set(')]}）】》」』')
QUOTE_CHARS = set('"""\'\'')

# 典型连接词，用于判断软边界后面是否适合开新段
ZH_CONNECTIVES = ('但是', '但', '然而', '不过', '所以', '因此', '于是', '然后', '同时', '之后', '接着', '相比之下')
EN_CONNECTIVES = ('but', 'however', 'therefore', 'so', 'then', 'for example', 'for instance',
                  'in addition', 'on the other hand', 'meanwhile', 'furthermore', 'moreover')


def is_connective_start(text: str, idx: int) -> bool:
    """判断从 idx 开始是否是常见连接词，用于软切分判断。"""
    if idx >= len(text):
        return False
    
    # 中文
    for w in ZH_CONNECTIVES:
        if text.startswith(w, idx):
            return True
    
    # 英文（简单大小写不敏感）
    if idx + 30 <= len(text):
        lower = text[idx:idx+30].lower()
    else:
        lower = text[idx:].lower()
    
    for w in EN_CONNECTIVES:
        if lower.startswith(w):
            # 确保是完整单词（后面是空格、标点或结尾）
            end_pos = idx + len(w)
            if end_pos >= len(text) or text[end_pos] in ' ,.;:!?，。；：！？':
                return True
    
    return False


def split_long_with_soft_punct(sentence: str, max_len: int = 100, hard_max_len: int = 150) -> List[str]:
    """
    在终结句内部，用软标点和长度限制继续切分。
    
    Args:
        sentence: 待切分的句子
        max_len: 理想长度（尽量不超过）
        hard_max_len: 绝对不能超过的长度，实在不行强切
    
    Returns:
        切分后的短句列表
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    if len(sentence) <= max_len:
        return [sentence]

    segments = []
    last_cut = 0
    paren_depth = 0
    quote_depth = 0

    def can_cut(pos: int) -> bool:
        """判断在pos位置是否可以切分"""
        # 不在括号或引号内部
        if paren_depth > 0 or quote_depth > 0:
            return False
        # 避免句首或句尾
        if pos <= 0 or pos >= len(sentence) - 1:
            return False
        return True

    i = 0
    while i < len(sentence):
        ch = sentence[i]

        # 更新成对符号状态
        if ch in LEFT_BRACKETS:
            paren_depth += 1
        elif ch in RIGHT_BRACKETS and paren_depth > 0:
            paren_depth -= 1
        elif ch in QUOTE_CHARS:
            # 简易处理：进入/退出引号状态
            quote_depth ^= 1

        # 软边界候选
        if ch in SOFT_PUNCT and can_cut(i):
            current_len = i + 1 - last_cut  # 包含这个标点
            # 如果当前段太长了，或者接近 max_len，就在这里切
            if current_len >= max_len or \
               (current_len >= max_len * 0.7 and is_connective_start(sentence, i + 1)):
                segments.append(sentence[last_cut:i+1].strip())
                last_cut = i + 1

        # 若这一段实在太长超过 hard_max_len，强制从这里切
        if i + 1 - last_cut >= hard_max_len:
            segments.append(sentence[last_cut:i+1].strip())
            last_cut = i + 1

        i += 1

    # 收尾
    tail = sentence[last_cut:].strip()
    if tail:
        # 如果最后一小段太短，且有前一段，则并回去，避免「碎片」
        if segments and len(tail) < max_len * 0.3:
            segments[-1] = (segments[-1] + tail).strip()
        else:
            segments.append(tail)

    return segments


def split_into_segments(text: str, max_len: int = 100, hard_max_len: int = 150) -> List[str]:
    """
    同传式切块：将文本切成适合翻译的短句列表。
    
    采用硬边界+软边界的策略：
    - 硬边界：句末终结符（。！？!?）强制切分
    - 软边界：基于长度和结构的启发式切分（，、；: 等）
    - 避免在成对结构（括号、引号）内部切分
    - 避免在连接词结构中间切分
    
    Args:
        text: 待划分的文本
        max_len: 理想短句长度（默认100字符）
        hard_max_len: 绝对最大长度，超过则强制切分（默认150字符）
    
    Returns:
        按顺序排列的短句列表（包含原标点）。
    """
    if not text or not isinstance(text, str):
        return []

    results = []

    # 按段落处理（空行分隔）
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue

        # 先按终结性标点做初步切分（保留标点）
        parts = re.split(HARD_PUNCT_PATTERN, para)
        # parts: [chunk, punct, chunk, punct, ..., last_chunk(可空)]

        for i in range(0, len(parts), 2):
            chunk = parts[i]
            if not chunk:
                continue
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            sentence = (chunk + punct).strip()

            # 对每个 sentence 再用软标点和长度规则细分
            sub_segments = split_long_with_soft_punct(sentence,
                                                      max_len=max_len,
                                                      hard_max_len=hard_max_len)
            for seg in sub_segments:
                seg = seg.strip()
                if not seg:
                    continue
                # 可以在这里做「合并短段」的小优化（可选）
                if results and len(results[-1]) + len(seg) <= max_len * 0.6:
                    # 与上一段合并，避免太碎
                    results[-1] = (results[-1] + seg).strip()
                else:
                    results.append(seg)

    # 如果没有结果（可能是没有标点的纯文本），返回原文本
    if not results:
        return [text.strip()] if text.strip() else []
    
    return results


# ============================================================
# 新增：使用 wtpsplit 的分句器
# ============================================================

def split_into_segments_wtpsplit(text: str) -> List[str]:
    """
    使用 SaT-3l-sm（ONNX 推理）进行句子切分。

    依赖:
        - pip install wtpsplit[onnx-gpu]
    """
    if not text or not isinstance(text, str):
        return []

    # SaT.split 支持传入单个字符串或字符串列表：
    #   - sat.split("...") -> ["句1", "句2", ...]
    #   - sat.split([...]) -> 迭代器，每个元素是一个句子列表
    try:
        segments = _sat_model.split(text)
    except Exception as e:
        # 保险起见出点问题直接退回整段
        print(f"[wtpsplit] SaT-3l-sm split failed, fallback to single segment. Error: {e}")
        return [text.strip()] if text.strip() else []

    # 清理一下空白
    segments = [
        s.strip()
        for s in segments
        if isinstance(s, str) and s.strip()
    ]

    # 如果 SaT 没切出来（极端情况），就退回原文
    if not segments and text.strip():
        return [text.strip()]

    return segments