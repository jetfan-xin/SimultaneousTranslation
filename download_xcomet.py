#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载XCOMET模型checkpoint
复用MT_Grpo/scripts/download_comet_ckpts.py中的实现
"""

import os
import argparse
from huggingface_hub import snapshot_download

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def download_xcomet(target_dir: str):
    """下载XCOMET-XL模型到目标目录，复用MT_Grpo的下载方式"""
    ckpt_path = os.path.join(target_dir, "checkpoints", "model.ckpt")
    
    if os.path.exists(ckpt_path):
        print(f"✅ XCOMET模型已存在：{ckpt_path}")
        return ckpt_path
    
    print(f"⬇️  开始下载 Unbabel/XCOMET-XL 到 {target_dir} ...")
    
    ensure_dir(os.path.join(target_dir, "checkpoints"))
    
    # 复用MT_Grpo中的下载方式
    snapshot_download(
        repo_id="Unbabel/XCOMET-XL",
        allow_patterns=["checkpoints/*", "hparams.yaml", "LICENSE", "README.md"],
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    if os.path.exists(ckpt_path):
        print(f"✅ 下载完成：{ckpt_path}")
    else:
        print(f"⚠️  下载完成但未检测到 {ckpt_path}，请检查下载内容：{target_dir}")
    
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="下载XCOMET模型checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/models/XCOMET-XL"),
        help="目标目录，用于存放XCOMET-XL模型"
    )
    
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    ckpt_path = download_xcomet(args.output_dir)
    
    print("\n🎉 XCOMET模型已准备完成。")
    print(f"XCOMET_CKPT={ckpt_path}")
    print(f"\n使用方法：")
    print(f"  export WORD_QE_CKPT={ckpt_path}")
    print(f"  或在main.py中使用 --xcomet_ckpt {ckpt_path}")


if __name__ == "__main__":
    main()

