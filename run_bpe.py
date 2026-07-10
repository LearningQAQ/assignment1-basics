"""
BPE 一体化脚本：训练词表 → 立即分词 → 保存 .bin

用法:
    python run_bpe.py --input data/train.txt --vocab-size 10000 --max-tokens 40000000
"""
import os
import sys
import time
import json
import base64
import argparse
import numpy as np
from typing import Dict, List, Tuple

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


# ==================== I/O 工具函数 ====================

def _b64enc(b: bytes) -> str:
    return base64.b64encode(b).decode('ascii')

def _b64dec(s: str) -> bytes:
    return base64.b64decode(s.encode('ascii'))

def save_vocab(vocab: Dict[int, bytes], path: str) -> None:
    data = {str(k): _b64enc(v) for k, v in vocab.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_vocab(path: str) -> dict[int, bytes]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {int(k): _b64dec(v) for k, v in data.items()}

def save_merges(merges: List[Tuple[bytes, bytes]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for left, right in merges:
            f.write(f"{_b64enc(left)} {_b64enc(right)}\n")

def load_merges(path: str) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            merges.append((_b64dec(parts[0]), _b64dec(parts[1])))
    return merges

def save_tokens(ids: np.ndarray, output_path: str) -> None:
    mmap = np.memmap(output_path, dtype=ids.dtype, mode='w+', shape=ids.shape)
    mmap[:] = ids[:]
    mmap.flush()
    del mmap


# ==================== 带进度条的分词函数 ====================

def tokenize_file_with_progress(tokenizer, file_path, dtype, max_tokens=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    all_ids = []
    start_time = time.time()
    processed_lines = 0
    total_tokens = 0
    reached_limit = False
    limit_str = f" | 上限: {max_tokens:,}" if max_tokens else ""
    print(f"  开始编码 ({total_lines:,} 行{limit_str})...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids = tokenizer.encode(line)
            if max_tokens is not None:
                remaining = max_tokens - total_tokens
                if remaining <= 0:
                    reached_limit = True
                    break
                ids = ids[:remaining]

            all_ids.extend(ids)
            total_tokens += len(ids)
            processed_lines += 1

            if processed_lines % 5000 == 0 or processed_lines == total_lines or reached_limit:
                elapsed = time.time() - start_time
                speed = processed_lines / elapsed if elapsed > 0 else 0
                if max_tokens:
                    pct = min(total_tokens / max_tokens * 100, 100)
                    filled = int(30 * pct // 100)
                    bar = '█' * filled + '░' * (30 - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}% | "
                        f"tokens: {total_tokens:,}/{max_tokens:,} | {speed:.0f} lines/s"
                    )
                else:
                    pct = processed_lines / total_lines * 100
                    filled = int(30 * pct // 100)
                    bar = '█' * filled + '░' * (30 - filled)
                    eta = (total_lines - processed_lines) / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}% | "
                        f"{processed_lines:,}/{total_lines:,} | {speed:.0f} lines/s | ETA: {eta:.0f}s"
                    )
                sys.stdout.flush()

            if reached_limit:
                break

    print()
    if reached_limit:
        print(f"  ⚠️  已达到 {max_tokens:,} tokens 上限，提前终止编码")
    return np.array(all_ids, dtype=dtype)


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="BPE 一体化：训练 + 分词")
    parser.add_argument('--input', default='data/TinyStoriesV2-GPT4-train.txt')
    parser.add_argument('--val-input', default='data/TinyStoriesV2-GPT4-valid.txt')
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--special-tokens', nargs='*', default=['<|endoftext|>'])
    parser.add_argument('--output-dir', default='./bpe_output')
    parser.add_argument('--dtype', choices=['uint16', 'int32'], default='uint16')
    parser.add_argument('--max-tokens', type=int, default=40_000_000,
                        help="训练集最大 token 数（默认 40M）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    merges_path = os.path.join(args.output_dir, 'merges.txt')
    dtype = np.uint16 if args.dtype == 'uint16' else np.int32

    # ===== Step 1: 训练 BPE =====
    print("=" * 60)
    print("📌 Step 1/2: 训练 BPE 词表")
    print("=" * 60)
    print(f"  输入: {args.input}")
    print(f"  词汇表大小: {args.vocab_size}")
    print(f"  特殊标记: {args.special_tokens}\n")

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    print(f"  ✅ 词汇表已保存: {vocab_path} ({len(vocab):,} tokens)")
    print(f"  ✅ 合并规则已保存: {merges_path} ({len(merges):,} merges)")

    # ===== Step 2: 加载词表并分词 =====
    print(f"\n{'=' * 60}")
    print("📌 Step 2/2: Tokenization")
    print("=" * 60)

    # 重新从磁盘加载，确保与持久化格式完全一致
    loaded_vocab = load_vocab(vocab_path)
    loaded_merges = load_merges(merges_path)
    tokenizer = Tokenizer(vocab=loaded_vocab, merges=loaded_merges, special_tokens=args.special_tokens)
    print(f"  已加载词表: {len(loaded_vocab):,} tokens | {len(loaded_merges):,} merges")

    # 处理训练集
    print(f"\n  📄 训练集: {args.input}")
    train_ids = tokenize_file_with_progress(tokenizer, args.input, dtype, max_tokens=args.max_tokens)
    train_bin = os.path.join(args.output_dir, 'train.bin')
    save_tokens(train_ids, train_bin)
    print(f"  ✅ {train_bin}")
    print(f"     tokens: {len(train_ids):,} | 大小: {os.path.getsize(train_bin)/1024/1024:.2f} MB")

    # 处理验证集
    if args.val_input and os.path.exists(args.val_input):
        print(f"\n  📄 验证集: {args.val_input}")
        val_ids = tokenize_file_with_progress(tokenizer, args.val_input, dtype)
        val_bin = os.path.join(args.output_dir, 'val.bin')
        save_tokens(val_ids, val_bin)
        print(f"  ✅ {val_bin}")
        print(f"     tokens: {len(val_ids):,} | 大小: {os.path.getsize(val_bin)/1024/1024:.2f} MB")

    print(f"\n{'=' * 60}")
    print("🎉 全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()