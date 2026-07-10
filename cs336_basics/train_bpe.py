import os
import regex as re
from collections import defaultdict
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries
from tqdm import tqdm


# GPT-2 预分词正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# ==================== Utils ====================

def split_by_special(text, special_tokens, drop_special=True):
    """
    按特殊标记(special tokens)分割文本
    
    Args:
        text: 输入文本
        special_tokens: 特殊标记列表，如 ['<|endoftext|>', '<|pad|>']
        drop_special: 是否丢弃特殊标记(True表示丢弃, False表示保留)
    
    Returns:
        分割后的文本片段列表
    """
    if not special_tokens:
        return [text]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"
    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]


def word2bytes(word):
    """
    将单词字符串转换为字节元组
    
    Args:
        word: 单词字符串
    
    Returns:
        字节元组, 每个元素是一个单字节的bytes对象
        例如: "hello" -> (b'h', b'e', b'l', b'l', b'o')
    """
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)


def count_word(text):
    """
    使用GPT-2模式分割文本并统计词频
    
    Args:
        text: 输入文本
    
    Returns:
        defaultdict: 键为单词的字节元组，值为出现频率
    """
    word_cnt = defaultdict(int)
    # 使用正则表达式查找所有匹配的token
    for m in PAT.finditer(text):
        word = m.group(0)   # 正则表达式匹配到的完整字符串
        word_bytes = word2bytes(word)
        # 只统计长度>=2的单词
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt


def merge_dicts(dicts):
    """
    合并多个词频字典
    
    Args:
        dicts: 词频字典列表
    
    Returns:
        合并后的字典
    """
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged


# ==================== 并行化处理 ====================

def _process_file_chunk(file_path, start, end, special_tokens):
    """
    子进程任务：
    1. 安全读取指定字节区间
    2. 解码并按 special tokens 切分
    3. 对每个切分块执行 pre-tokenization 并统计词频
    """
    # 按需读取字节区间
    with open(file_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")

    # 在 chunk 内部按 special tokens 切分
    chunks = split_by_special(text, special_tokens, drop_special=True)

    # 对每个子块统计词频并合并
    local_counters = [count_word(chunk) for chunk in chunks]
    return merge_dicts(local_counters)


def _get_safe_boundaries(input_path, num_chunks, special_tokens):
    """计算安全的并行切分边界"""
    file_size = os.path.getsize(input_path)

    if not special_tokens:
        # 无 special token 时退化为均匀切分
        step = max(1, file_size // num_chunks)
        bounds = [i * step for i in range(num_chunks + 1)]
        bounds[-1] = file_size
        return bounds

    # 可动态选取第一个 special token 作为边界锚点
    # boundary_token = special_tokens[0].encode("utf-8")
    boundary_token = b"<|endoftext|>"

    with open(input_path, "rb") as f:
        return find_chunk_boundaries(f, num_chunks, boundary_token)


# ==================== BPE 训练主逻辑 ====================

def count_pair(word_cnt):
    """
    统计所有相邻字符对(pair)的出现频率
    
    Args:
        word_cnt: 词频字典 {单词字节元组: 频率}
    
    Returns:
        defaultdict: {字符对: 频率}
        例如: { (b'a', b'b'): 10, (b'b', b'c'): 5 }
    """
    pair_cnt = defaultdict(int)
    for word_bytes, cnt in word_cnt.items():
        # 遍历单词中所有相邻的字符对
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_cnt[pair] += cnt
    return pair_cnt


def get_max_pair(pair_cnt):
    """
    找到频率最高的相邻字节对；若频率并列，则按字节字典序取最大者
    
    Args:
        pair_cnt: 字符对频率字典
    
    Returns:
        tuple: (字符对, 频率) 中的字符对
    """
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))
    return max_pair


def get_basic_vocab(special_tokens):
    """
    构建基础词汇表
    
    词汇表包含：
    1. 256个单字节token (0-255)
    2. 特殊标记token (从256开始分配)
    
    Args:
        special_tokens: 特殊标记列表
    
    Returns:
        dict: {token_id: token_bytes}
    """
    vocab = {token: bytes([token]) for token in range(256)}
    for i, token in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes, merge):
    """
    对单个单词应用一次BPE合并操作
    
    将单词中所有出现的 (merge[0] + merge[1]) 替换为 merge[0] + merge[1]
    
    Args:
        word_bytes: 单词的字节元组
        merge: 要合并的字符对 (byte1, byte2)
    
    Returns:
        tuple: 合并后的新单词字节元组
    """
    merged = merge[0] + merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        if (i < len(word_bytes) - 1 and
                word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]):
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)


def update_cnt(word_cnt, pair_cnt, merge_pair):
    """
    更新词频和字符对频率
    
    Args:
        word_cnt: 当前词频字典
        pair_cnt: 当前字符对频率字典
        merge_pair: 要合并的字符对
    
    Returns:
        tuple: (更新后的词频字典, 更新后的字符对频率字典)
    """
    new_word_cnt = defaultdict(int)             # 新建词频字典, 若出现 new_word, 自动初始化为0
    new_pair_cnt = defaultdict(int, pair_cnt)   # 复制 pair_cnt

    for word_bytes, cnt in word_cnt.items():
        # 提取当前 word 的所有相邻字符对
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # 跳过不包含 merge_pair 的 word
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes] += cnt
            continue
        
        # --- 以下仅处理包含 merge_pair 的单词 ---

        # 执行合并
        new_word = apply_merge(word_bytes, merge_pair)
        new_word_cnt[new_word] += cnt

        # 首先移除 old_pairs 的贡献
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]
        # 再添加 new_pairs 的贡献
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt, new_pair_cnt


def train_bpe(input_path, vocab_size, special_tokens):
    num_processes = os.cpu_count() or 4

    # 1. 使用 pretokenization_example.py 的方法切分文件
    boundaries = _get_safe_boundaries(input_path, num_processes, special_tokens)
    n_chunks = len(boundaries) - 1
    print(f"[BPE] Splitting file into {n_chunks} safe chunks, using {num_processes} workers...")

    # 2. 并行预分词 + 词频统计
    tasks = [
        (input_path, s, e, special_tokens)
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        word_dicts = pool.starmap(_process_file_chunk, tasks)
    
    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    # 3. BPE 合并循环
    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size

    merges = []
    for i in tqdm(range(n_merges), desc="BPE Merges", unit="merge"):
        max_pair = get_max_pair(pair_cnt)
        vocab[base_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        word_cnt, pair_cnt = update_cnt(word_cnt, pair_cnt, max_pair)
    return vocab, merges