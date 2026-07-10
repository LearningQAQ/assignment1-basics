# import regex as re
# from typing import Iterator, Iterable
# import json
# from .train_bpe import split_by_special, word2bytes

# PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# def split_to_words(text):
#     """拆分文本"""
#     return PAT.findall(text)

# def apply_merges(word_bytes, merges_set, vocab_to_id):
#     """
#     对字节序列反复应用BPE合并规则, 直至无法继续合并
#     每次选择词表中序号最小的可合并对进行合并
#     """
#     word_bytes = list(word_bytes)
    
#     while True:
#         min_token_id = float('inf')
#         best_pair_idx = -1
#         merged = None

#         # 遍历所有相邻对，找词表中序号最小的可合并对
#         for i in range(len(word_bytes) - 1):
#             pair = (word_bytes[i], word_bytes[i + 1])
#             if pair in merges_set:
#                 combined = pair[0] + pair[1]
#                 token_id = vocab_to_id.get(combined)
#                 if token_id is not None and token_id < min_token_id:
#                     min_token_id = token_id
#                     best_pair_idx = i
#                     merged = combined

#         # 没有可合并对则退出
#         if best_pair_idx == -1:
#             break

#         # 执行选中的合并操作
#         word_bytes = (
#             word_bytes[:best_pair_idx]
#             + [merged]
#             + word_bytes[best_pair_idx + 2:]
#         )

#     return tuple(word_bytes)

# def encode_merged(text, merges, vocab_to_id):
#     """
#     对文本进行编码：
#     1. 分词
#     2. 将每个词转为UTF-8字节
#     3. 应用BPE合并
#     4. 查表得到token ID
#     """
#     word_list = split_to_words(text)
#     tokens=[]
#     for word in word_list:
#         word_bytes = word2bytes(word)
#         merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_id)
#         tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
#     return tokens



# class Tokenizer:
#     def __init__(
#         self,
#         vocab: dict[int, bytes],
#         merges: list[tuple[bytes, bytes]],
#         special_tokens: list[str] | None = None,
#     ):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens if special_tokens else []
#         self.special_tokens_bytes = [t.encode('utf-8') for t in self.special_tokens]
#         # 构建反向映射：字节序列 -> token ID
#         self.vocab_to_id={v: k for k, v in vocab.items()}

        
#     @classmethod
#     def from_files(
#         cls,
#         vocab_filepath: str,
#         merges_filepath: str,
#         special_tokens: list[str] | str | None = None,
#     ) -> "Tokenizer":
        
#         """从文件加载 BPE 词表与合并规则，构造 Tokenizer 实例。"""

#         # 加载词表：JSON 键转 int，值用 latin-1 还原为原始字节
#         with open(vocab_filepath, "r", encoding="utf-8") as vf:
#             vocab_data = json.load(vf)
#             vocab = {
#                 int(k): bytes(v, "latin-1") if isinstance(v, str) else bytes(v)
#                 for k, v in vocab_data.items()
#             }

#         # 加载合并规则：过滤注释/空行，文本 token 对编码为字节对
#         with open(merges_filepath, "r", encoding="utf-8") as mf:
#             lines = mf.readlines()
#             merge_pairs = [
#                 tuple(line.strip().split())
#                 for line in lines
#                 if line.strip() and not line.startswith("#")
#             ]
#             merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merge_pairs]

#         return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
#     def encode(self, text: str) -> list[int]:
#         chunks = split_by_special(text, self.special_tokens, drop_special=False)
#         tokens = []
#         for chunk in chunks:
#             # special_tokens 直接编码后查表获取ID, 不经过 PAT，直接作为一个完整 token
#             if self.special_tokens and chunk in self.special_tokens:
#                 tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
#             # 普通 word 走 BPE 合并流程，返回多个 ID
#             else:
#                 tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
#         return tokens

#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         """对可迭代的文本流进行编码, 惰性产出token ID"""
#         # 逐块读取
#         for chunk in iterable:
#             # encode 返回的 list[int] 逐个展开为独立的 int 产出
#             yield from self.encode(chunk)

#     def decode(self, ids: list[int]) -> str:
#         """将token ID列表还原为文本"""
#         return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')

import heapq
import json
import regex as re
from typing import Iterator, Iterable
from .train_bpe import split_by_special, word2bytes

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def split_to_words(text):
    """拆分文本"""
    return PAT.findall(text)


# ==================== O(n log n) BPE 内部实现 ====================

class _Node:
    """双向链表节点（内部使用）"""
    __slots__ = ('val', 'prev', 'next', 'alive')

    def __init__(self, val: bytes):
        self.val = val
        self.prev: '_Node | None' = None
        self.next: '_Node | None' = None
        self.alive = True


def _apply_merges_fast(word_bytes: tuple[bytes, ...], merges_set: set[tuple[bytes, bytes]], vocab_to_id: dict[bytes, int]) -> tuple[bytes, ...]:
    """O(n log n) BPE 合并（内部函数，不暴露给外部）"""
    if len(word_bytes) <= 1:
        return word_bytes

    # 1. 构建双向链表
    nodes = [_Node(b) for b in word_bytes]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
        nodes[i + 1].prev = nodes[i]

    # 2. 初始化最小堆: (token_id, counter, left_node)
    heap: list[tuple[int, int, _Node]] = []
    counter = 0
    for node in nodes[:-1]:
        pair = (node.val, node.next.val)
        if pair in merges_set:
            combined = node.val + node.next.val
            tid = vocab_to_id.get(combined)
            if tid is not None:
                heapq.heappush(heap, (tid, counter, node))
                counter += 1

    # 3. 贪心合并
    while heap:
        tid, _, left_node = heapq.heappop(heap)

        # 惰性跳过已失效条目
        if not left_node.alive or left_node.next is None or not left_node.next.alive:
            continue
        right_node = left_node.next
        combined = left_node.val + right_node.val
        actual_tid = vocab_to_id.get(combined)
        if actual_tid != tid:
            continue

        # 执行合并：更新左节点，标记右节点死亡
        left_node.val = combined
        right_node.alive = False
        left_node.next = right_node.next
        if right_node.next is not None:
            right_node.next.prev = left_node

        # 新左邻居对入堆
        if left_node.prev is not None and left_node.prev.alive:
            p = (left_node.prev.val, left_node.val)
            if p in merges_set:
                c = left_node.prev.val + left_node.val
                t = vocab_to_id.get(c)
                if t is not None:
                    heapq.heappush(heap, (t, counter, left_node.prev))
                    counter += 1

        # 新右邻居对入堆
        if left_node.next is not None and left_node.next.alive:
            p = (left_node.val, left_node.next.val)
            if p in merges_set:
                c = left_node.val + left_node.next.val
                t = vocab_to_id.get(c)
                if t is not None:
                    heapq.heappush(heap, (t, counter, left_node))
                    counter += 1

    # 4. 收集存活节点
    result = []
    node = nodes[0]
    while node is not None:
        if node.alive:
            result.append(node.val)
        node = node.next
    return tuple(result)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [t.encode('utf-8') for t in self.special_tokens]
        # 构建反向映射：字节序列 -> token ID
        self.vocab_to_id = {v: k for k, v in vocab.items()}
        # 【新增】预构建 merges_set，供 O(n log n) 合并使用
        self._merges_set = set(merges)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | str | None = None,
    ) -> "Tokenizer":
        """从文件加载 BPE 词表与合并规则，构造 Tokenizer 实例。"""
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)
            vocab = {
                int(k): bytes(v, "latin-1") if isinstance(v, str) else bytes(v)
                for k, v in vocab_data.items()
            }

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            lines = mf.readlines()
            merge_pairs = [
                tuple(line.strip().split())
                for line in lines
                if line.strip() and not line.startswith("#")
            ]
            merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                word_list = split_to_words(chunk)
                for word in word_list:
                    wb = word2bytes(word)
                    merged = _apply_merges_fast(wb, self._merges_set, self.vocab_to_id)
                    tokens.extend(self.vocab_to_id[b] for b in merged)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """对可迭代的文本流进行编码, 惰性产出token ID"""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """将token ID列表还原为文本"""
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8', errors='replace')