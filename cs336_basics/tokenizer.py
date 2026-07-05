import regex as re
from typing import Iterator, Iterable
import json
from .train_bpe import split_by_special, word2bytes

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def split_to_words(text):
    """拆分文本"""
    return PAT.findall(text)

def apply_merges(word_bytes, merges_set, vocab_to_id):
    """
    对字节序列反复应用BPE合并规则, 直至无法继续合并
    每次选择词表中序号最小的可合并对进行合并
    """
    word_bytes = list(word_bytes)
    
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        # 遍历所有相邻对，找词表中序号最小的可合并对
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        # 没有可合并对则退出
        if best_pair_idx == -1:
            break

        # 执行选中的合并操作
        word_bytes = (
            word_bytes[:best_pair_idx]
            + [merged]
            + word_bytes[best_pair_idx + 2:]
        )

    return tuple(word_bytes)

def encode_merged(text, merges, vocab_to_id):
    """
    对文本进行编码：
    1. 分词
    2. 将每个词转为UTF-8字节
    3. 应用BPE合并
    4. 查表得到token ID
    """
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes = word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens



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
        self.vocab_to_id={v: k for k, v in vocab.items()}

        
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | str | None = None,
    ) -> "Tokenizer":
        
        """从文件加载 BPE 词表与合并规则，构造 Tokenizer 实例。"""

        # 加载词表：JSON 键转 int，值用 latin-1 还原为原始字节
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)
            vocab = {
                int(k): bytes(v, "latin-1") if isinstance(v, str) else bytes(v)
                for k, v in vocab_data.items()
            }

        # 加载合并规则：过滤注释/空行，文本 token 对编码为字节对
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
            # special_tokens 直接编码后查表获取ID, 不经过 PAT，直接作为一个完整 token
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            # 普通 word 走 BPE 合并流程，返回多个 ID
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """对可迭代的文本流进行编码, 惰性产出token ID"""
        # 逐块读取
        for chunk in iterable:
            # encode 返回的 list[int] 逐个展开为独立的 int 产出
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """将token ID列表还原为文本"""
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')