import os
import regex as re
import json
import gc
import multiprocessing
from typing import Iterable, Iterator, List, Optional, BinaryIO, Tuple, Dict
from collections import Counter, defaultdict

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ==========================================
# Section 2.5: BPE Training Helpers
# ==========================================

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def _parallel_count_worker(input_path: str, start: int, end: int, special_tokens: list) -> Counter:
    """
    多进程 Worker：读取文件指定区间，执行预分词并统计词频。
    """
    try:
        pattern = re.compile(GPT2_SPLIT_PATTERN)
        counts = Counter()
        
        with open(input_path, "rb") as f:
            f.seek(start)
            # 优化：不一次性读取超大块，防止峰值过高。
            # 但为了正则匹配正确性，这里仍需读取完整块，我们通过减少 worker 数量来平衡内存。
            chunk_bytes = f.read(end - start)
            
        # 解码时忽略错误
        # 优化：解码后立即释放 chunk_bytes 内存
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
        del chunk_bytes
        gc.collect() # 强制垃圾回收

        # 针对 Windows 的换行符修正
        if "\r\n" in chunk_str:
            chunk_str = chunk_str.replace("\r\n", "\n")
        
        if special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
            segments = re.split(special_pattern, chunk_str)
        else:
            segments = [chunk_str]
        
        # 释放 chunk_str，因为 segments 已经持有了切分后的字符串引用
        # (注意 re.split 可能会返回原始字符串的视图或副本，这里显式删除 chunk_str 帮助不大，
        #  但在 segments 生成后 chunk_str 确实不再需要)
        del chunk_str

        for segment in segments:
            if not segment:
                continue
            if special_tokens and segment in special_tokens:
                continue
                
            for word in pattern.findall(segment):
                counts[tuple(word.encode("utf-8"))] += 1
                
        return counts
    except MemoryError:
        print(f"Worker OOM processing chunk {start}-{end}. Try reducing num_processes.")
        raise

def _merge_ids(ids: tuple, pair: tuple, new_token: int) -> tuple:
    """
    辅助函数：在 ids 序列中合并指定的 pair
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_token)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return tuple(new_ids)

# ==========================================
# Section 2.5: Main BPE Training Function
# ==========================================

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None):
    """
    Section 2.5: 训练 BPE 分词器，返回 vocab 和 merges
    """
    # 1. 并行化预分词 (Parallel Pre-tokenization)
    # 设定每个块的目标大小为 20MB (这是一个对内存非常友好的数值)
    # 获取文件大小
    file_size = os.path.getsize(input_path)
    # Python 处理 100MB 文本大约需要 750MB-1000MB 内存，(防止 OOM)
    target_chunk_size = 100 * 1024 * 1024
    # 计算需要切分成多少块
    num_chunks = max(1, file_size // target_chunk_size)

    """ --- 关键修复：计算安全的进程数 ---
    # 不要用 num_chunks，也不要单纯用 cpu_count
    # 而是取两者较小值，并且在 Windows 上强制限制在 60 以内
    """
    # 1. 基础值：CPU 核心数 - 1
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # 2. Windows 限制修复：WaitForMultipleObjects 最多支持 64 个句柄
    if os.name == 'nt' and num_processes > 60:
        print(f"Windows detected with high core count ({num_processes}). Capping processes to 60.")
        num_processes = 60
        
    print(f"Training Config:")
    print(f"  - File Size: {file_size/1024/1024/1024:.2f} GB")
    print(f"  - Target Chunk Size: {target_chunk_size/1024/1024} MB")
    print(f"  - Total Tasks (Chunks): {num_chunks}")
    print(f"  - Active Workers (Processes): {num_processes}")
    
    # 严格按照 2.5 节文档说明，使用 <|endoftext|> 作为切分 token
    split_token = b"<|endoftext|>"
    
    # print(f"Finding chunk boundaries for {num_processes} processes...")
    with open(input_path, "rb") as f:
        # 这里传入 num_chunks 而不是 num_processes
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)
        
    args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    print("Pre-tokenizing and counting words in parallel...")
    word_counts = Counter()
    
    # --- 关键修复：Pool 使用 num_processes 而不是 len(args) ---
    # len(args) 是任务数 (几百个)，num_processes 是工人数 (60个)
    # starmap 会自动让工人们排队领取任务
    with multiprocessing.Pool(num_processes) as pool:
        # 添加 chunksize=1 可以让进度更平滑
        for i, res in enumerate(pool.starmap(_parallel_count_worker, args)):
            word_counts.update(res)
            if (i + 1) % 10 == 0 or (i + 1) == len(args):
                print(f"  Processed chunk {i+1}/{len(args)}")
            
    word_counts = dict(word_counts)
    
    # 2. 建立增量更新所需的“倒排索引” (Optimizing the merging step)
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    
    for word, freq in word_counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # 3. 开始 BPE 合并循环
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []  # 保存的是 list[tuple[bytes, bytes]]
    
    num_special = len(special_tokens) if special_tokens else 0
    num_merges = vocab_size - 256 - num_special
    
    print(f"Starting {num_merges} BPE merges...")
    for i in range(num_merges):
        if not pair_counts:
            break
            
        # Tie-breaking: 频率相同时，选字典序大的 pair (Lexicographically greater)
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))
        
        idx = 256 + i
        vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        # --- 增量更新逻辑 (Incremental Updates) ---
        words_to_process = list(pair_to_words[best_pair])
        for old_word in words_to_process:
            freq = word_counts[old_word]
            
            # (a) 移除旧单词中的所有 pair 计数
            for j in range(len(old_word) - 1):
                p = (old_word[j], old_word[j+1])
                pair_counts[p] -= freq
                if old_word in pair_to_words[p]:
                    pair_to_words[p].remove(old_word)
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                    
            # (b) 生成新单词
            new_word = _merge_ids(old_word, best_pair, idx)
            
            # (c) 替换 word_counts 中的记录
            del word_counts[old_word]
            word_counts[new_word] = word_counts.get(new_word, 0) + freq
            
            # (d) 添加新单词带来的新 pair 计数
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j+1])
                pair_counts[p] += freq
                pair_to_words[p].add(new_word)
                
        if (i + 1) % 1000 == 0:
            print(f"Merged {i+1}/{num_merges}: {vocab[best_pair[0]]} + {vocab[best_pair[1]]} -> ID {idx}")

    # 4. 追加特殊 Token 
    # 作业指出它们不参与合并，只在最后占据固定的 token ID
    current_id = 256 + len(merges)
    if special_tokens:
        for st in special_tokens:
            vocab[current_id] = st.encode('utf-8')
            current_id += 1
            
    return vocab, merges


# ==========================================
# Section 2.6: Tokenizer Class
# ==========================================
class Tokenizer:
    def __init__(
            self, vocab: dict[int:bytes],
            merges: list[tuple[bytes,bytes]], 
            special_tokens: list[str] = None
            ):
        self.vocab = vocab  # int : Bytes
        self.bytes_to_id = {v: k for k,v in self.vocab.items()}    # Bytes : int
        self.merges = {}    # (int, int) : int
        self.special_tokens = {}    # str : int
        if special_tokens:
            # 确保 special_tokens 有对应的 ID
            for st in special_tokens:
                st_bytes = st.encode('utf-8')
                if st_bytes in self.bytes_to_id:
                    self.special_tokens[st] = self.bytes_to_id[st_bytes]
                else:
                    # 这通常不应该发生，除非 vocab 和 special_tokens 不匹配
                    pass
        # ============ 修改开始 ============
        # --- 优化：在初始化时预编译特殊 Token 正则 ---
        # 1. 必须按长度降序排列，避免短 token 抢先匹配长 token 的前缀
        # 2. 预编译可以避免每次 encode 都重新构建正则，提高性能
        if self.special_tokens:
            # 关键点：这里用了 sorted(..., key=len, reverse=True)
            # 确保 "<|endoftext|><|endoftext|>" 排在 "<|endoftext|>" 前面
            sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(k) for k in sorted_specials) + ")"
            )
        # ============ 修改结束 ============
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)

        current_id = 256
        for p1_byte, p2_byte in merges:
            idx1 = self.bytes_to_id[p1_byte]
            idx2 = self.bytes_to_id[p2_byte]
            self.merges[(idx1, idx2)] = current_id
            current_id += 1

        
    @classmethod
    def from_files(cls, 
                   vocab_filepath: str, 
                   merges_filepath: str, 
                   special_tokens:list[str]=None):
        """
        Class method that constructs and return a Tokenizer from a serialized 
        vocabulary and list of merges(in the same format that your BPE training
        code output) and (optionally) a list of special tokens.
        """
        # vocab: dict[int:bytes]
        with open(vocab_filepath, "r") as f:
            data = json.load(f)
            # 检查是否是嵌套结构（save方法生成的格式）
            if isinstance(data, dict) and "vocab" in data:
                vocab_data = data["vocab"]
                # 如果调用时没传 special_tokens，尝试从文件中读取
                if special_tokens is None:
                    special_tokens = data.get("special_tokens")
            else:
                # 兼容旧格式：整个文件就是 vocab
                vocab_data = data
            vocab = {int(idx): bytes.fromhex(hex_str) for idx,hex_str in vocab_data.items()}

        # merges:list[tuple[bytes,bytes]]
        with open(merges_filepath, "r") as f:
            # merges文件格式: [["byte1_hex", "byte2_hex"], ...] 直接就是 merges 列表
            merges_data = json.load(f)
            merges = [(bytes.fromhex(p1_hex), bytes.fromhex(p2_hex)) for p1_hex, p2_hex in merges_data]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]: 
        """
        Encode an input text into a sequence of token IDs.
        """
        # 1.预分词
        if self.special_tokens:
            segments = re.split(self.special_pattern, text)
            final_chunks = []   # list[str]
            for seg in segments:
                if not seg:
                    continue
                if seg in self.special_tokens:
                    final_chunks.append(seg)  # 保留特殊 token 原样
                else:
                    final_chunks.extend(self.pattern.findall(seg))
        else:
            final_chunks = self.pattern.findall(text)

        # 2. 将预分词结果转换为 token ID
        ids = []
        for chunk in final_chunks:
            # chunk: str
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue
            
            tokens = list(bytes(chunk.encode("utf-8"))) # tokens: bytes
            tokens_ids = [self.bytes_to_id[bytes([b])] for b in tokens]
            while len(tokens_ids) > 1:
                stats = {}  # (idx1, idx2) : new_id
                for pair in zip(tokens_ids, tokens_ids[1:]):
                    if pair in self.merges:
                        stats[pair] = self.merges[pair]
                if not stats:   # 整个单词都没有可合并的 pair 
                    break
                # 寻找拥有最小 ID (最早合并) 的 pair
                best_pair = min(stats.keys(), key=lambda p: stats[p]) 
                new_token = stats[best_pair]

                # 执行替换
                new_ids = []
                i = 0
                while i < len(tokens_ids):
                    if i < len(tokens_ids) - 1 and tokens_ids[i] == best_pair[0] and tokens_ids[i+1] == best_pair[1]:
                        new_ids.append(new_token)
                        i += 2
                    else:
                        new_ids.append(tokens_ids[i])
                        i += 1
                tokens_ids = new_ids
            
            ids.extend(tokens_ids)
        return ids
    
    def decode(self, ids: list[int]) -> str:
        res = []
        # 反转 special_tokens 得到 id -> str
        id_to_special = {v: k for k, v in self.special_tokens.items()}
        
        for t in ids:
            if t in id_to_special:
                res.append(id_to_special[t].encode('utf-8'))
            elif t in self.vocab:
                res.append(self.vocab[t])
            else:
                # Fallback, 虽然正常训练的 BPE 不应该走到这里（除非 ID 越界）
                res.append(bytes([t]))
        
        return b"".join(res).decode('utf-8', errors='replace')
    def save(self, file_prefix: str):
        vocab_filepath = f"{file_prefix}_vocab.json"
        merges_filepath = f"{file_prefix}_merges.json"
        
        vocab_data = {
            "vocab": {idx: token.hex() for idx, token in self.vocab.items()},
            "special_tokens": list(self.special_tokens.keys())
        }
        with open(vocab_filepath, "w") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        merges_list = []
        # 还原回 bytes 形式保存
        # self.merges: (id1, id2) -> new_id
        # 为了保持顺序，按 new_id 排序
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (id1, id2), _ in sorted_merges:
            merges_list.append([self.vocab[id1].hex(), self.vocab[id2].hex()])
            
        with open(merges_filepath, "w") as f:
            json.dump(merges_list, f, ensure_ascii=False)
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            for tid in self.encode(text_chunk):
                yield tid
