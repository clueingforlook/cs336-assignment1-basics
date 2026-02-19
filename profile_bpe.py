import time
import os
import argparse
# 根据你的目录结构，从 cs336_basics 包中导入 train_bpe
# 确保在运行此脚本时，你在根目录下 (即能看到 cs336_basics 文件夹的地方)
from cs336_basics import bpe
from cs336_basics.bpe import Tokenizer

# 数据集路径配置
DATA_CONFIG = {
    "tinystory": {
        "train": os.path.join("data", "TinyStoriesV2-GPT4-train.txt"),
        "valid": os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"),
        "model_prefix": "tinystories_tokenizer"
    },
    "owt": {
        "train": os.path.join("data", "owt_train.txt"),
        "valid": os.path.join("data", "owt_valid.txt"),
        "model_prefix": "owt_tokenizer"
    }
}

def run_training(dataset_key: str, vocab_size: int = 10000):
    config = DATA_CONFIG[dataset_key]
    data_path = config["train"]
    model_prefix = config["model_prefix"]

    print(f"--- Start BPE Training on {dataset_key} ---")
    print(f"Reading from: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"Error: 找不到文件 {data_path}")
        print("请确认你是在项目根目录下运行，并且 data 文件夹里有这个文件。")
        return

    special_tokens = ["<|endoftext|>"]
    print(f"Target Vocab Size: {vocab_size}")
    
    start_time = time.time()
    
    # 运行训练
    vocab, merges = bpe.train_bpe(data_path, vocab_size, special_tokens)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # 保存模型
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(model_prefix)
    print(f"Tokenizer saved successfully to {model_prefix}_merges.json and {model_prefix}_vocab.json")
    
    # 简单验证
    print("\n--- Verifying ---")
    test_str = "Hello world I am Kevin and this is a test hahahahahahahahahhahah<|endoftext|>"
    ids = tokenizer.encode(test_str)
    decoded = tokenizer.decode(ids)
    print(f"Test String: {test_str}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: {decoded}")

def evaluate_file(tokenizer: Tokenizer, filepath: str):
    """
    流式读取文件，计算压缩率，并定期打印进度。
    这是处理大文件的标准范式。
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Skipping evaluation.")
        return

    print(f"\n--- Evaluating {filepath} ---")
    
    total_bytes = 0
    total_tokens = 0
    line_count = 0
    
    # 使用 'utf-8' 编码读取，因为 BPE 通常基于 UTF-8 字节
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            
            # 1. 计算原始字节数
            # 注意：必须 encode 回 bytes 才能计算准确的“字节数”
            # len(str) 只是字符数，len(str.encode('utf-8')) 才是字节数
            line_bytes = line.encode('utf-8')
            total_bytes += len(line_bytes)
            
            # 2. 计算 Token 数量
            ids = tokenizer.encode(line)
            total_tokens += len(ids)
            
            # 3. 监控进度 (每 10000 行)
            if line_count % 10000 == 1:
                # 实时计算当前的压缩率
                current_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
                print(f"[Line {line_count}] Tokens: {len(ids)}, Current Ratio: {current_ratio:.4f}X")
                
                # 优化显示：如果是空行，显示提示信息
                display_text = line[:50].strip()
                if not display_text:
                    display_text = "(Empty Line / Newline)"
                print(f"   Snippet (Text): {display_text}")
                print(f"   Snippet (IDs) : {ids[:10]}...")

    # 4. 最终结果
    final_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    print(f"\n=== Evaluation Result ===")
    print(f"Total Lines: {line_count}")
    print(f"Total Bytes: {total_bytes}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Compression Ratio: {final_ratio:.4f}X (Bytes/Token)")
    print("=========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE Tokenizer Training & Evaluation Tool")
    
    # 1. 运行模式：训练 或 测试
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], 
                        help="运行模式: 'train' (训练模型) 或 'test' (计算压缩率)")
    
    # 2. 数据集选择
    parser.add_argument("--data", type=str, required=True, choices=["tinystory", "owt"],
                        help="数据集: 'tinystory' 或 'owt'")
    
    # 3. 可选参数：词表大小 (仅训练用)
    parser.add_argument("--vocab_size", type=int, default=10000, 
                        help="词表大小 (默认: 10000)")

    # 4. 可选参数：指定模型前缀 (如果你想用 owt 数据集测试 tinystory 的模型，可以用这个参数覆盖)
    parser.add_argument("--model_prefix", type=str, default=None,
                        help="指定要加载/保存的模型文件前缀 (默认为数据集对应的名称)")

    args = parser.parse_args()

    # 获取当前配置
    current_config = DATA_CONFIG[args.data]
    # 如果用户没有指定 model_prefix，就用数据集默认的
    model_prefix = args.model_prefix if args.model_prefix else current_config["model_prefix"]

    if args.mode == "train":
        # 训练模式
        # 注意：owt_train.txt 很大，请确保 cs336_basics.bpe.train_bpe 已经实现了之前讨论的内存优化逻辑
        print(f"Mode: TRAIN | Dataset: {args.data} | Vocab Size: {args.vocab_size}")
        
        # 临时修改配置里的 model_prefix 以便 run_training 使用
        current_config["model_prefix"] = model_prefix 
        run_training(args.data, args.vocab_size)

    elif args.mode == "test":
        # 测试模式
        vocab_file = f"{model_prefix}_vocab.json"
        merges_file = f"{model_prefix}_merges.json"
        test_file = current_config["valid"]

        print(f"Mode: TEST | Dataset: {args.data}")
        print(f"Loading Tokenizer from: {vocab_file}, {merges_file}")
        print(f"Evaluating on: {test_file}")

        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            print(f"Error: 找不到模型文件 {vocab_file}。请先运行 --mode train 进行训练。")
        else:
            # 加载 Tokenizer
            tok = Tokenizer.from_files(vocab_file, merges_file, ["<|endoftext|>"])
            # 运行评估
            evaluate_file(tok, test_file)
# import time
# import os
# # 根据你的目录结构，从 cs336_basics 包中导入 train_bpe
# # 确保在运行此脚本时，你在根目录下 (即能看到 cs336_basics 文件夹的地方)
# from cs336_basics import bpe
# from cs336_basics.bpe import Tokenizer

# def run_training():
#     print("--- Start BPE Training on TinyStoriesV2-GPT4-train ---")
    
#     # 1. 设定数据路径
#     # 根据你的截图，数据在 data 文件夹下
#     data_path = os.path.join("data", "TinyStoriesV2-GPT4-train.txt")
    
#     # 检查文件是否存在
#     if not os.path.exists(data_path):
#         print(f"Error: 找不到文件 {data_path}")
#         print("请确认你是在项目根目录下运行，并且 data 文件夹里有这个文件。")
#         return

#     # 2. 设定参数 (作业要求 vocab_size=10000)
#     vocab_size = 10000
#     special_tokens = ["<|endoftext|>"]
    
#     print(f"Reading from: {data_path}")
#     print(f"Target Vocab Size: {vocab_size}")
    
#     start_time = time.time()
    
#     # 3. 运行训练
#     # 注意：train_bpe 现在应该已经在 cs336_basics/bpe.py 里了
#     vocab, merges = bpe.train_bpe(data_path, vocab_size, special_tokens)
    
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Training completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
#     # 4. 保存模型 (JSON)
#     tokenizer = Tokenizer(vocab, merges, special_tokens)
#     tokenizer.save("tinystories_tokenizer")
#     print("Tokenizer saved successfully to tinystories_tokenizer_merges.json and tinystories_tokenizer_vocab.json")
#     # 5. 简单验证
#     print("\n--- Verifying ---")
#     # 使用保存的组件重新加载一个 Tokenizer 进行测试
#     test_str = "Hello world I am Kevin and this is a test hahahahahahahahahhahah<|endoftext|>"
#     ids = tokenizer.encode(test_str)
#     decoded = tokenizer.decode(ids)
#     print(f"Test String: {test_str}")
#     print(f"Encoded IDs: {ids}")
#     print(f"Decoded: {decoded}")

# def evaluate_file(tokenizer: Tokenizer, filepath: str):
#     """
#     流式读取文件，计算压缩率，并定期打印进度。
#     这是处理大文件的标准范式。
#     """
#     if not os.path.exists(filepath):
#         print(f"File {filepath} not found. Skipping evaluation.")
#         return

#     print(f"\n--- Evaluating {filepath} ---")
    
#     total_bytes = 0
#     total_tokens = 0
#     line_count = 0
    
#     # 使用 'utf-8' 编码读取，因为 BPE 通常基于 UTF-8 字节
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             line_count += 1
            
#             # 1. 计算原始字节数
#             # 注意：必须 encode 回 bytes 才能计算准确的“字节数”
#             # len(str) 只是字符数，len(str.encode('utf-8')) 才是字节数
#             line_bytes = line.encode('utf-8')
#             total_bytes += len(line_bytes)
            
#             # 2. 计算 Token 数量
#             # 这里调用 encode(str) 而不是 encode_iterable
#             # 因为我们需要保留行的概念
#             ids = tokenizer.encode(line)
#             total_tokens += len(ids)
            
#             # 3. 监控进度 (每 10000 行)
#             if line_count % 10000 == 1:
#                 # 实时计算当前的压缩率
#                 current_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
#                 print(f"[Line {line_count}] Tokens: {len(ids)}, Current Ratio: {current_ratio:.4f}X")
#                 print(f"   Snippet (Text): {line[:50].strip()}...")
#                 print(f"   Snippet (IDs) : {ids[:10]}...")

#     # 4. 最终结果
#     final_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
#     print(f"\n=== Evaluation Result ===")
#     print(f"Total Lines: {line_count}")
#     print(f"Total Bytes: {total_bytes}")
#     print(f"Total Tokens: {total_tokens}")
#     print(f"Compression Ratio: {final_ratio:.4f}X (Bytes/Token)")
#     print("=========================")


# if __name__ == "__main__":
#     """
#     用tinystories训练好的tokenizer，encode owt_valid.txt，计算压缩率
#     """
#     # 模拟运行测试 (用户可以将这里替换为真实的文件路径)
#     tok = Tokenizer.from_files("tinystories_tokenizer_vocab.json", 
#                                "tinystories_tokenizer_merges.json",
#                                ["<|endoftext|>"])
#     test_file_path = os.path.join("data","owt_valid.txt")
#     evaluate_file(tok, test_file_path)