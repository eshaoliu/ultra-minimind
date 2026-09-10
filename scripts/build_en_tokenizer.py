"""
在英文语料（parquet 目录，Nemotron-CC-Math-v1 格式，text 列）上训练 byte-level BPE 分词器。

用法:
    python scripts/build_en_tokenizer.py \
        --data_dir /root/gpufree-data/dataset/Nemotron-CC-Math-v1/4plus \
        --output_dir ../model/tokenizer_en \
        --vocab_size 32768 \
        --max_samples 2000000

说明:
    - 从每个 shard 顺序抽样，直到收集够 --max_samples 条文档（或读完全部 shard）
    - byte-level BPE（GPT-2 风格 pre-tokenizer），天然覆盖 LaTeX/公式/代码等任意字符，无需 unk
    - 特殊 token 顺序固定: <|endoftext|>(pad/unk) -> <|im_start|>(bos) -> <|im_end|>(eos)
      其后为 think/tool 标签，与 MiniMind 的 chat template 及 MiniMindConfig 默认 bos=1/eos=2 对齐
    - 产出 transformers 可加载的 fast tokenizer 目录（tokenizer.json + tokenizer_config.json）
"""
import os
import sys
import glob
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    parser = argparse.ArgumentParser(description="Train an English byte-level BPE tokenizer")
    parser.add_argument("--data_dir", type=str, default="/root/gpufree-data/dataset/Nemotron-CC-Math-v1/4plus",
                        help="parquet 数据目录（或其上级目录，自动递归匹配 *.parquet）")
    parser.add_argument("--output_dir", type=str, default="../model/tokenizer_en", help="分词器保存目录")
    parser.add_argument("--vocab_size", type=int, default=32768, help="词表大小（含 256 字节基础 token）")
    parser.add_argument("--max_samples", type=int, default=2_000_000, help="用于训练的文档条数上限")
    parser.add_argument("--min_doc_chars", type=int, default=100, help="过滤过短文档")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from transformers import PreTrainedTokenizerFast

    files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.parquet'), recursive=True))
    assert files, f'在 {args.data_dir} 下未找到 parquet 文件'

    def doc_iter():
        n = 0
        for f in files:
            if n >= args.max_samples:
                break
            table = pq.read_table(f, columns=['text'])
            for text in table.column('text').to_pylist():
                if n >= args.max_samples:
                    break
                if text and len(text) >= args.min_doc_chars:
                    n += 1
                    yield text

    special_tokens = [
        "<|endoftext|>",   # 0: pad / unk
        "<|im_start|>",    # 1: bos
        "<|im_end|>",      # 2: eos
        "<think>", "</think>",
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
    ]

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(doc_iter(), trainer=trainer)

    # 不在 encode 时自动加 bos/eos（训练脚本手动控制边界），decode 时跳过特殊 token
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=131072,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(args.output_dir)

    print(f'词表大小: {len(fast_tokenizer)}')
    print(f'bos={fast_tokenizer.bos_token}({fast_tokenizer.bos_token_id}) '
          f'eos={fast_tokenizer.eos_token}({fast_tokenizer.eos_token_id}) '
          f'pad={fast_tokenizer.pad_token}({fast_tokenizer.pad_token_id})')
    demo = "Let $x$ be a real number such that $x^2 - 5x + 6 = 0$. Find all values of $x$."
    ids = fast_tokenizer(demo, add_special_tokens=False).input_ids
    print(f'示例: {len(demo)} chars -> {len(ids)} tokens')
    print(f'还原: {fast_tokenizer.decode(ids)}')


if __name__ == "__main__":
    main()
