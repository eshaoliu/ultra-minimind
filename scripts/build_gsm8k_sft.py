"""从 GSM8K 构建 CoT warmup SFT 数据：question -> 人工 gold CoT（含 #### 终答）。

这是 RLVR 冷启动数据：先把"分步推理 + 终答格式"的结构通过 SFT 注入模型，
让后续 GRPO 的组内正确率离开 0%，zero-variance 组比例下降。

Usage: python scripts/build_gsm8k_sft.py [--src dataset/gsm8k/train.jsonl]
       [--dst dataset/gsm8k_sft.jsonl] [--n 8000]
"""
import argparse
import json
import re

INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'


def clean_cot(answer_text):
    """清理 gold CoT：保留计算行与 #### 终答，去掉 <<>> 计算器标记。"""
    lines = []
    for line in answer_text.split('\n'):
        line = re.sub(r'<<[^>]*>>', '', line).rstrip()
        if line.strip():
            lines.append(line)
    return '\n'.join(lines).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='dataset/gsm8k/train.jsonl')
    parser.add_argument('--dst', default='dataset/gsm8k_sft.jsonl')
    parser.add_argument('--n', type=int, default=8000, help='样本数（-1 表示全部）')
    args = parser.parse_args()

    n_out = 0
    with open(args.src) as f_in, open(args.dst, 'w') as f_out:
        for line in f_in:
            if args.n >= 0 and n_out >= args.n:
                break
            sample = json.loads(line)
            if '####' not in sample['answer']:
                continue
            convs = [
                {'role': 'user', 'content': sample['question'].strip() + INSTRUCTION},
                {'role': 'assistant', 'content': clean_cot(sample['answer'])},
            ]
            f_out.write(json.dumps({'conversations': convs}, ensure_ascii=False) + '\n')
            n_out += 1

    print(f'写出 {n_out} 条 -> {args.dst}')


if __name__ == '__main__':
    main()
