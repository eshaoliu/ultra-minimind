"""从多种格式的数据集构建 RLVR 训练数据：自动识别格式并提取可验证的标准答案。

支持三种输入格式（自动逐行识别）：
A. GSM8K 型:  {"question": ..., "answer": "...#### 18"}          —— 从 #### 提取
B. RLVR 型:   {"conversations": [...], "answer": "18"}            —— answer 字段直接可用
C. 对话型:    {"conversations": [..., {"role":"assistant","content":"...答案...18"}]}
              —— 从末轮 assistant 回复中提取终答（#### / 答案是 / answer: 标记）

无法提取标准答案的行会被跳过（不做可验证奖励的样本没有训练信号）。

Usage:
  python scripts/build_rlvr_data.py --src dataset/gsm8k/train.jsonl --dst dataset/rlvr_math.jsonl
  python scripts/build_rlvr_data.py --src a.jsonl b.jsonl --dst out.jsonl   # 多文件合并
"""
import argparse
import json
import re

# 数字归一化：去千分位逗号、末尾点
NUM_RE = r'-?[\d,]+(?:\.\d+)?'

GOLD_PATTERNS = [
    r'####\s*\$?\s*(' + NUM_RE + r')',              # GSM8K 标记: #### 18
    r'答案[是为]?\s*[:：]?\s*\$?\s*(' + NUM_RE + r')',  # 中文标记: 答案：18 / 答案是18
    r'(?i)answer\s*[:=]\s*\$?\s*(' + NUM_RE + r')',  # 英文标记: answer: 18
]
PLAIN_NUM_RE = re.compile(r'^' + NUM_RE + r'$')


def normalize_num(s):
    return s.replace(',', '').rstrip('.') if s else None


def extract_gold(text):
    """从文本中提取最终数值答案，按标记优先级；无标记返回 None。"""
    if not text:
        return None
    for pat in GOLD_PATTERNS:
        m = re.findall(pat, text)
        if m:
            return normalize_num(m[-1])
    return None


def gold_from_sample(sample):
    """从任意格式的样本中提取标准答案，返回 (gold, conversations) 或 (None, None)。

    conversations 为 None 表示该样本无对话结构（gsm8k 型，需另行构造 prompt）。
    """
    # B/C 型：conversations 结构
    if 'conversations' in sample:
        convs = sample['conversations']
        # B 型：显式 answer 字段
        ans = sample.get('answer')
        if ans is not None:
            gold = normalize_num(ans) if PLAIN_NUM_RE.match(str(ans).strip()) else extract_gold(str(ans))
            if gold is not None:
                return gold, convs
        # C 型：末轮 assistant 中提取
        if convs and convs[-1].get('role') == 'assistant':
            gold = extract_gold(convs[-1].get('content', ''))
            if gold is not None:
                return gold, convs
        return None, None

    # A 型：question + answer
    if 'question' in sample and 'answer' in sample:
        gold = extract_gold(sample['answer'])
        return gold, None

    return None, None


INSTRUCTION = '\nPlease reason step by step, and give the final numeric answer after #### (e.g. #### 42).'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', nargs='+', default=['dataset/gsm8k/train.jsonl'],
                        help='输入 jsonl（可多文件，自动识别格式）')
    parser.add_argument('--dst', default='dataset/rlvr_math.jsonl', help='RLVR 输出路径')
    parser.add_argument('--no_instruction', action='store_true',
                        help='不在 gsm8k 型题目的 question 后附加作答要求（输入自带要求时用）')
    args = parser.parse_args()

    total_in, total_out = 0, 0
    with open(args.dst, 'w') as f_out:
        for src_path in args.src:
            n_in, n_out = 0, 0
            with open(src_path) as f_in:
                for line in f_in:
                    n_in += 1
                    sample = json.loads(line)
                    gold, convs = gold_from_sample(sample)
                    if gold is None:
                        continue
                    if convs is None:  # A 型：构造单轮对话
                        question = sample['question'].strip()
                        if not args.no_instruction:
                            question += INSTRUCTION
                        convs = [{'role': 'user', 'content': question}]
                    out = {'conversations': convs, 'answer': gold}
                    f_out.write(json.dumps(out, ensure_ascii=False) + '\n')
                    n_out += 1
            total_in += n_in
            total_out += n_out
            print(f'{src_path}: 读取 {n_in}, 写出 {n_out}, 跳过 {n_in - n_out}')

    print(f'合计: 读取 {total_in}, 写出 {total_out} -> {args.dst}')


if __name__ == '__main__':
    main()
