"""中文数学 think+工具 RLVR 数据集：APE15k + M23k6k 池内抽 cap 题。

行格式（RLVRDataset._parse 直接可用）：
  {"conversations": [{"role": "user", "content": "<题面>+中文指令"}],
   "answer": "<gold 十进制>"}
与冷启动 SFT 同源同 seed（ape 15k / m23k 6k 与 cnmath_thinktool_v1 一致），
再从池中随机抽 cap 条（默认 1000），不与 eval300 训练/测试集混。
"""
import argparse
import json
import random
import sys

ROOT = '/root/autodl-tmp/minimind'
sys.path.insert(0, ROOT + '/scripts')
import build_cnmath_tool_sft as T  # noqa: E402

cn = T.cn
CH_INSTRUCTION = T.CH_INSTRUCTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ape_cap', type=int, default=15000)
    ap.add_argument('--m23k_cap', type=int, default=6000)
    ap.add_argument('--rl_cap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--dst', default=f'{ROOT}/dataset/rlvr_cnmath_thinktool_v1.jsonl')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = {'bad_q': 0, 'bad_eq': 0, 'bad_ans': 0, 'dup': 0}
    seen = set()
    ape_rows = cn.read_ape(f'{ROOT}/dataset/ape210k/train.ape.json')
    m23_rows = cn.read_m23k(f'{ROOT}/dataset/math23k/math23k_train.json')
    rng.shuffle(ape_rows)
    rng.shuffle(m23_rows)
    ape = T.collect_valid(ape_rows, args.ape_cap, seen, stats, 'ape')
    m23 = T.collect_valid(m23_rows, args.m23k_cap, seen, stats, 'm23k')
    pool = ape + m23
    rng.shuffle(pool)
    pool = pool[:args.rl_cap]

    with open(args.dst, 'w', encoding='utf-8') as f:
        for rec in pool:
            line = {'conversations': [
                {'role': 'user', 'content': rec['q'] + CH_INSTRUCTION}],
                'answer': str(rec['gv'])}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'rows {len(pool)} -> {args.dst}  (stats {stats})')


if __name__ == '__main__':
    main()
