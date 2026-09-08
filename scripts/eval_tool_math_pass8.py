"""中文数学工具版 pass@8：300 题 x K 次采样，任一次答对即通过。

复用 eval_tool_math 的 rollout / 判分，不改动原脚本。
用法:
  python scripts/eval_tool_math_pass8.py \
      --weight out/cnmath_tool_v1_768.pth \
      --data dataset/cnmath_eval300.jsonl --n 300 --k 8
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

ROOT = '/root/autodl-tmp/minimind'
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/scripts')

import eval_tool_math as ev  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight', default=f'{ROOT}/out/cnmath_tool_v1_768.pth')
    ap.add_argument('--data', default=f'{ROOT}/dataset/cnmath_eval300.jsonl')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_turns', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=300)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dump', default='')
    args = ap.parse_args()

    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    tokenizer = AutoTokenizer.from_pretrained(f'{ROOT}/model')
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(args.weight, map_location=args.device), strict=False)
    model = model.half().to(args.device).eval()

    rargs = SimpleNamespace(device=args.device, max_turns=args.max_turns,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p)
    samples = ev.load_samples(args.data, args.n, args.seed)
    instr = ev.CH_INSTRUCTION
    K = args.k

    sample_pass = 0
    total_ok_runs = 0
    total_runs = 0
    fmt_ok_samples = 0
    tool_use_samples = 0
    results = []
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for i, s in enumerate(samples):
        ok_runs = 0
        runs = []
        sample_tool = 0
        sample_fmt = True
        for k in range(K):
            torch.manual_seed(args.seed + i * K + k)
            torch.cuda.manual_seed(args.seed + i * K + k)
            final_text, n_calls, fmt_ok, _trace = ev.rollout(
                model, tokenizer, s['q'], instr, rargs)
            pred = ev.extract_answer(final_text)
            pv, gv = ev.to_float(pred), ev.to_float(s['a'])
            ok = pv is not None and gv is not None and abs(pv - gv) < 1e-4 * max(1.0, abs(gv))
            ok_runs += int(ok)
            total_ok_runs += int(ok)
            total_runs += 1
            sample_tool += n_calls > 0
            sample_fmt = sample_fmt and fmt_ok
            runs.append({'ok': ok, 'pred': pred, 'n_calls': n_calls,
                         'fmt_ok': fmt_ok, 'final': final_text[-220:]})
        passed = ok_runs > 0
        sample_pass += int(passed)
        fmt_ok_samples += int(sample_fmt)
        tool_use_samples += int(sample_tool > 0)
        results.append({'q': s['q'][:200], 'gold': s['a'], 'pass': passed,
                        'ok_runs': ok_runs, 'runs': runs})
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            print(f'[{i+1}/{len(samples)}] pass_so_far={sample_pass} '
                  f'ok_runs={total_ok_runs}/{total_runs}', flush=True)

    t1.record(); torch.cuda.synchronize()
    n = len(samples)
    print(f'\nPass@{K}: {sample_pass}/{n} = {sample_pass/n:.2%}')
    print(f'总正确 rollout: {total_ok_runs}/{total_runs} ({total_ok_runs/total_runs:.2%})')
    print(f'8/8 全对: {sum(r["ok_runs"] == K for r in results)}  0/8: {sum(r["ok_runs"] == 0 for r in results)}')
    print(f'至少一次调工具: {tool_use_samples}/{n}  全部 rollout 格式合法: {fmt_ok_samples}/{n}')
    print(f'耗时 {t1.elapsed_time(t0)/1000:.1f}s')

    # cnmath_eval300 固定顺序: ape 100 / m23k 100 / cmath 100
    if n == 300:
        for j, src in enumerate(['ape', 'm23k', 'cmath']):
            seg = results[j * 100:(j + 1) * 100]
            p = sum(r['pass'] for r in seg)
            okr = sum(r['ok_runs'] for r in seg)
            print(f'{src}: pass@{K}={p}/100, ok_runs={okr}/800')

    wstem = os.path.basename(args.weight).split('.')[0]
    dstem = os.path.basename(args.data).split('.')[0]
    dump = args.dump or f'{ROOT}/out/pass8_{wstem}_{dstem}.json'
    with open(dump, 'w', encoding='utf-8') as f:
        json.dump({'pass_k': sample_pass / n, 'n': n, 'k': K,
                   'ok_runs': total_ok_runs, 'total_runs': total_runs,
                   'temperature': args.temperature, 'results': results},
                  f, ensure_ascii=False, indent=1)
    print(f'明细 -> {dump}')


if __name__ == '__main__':
    main()
