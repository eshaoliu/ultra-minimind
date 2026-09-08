"""think+工具（OpenAI <tool_call> JSON 协议）的 RLVR+GRPO 训练器。

从 trainer/train_tool_grpo.py 改造：把 ReAct(Thought/Action) 协议换成
cnmath_thinktool 冷启动 SFT 同款协议：
- 生成提示：system(带 tools schema) + user(题面+中文指令)，模板以
  `<think>\n` 开头（open_thinking=True）；
- 回合制 rollout：模型写 `<think>推理</think>` 后可发
  `<tool_call>{"name":..., "arguments":...}</tool_call>`，工具真执行，
  结果以 `<tool_response>` 回填；直到不再发 tool_call 或到 max_turns；
- 终答判分用 etm.extract_answer（#### 优先，兼容分数/百分数），
  GRPO advantage 按 batch*num_generations 分组归一；
- loss/KL 只盖模型生成 token（prompt/工具回填段 mask 掉）。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import argparse
import json
import time
import warnings
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import Logger, get_model_params, setup_seed, lm_checkpoint
import eval_tool_math as etm

warnings.filterwarnings('ignore')

ASSISTANT_HEADER = '<|im_start|>assistant\n'
TOOL_SEG_TMPL = '<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n'
EOS_ = None  # lazy: tokenizer.eos_token_id


def load_data(path):
    from dataset.lm_dataset import RLVRDataset
    questions, golds = [], []
    for line in open(path):
        gold, convs = RLVRDataset._parse(json.loads(line))
        if gold is None:
            continue
        questions.append(convs[0]['content'])
        golds.append(gold)
    return questions, golds


def pad_batch(seqs, pad_id, device):
    maxlen = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, device=device)
    return ids


def truncate_at_eos(ids, eos_id):
    ids = ids.tolist()
    if eos_id in ids:
        ids = ids[:ids.index(eos_id) + 1]
    return ids


def rollout_batch(model, tokenizer, prompt_seqs, args):
    """prompt_seqs: n 个 prompt token list。返回逐样本轨迹与统计。"""
    global EOS_
    EOS_ = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or EOS_
    device = args.device
    n = len(prompt_seqs)
    ctx = [list(p) for p in prompt_seqs]
    completions = [[] for _ in range(n)]
    masks = [[] for _ in range(n)]
    states = [{} for _ in range(n)]
    finals = [''] * n
    n_calls = [0] * n
    calc_ok = [0] * n
    calc_n = [0] * n
    done = [False] * n
    header_ids = tokenizer(ASSISTANT_HEADER, add_special_tokens=False).input_ids

    for turn in range(args.max_turns):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        if turn > 0:
            for i in active:
                ctx[i] += header_ids
                completions[i] += header_ids
                masks[i] += [0] * len(header_ids)
        input_ids = pad_batch([ctx[i] for i in active], pad_id, device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_gen_len,
                                 do_sample=True, temperature=args.temperature, top_p=0.9,
                                 eos_token_id=EOS_, pad_token_id=pad_id)
        plen = input_ids.shape[1]
        for row, i in enumerate(active):
            gen = truncate_at_eos(out[row, plen:], EOS_)
            gen_text = tokenizer.decode(gen, skip_special_tokens=True)
            ctx[i] += gen
            completions[i] += gen
            masks[i] += [1] * len(gen)
            finals[i] = gen_text
            calls = etm.parse_tool_calls(gen_text)
            if len(completions[i]) > args.max_seq_len:
                done[i] = True
                continue
            if not calls:
                done[i] = True
                continue
            for c in calls:
                result = etm.execute_tool(c.get('name', ''), c.get('arguments', {}) or {}, states[i])
                n_calls[i] += 1
                if c.get('name') == 'calculate':
                    calc_n[i] += 1
                    if 'result' in result:
                        calc_ok[i] += 1
                seg = tokenizer(TOOL_SEG_TMPL.format(content=json.dumps(result, ensure_ascii=False)),
                                add_special_tokens=False).input_ids
                ctx[i] += seg
                completions[i] += seg
                masks[i] += [0] * len(seg)
                if len(completions[i]) > args.max_seq_len:
                    done[i] = True
                    break
    hit_cap = [n_calls[i] > 0 and not done[i] for i in range(n)]
    return completions, masks, n_calls, calc_ok, calc_n, finals, hit_cap


def calculate_reward(final_text, gold, n_calls, calc_ok, calc_n, hit_cap, args):
    correct = 0.0
    pred = etm.extract_answer(final_text)
    try:
        if pred is not None and abs(float(pred) - float(gold)) < 1e-4:
            correct = 1.0
    except (TypeError, ValueError):
        pass
    r = args.outcome_weight * correct
    if args.format_weight > 0:
        r += args.format_weight if '####' in final_text else -0.5 * args.format_weight
    if args.process_weight > 0 and calc_n > 0:
        r += args.process_weight * (calc_ok / calc_n)
    if args.think_weight > 0:
        if '</think>' in final_text:
            r += args.think_weight if final_text.count('</think>') == 1 else -0.5 * args.think_weight
        else:
            r -= args.think_weight
    if hit_cap and (args.format_weight > 0 or args.process_weight > 0):
        r -= 0.1
    return r, correct


def train_epoch(model, ref_model, tokenizer, questions, golds, args, optimizer, scheduler, wandb):
    iters = min(len(questions) // args.batch_size, args.max_steps or 10 ** 9)
    start = time.time()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for step in range(1, iters + 1):
        idxs = random.sample(range(len(questions)), args.batch_size)
        B, G = args.batch_size, args.num_generations

        model.eval()
        prompt_seqs = []
        for i in idxs:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": etm.SYSTEM},
                 {"role": "user", "content": questions[i]}],
                tokenize=False, add_generation_prompt=True,
                tools=etm.TOOLS, open_thinking=True)
            p_ids = tokenizer(prompt_text, truncation=True,
                              max_length=args.max_prompt_len).input_ids
            prompt_seqs += [p_ids] * G

        completions, masks, n_calls, calc_ok, calc_n, finals, hit_cap = \
            rollout_batch(model, tokenizer, prompt_seqs, args)

        rewards, corrects = [], []
        for k in range(B * G):
            r, c = calculate_reward(finals[k], golds[idxs[k // G]], n_calls[k],
                                    calc_ok[k], calc_n[k], hit_cap[k], args)
            rewards.append(r)
            corrects.append(c)

        rewards_t = torch.tensor(rewards, device=args.device)
        grouped = rewards_t.view(B, G)
        adv = ((grouped - grouped.mean(dim=1, keepdim=True))
               / (grouped.std(dim=1, keepdim=True) + 1e-4)).reshape(-1)

        model.train()
        seqs = [prompt_seqs[k] + completions[k] for k in range(B * G)]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((B * G, maxlen), pad_id, dtype=torch.long, device=args.device)
        loss_mask = torch.zeros((B * G, maxlen), dtype=torch.float, device=args.device)
        attn = torch.zeros((B * G, maxlen), dtype=torch.long, device=args.device)
        for k, s in enumerate(seqs):
            L = len(s)
            input_ids[k, :L] = torch.tensor(s, device=args.device)
            attn[k, :L] = 1
            loss_mask[k, len(prompt_seqs[k]):L] = torch.tensor(masks[k], dtype=torch.float, device=args.device)
        lm = loss_mask[:, 1:]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            tok_logp = F.log_softmax(model(input_ids, attention_mask=attn).logits[:, :-1, :], dim=-1) \
                .gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            pg = -(adv.unsqueeze(1) * tok_logp * lm).sum() / lm.sum()
            with torch.no_grad():
                ref_tok = F.log_softmax(ref_model(input_ids, attention_mask=attn).logits[:, :-1, :], dim=-1) \
                    .gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            kl_div = ref_tok - tok_logp
            kl = ((torch.exp(kl_div) - kl_div - 1) * lm).sum() / lm.sum()
            loss = pg + args.beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            acc_t = sum(corrects) / len(corrects)
            avg_turns = sum(n_calls) / len(n_calls)
            eta = (time.time() - start) / step * (iters - step) / 60
            Logger(f'({step}/{iters}), Reward: {rewards_t.mean().item():.4f}, Acc: {acc_t:.4f}, '
                   f'Turns: {avg_turns:.1f}, KL_ref: {kl.item():.4f}, Adv Std: {adv.std().item():.4f}, '
                   f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.8f}, eta: {eta:.0f}min')
            if wandb:
                wandb.log({"reward": rewards_t.mean().item(), "acc": acc_t, "avg_turns": avg_turns,
                           "kl_ref": kl.item(), "adv_std": adv.std().item(),
                           "loss": loss.item(), "learning_rate": optimizer.param_groups[0]["lr"]})

        if step % args.save_interval == 0:
            lm_checkpoint(args, weight=args.save_weight, model=model,
                          optimizer=optimizer, step=step, save_dir=args.save_dir)

    ckp = f'{args.save_dir}/{args.save_weight}_{args.hidden_size}.pth'
    torch.save(model.state_dict(), ckp)
    Logger(f'保存最终权重 -> {ckp}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='rlvr_cnmath_thinktool_v1', type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0, help="0=跑完整轮")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--max_prompt_len", type=int, default=640)
    parser.add_argument("--max_seq_len", type=int, default=1600)
    parser.add_argument("--max_gen_len", type=int, default=160, help="每轮最大生成 token")
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--data_path", type=str, default="../dataset/rlvr_cnmath_thinktool_v1.jsonl")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--outcome_weight", type=float, default=1.0)
    parser.add_argument("--process_weight", type=float, default=0.1)
    parser.add_argument("--format_weight", type=float, default=0.2,
                        help="#### 格式 shaping，置 0 关闭（v2 纯 outcome 用）")
    parser.add_argument("--think_weight", type=float, default=0.05,
                        help="think 格式 shaping：恰一个 </think> 加分，否则扣分")
    parser.add_argument('--from_weight', default='cnmath_thinktool_v1', type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    setup_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('../model')
    questions, golds = load_data(args.data_path)
    Logger(f'数据 {len(questions)} 条')

    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers))
    ckp = f'{args.save_dir}/{args.from_weight}_{args.hidden_size}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    model = model.to(args.device)
    ref_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers))
    ref_model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    ref_model = ref_model.to(args.device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters = min(len(questions) // args.batch_size, args.max_steps or 10 ** 9)
    scheduler = CosineAnnealingLR(optimizer, T_max=iters, eta_min=args.learning_rate * 0.1)

    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb.init(project="MiniMind-RLVR",
                   experiment_name=f"RLVR-ToolGRPO-{args.save_weight}-B{args.batch_size}"
                                   f"-G{args.num_generations}-LR{args.learning_rate}")

    train_epoch(model, ref_model, tokenizer, questions, golds, args, optimizer, scheduler, wandb)


if __name__ == "__main__":
    main()
