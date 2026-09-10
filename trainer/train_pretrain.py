import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset, PretrainStreamDataset, estimate_stream_iters
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def compute_lr(current_step, total_steps, lr):
    """学习率调度：cosine（旧默认）或 WSD（Warmup-Stable-Decay）"""
    if args.lr_scheduler == 'wsd':
        warmup = max(int(args.warmup_ratio * total_steps), 1)
        decay_start = int((1.0 - args.decay_ratio) * total_steps)
        if current_step < warmup:
            return lr * max(current_step, 1) / warmup
        if current_step < decay_start:
            return lr
        # 衰减段（最后 decay_ratio 比例）：cosine 从 lr 降到 0.1*lr
        prog = (current_step - decay_start) / max(total_steps - decay_start, 1)
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * prog)))
    return get_lr(current_step, total_steps, lr)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch
        else:  # 流式模式：dataset 只产出 packed input_ids，labels 克隆自 input_ids
            input_ids = batch
            labels = input_ids.clone()
            labels[input_ids == args.pad_token_id] = -100
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step
        lr = compute_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            tok_s = step * args.batch_size * args.max_seq_len / spend_time if spend_time > 0 else 0
            ppl = math.exp(min(current_logits_loss, 20))
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, ppl: {ppl:.2f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, tok/s: {tok_s:.0f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "ppl": ppl, "aux_loss": current_aux_loss, "learning_rate": current_lr, "tok_per_s": tok_s, "epoch_time": eta_min}, step=step)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', resume_dir=args.resume_dir)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--resume_dir", type=str, default="../checkpoints", help="完整断点(resume)保存目录，默认与权重一起；磁盘紧张时建议指向大空间磁盘")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size（packed 定长序列数，vocab 32k 时 L40 建议 32，显存大可上调）")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--lr_scheduler", type=str, default="wsd", choices=["cosine", "wsd"], help="学习率调度：wsd=warmup-stable-decay，cosine=旧默认")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="WSD：warmup 步数占比")
    parser.add_argument("--decay_ratio", type=float, default=0.2, help="WSD：末尾衰减段占比")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="packing 定长块长度（英文语料建议512~2048）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--gradient_checkpointing', default=0, type=int, choices=[0, 1], help="梯度检查点：以~30%重算开销换显存，大模型必开")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="/root/gpufree-data/dataset/Nemotron-CC-Math-v1/4plus", help="预训练数据路径（parquet 目录/glob，或 jsonl 文件）")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名")
    parser.add_argument("--max_samples", type=int, default=0, help="最多加载的文档条数（0=全部）")
    parser.add_argument("--num_proc", type=int, default=16, help="数据预处理进程数（非流式）")
    parser.add_argument("--packed_cache", type=str, default="", help="packed arrow 缓存路径（非流式；默认自动生成到 ../dataset_cache/，已存在则直接加载）")
    parser.add_argument("--streaming", default=1, type=int, choices=[0, 1], help="流式读取模式（1=开，参考 axolotl streaming：在线分词+packing，无全量预处理）")
    parser.add_argument("--num_threads", type=int, default=32, help="流式模式的分词线程数（fast tokenizer 释放 GIL）")
    parser.add_argument("--skip_samples", type=int, default=0, help="流式模式：跳过前 N 篇文档（消费下一批数据用，不触发分词）")
    parser.add_argument("--tokenizer_path", type=str, default="../model/tokenizer_en", help="分词器目录（英文 BPE）")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否启用实验追踪（默认swanlab，--logger wandb可切换）")
    parser.add_argument("--logger", type=str, default="swanlab", choices=["swanlab", "wandb"], help="实验追踪后端")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline"], help="wandb模式：auto=有凭证在线、无则离线；offline=仅本地记录可后续sync")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="实验追踪项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    # Ampere 及以上 GPU：允许 TF32 矩阵运算，显著提速且对预训练影响可忽略
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ========== 2. 分词器、配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.pad_token_id = tokenizer.pad_token_id
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe), vocab_size=len(tokenizer),
                               bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                               gradient_checkpointing=bool(args.gradient_checkpointing))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints', resume_dir=args.resume_dir) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配实验追踪（swanlab / wandb） ==========
    wandb = None  # 变量名沿用 wandb，作为通用 logger 句柄
    if args.use_wandb and is_main_process():
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb_run_name = f"MiniMind-Pretrain-{args.hidden_size}x{args.num_hidden_layers}-Epoch-{args.epochs}-LR-{args.learning_rate}-{args.lr_scheduler}"
        run_config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}
        if args.logger == 'swanlab':
            try:
                import swanlab
                if os.environ.get('SWANLAB_API_KEY'):
                    swanlab.login(os.environ['SWANLAB_API_KEY'], save=True)
                    swan_mode = 'cloud'   # 云端：swanlab.cn 网页实时看
                else:
                    swan_mode = 'local'   # 本地：数据存 log_dir，swanlab watch 查看
                wandb = swanlab.init(project=args.wandb_project, name=wandb_run_name,
                                     id=wandb_id, resume='must' if wandb_id else None,
                                     mode=swan_mode, config=run_config)
                Logger(f'swanlab 已启用 (mode={swan_mode})')
            except Exception as e:
                Logger(f'swanlab 初始化失败（{type(e).__name__}: {e}），降级为纯日志记录')
        else:
            try:
                import wandb as _wandb
                mode = args.wandb_mode
                if mode == 'auto':
                    mode = 'online' if os.environ.get('WANDB_API_KEY') else 'offline'
                _wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id,
                            resume='must' if wandb_id else None, mode=mode, config=run_config)
                wandb = _wandb
                Logger(f'wandb 已启用 (mode={mode})')
            except Exception as e:
                Logger(f'wandb 初始化失败（{type(e).__name__}: {e}），降级为纯日志记录')
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, _ = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, device=args.device, tokenizer=tokenizer)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            fused=(device_type == 'cuda' and torch.cuda.is_available()))

    if args.streaming == 1:
        assert not os.path.splitext(args.data_path.rstrip('/'))[1] in ('.jsonl', '.json'), '流式模式仅支持 parquet 目录/glob'
        assert not dist.is_initialized(), '流式模式暂不支持 DDP，请单卡运行'
        train_ds = PretrainStreamDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                                         text_column=args.text_column, max_samples=args.max_samples,
                                         num_threads=args.num_threads, skip_samples=args.skip_samples)
        stream_iters = estimate_stream_iters(train_ds.files, args.batch_size, args.max_seq_len, args.max_samples)
        train_sampler = None
        Logger(f'流式模式: 预计 {stream_iters} steps/epoch (按元数据估算，±2%), '
               f'每 step {args.batch_size * args.accumulation_steps} 条 x {args.max_seq_len} tokens')
    else:
        if not args.packed_cache and os.path.isdir(args.data_path):
            src = os.path.basename(os.path.normpath(args.data_path))
            n = args.max_samples or 'all'
            args.packed_cache = f'../dataset_cache/packed_{src}_{n}_s{args.max_seq_len}.arrow'
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                                   text_column=args.text_column, max_samples=args.max_samples,
                                   num_proc=args.num_proc, packed_cache=args.packed_cache or None)
        stream_iters = None
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        Logger(f'packed blocks: {len(train_ds)}, 每块 {args.max_seq_len} tokens, '
               f'每 step {args.batch_size * args.accumulation_steps} 条 x {args.max_seq_len} tokens')
    if is_main_process():
        tokenizer.save_pretrained(args.save_dir)  # 分词器随权重一并保存，便于推理加载
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        if args.streaming == 1:
            # 续训时按 block 边界对齐跳过（需重放分词以保持块序列一致）
            train_ds.skip_blocks = (start_step * args.batch_size) if (epoch == start_epoch and start_step > 0) else 0
            if train_ds.skip_blocks:
                Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 流式跳过前 {train_ds.skip_blocks} 个 block（从 step {start_step + 1} 继续）')
            loader = DataLoader(train_ds, batch_size=args.batch_size)  # 预取由数据集内部线程池承担
            last_step = train_epoch(epoch, loader, stream_iters, start_step if epoch == start_epoch else 0, wandb)
            if is_main_process():  # 流式总 step 为估算值，每轮结束强制保存一次
                model.eval()
                raw_model = getattr(model, '_orig_mod', model)
                state_dict = raw_model.state_dict()
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth')
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler,
                              epoch=epoch, step=last_step, wandb=wandb, save_dir='../checkpoints', resume_dir=args.resume_dir)
                model.train()
                del state_dict
        else:
            train_sampler and train_sampler.set_epoch(epoch)
            setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
            skip = start_step if (epoch == start_epoch and start_step > 0) else 0
            batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                pin_memory=True, persistent_workers=args.num_workers > 0,
                                prefetch_factor=4 if args.num_workers > 0 else None)
            if skip > 0: 
                Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
                train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
            else:
                train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()