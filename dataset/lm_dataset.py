from torch.utils.data import Dataset, IterableDataset
import torch
import json
import os
import glob
import random
import re
from itertools import chain, islice
from datasets import Dataset as HFDataset, load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """英文语料预训练数据集：支持 parquet 分片目录 / glob / 单个 jsonl，文本列默认为 'text'。

    采用 packing 方式：每条文档编码为 [bos] + tokens + [eos]，随后无缝拼接成
    max_length 定长块，消除 padding 浪费，显著提升 GPU 利用率。
    大规模语料（百万级文档）会流式预处理并落盘为 arrow 缓存（int16 内存映射加载），
    重复训练/续训时可直接复用，跳过数小时的分词+packing。
    """

    def __init__(self, data_path, tokenizer, max_length=1024, text_column='text',
                 max_samples=0, num_proc=16, min_doc_chars=32, packed_cache=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ---- 命中缓存：直接内存映射加载（零拷贝、秒开）----
        if packed_cache and os.path.exists(packed_cache):
            self.samples = HFDataset.from_file(packed_cache)
            return

        # ---- 解析数据源：目录递归匹配 *.parquet，或 glob 模式，或单文件 ----
        if os.path.isdir(data_path):
            files = sorted(glob.glob(os.path.join(data_path, '**', '*.parquet'), recursive=True))
            assert files, f'目录 {data_path} 下未找到 parquet 文件'
        elif any(ch in data_path for ch in '*?['):
            files = sorted(glob.glob(data_path, recursive=True))
            assert files, f'glob 模式 {data_path} 未匹配到文件'
        else:
            files = [data_path]
        ext = os.path.splitext(files[0])[1].lstrip('.')

        if ext == 'parquet':
            ds = _stream_pack_parquet(files, text_column, max_samples, min_doc_chars,
                                      max_length, tokenizer, num_proc, packed_cache)
        else:
            ds = _load_jsonl_packed(files, ext, text_column, max_samples, min_doc_chars,
                                    max_length, tokenizer, num_proc)
        self.samples = ds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.samples[index]['input_ids'], dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


def _stream_pack_parquet(files, text_column, max_samples, min_doc_chars, max_length,
                         tokenizer, num_proc, packed_cache):
    """流式预处理 parquet：按 shard 读 text 列 -> 线程池并行分词（Rust 实现释放 GIL）
    -> 拼接成定长块 -> 逐批写入 arrow IPC 文件。峰值内存约一个 chunk，磁盘仅最终缓存。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    id_dtype = pa.int16() if len(tokenizer) <= 32768 else pa.int32()
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    tok = tokenizer  # fast tokenizer，encode 期间释放 GIL，线程即可打满多核

    def read_texts():
        n = 0
        for f in files:
            if max_samples and n >= max_samples:
                break
            for t in pq.read_table(f, columns=[text_column]).column(text_column).to_pylist():
                if max_samples and n >= max_samples:
                    break
                n += 1
                if t and len(t) >= min_doc_chars:
                    yield t

    def chunk_iter(it, size):
        chunk = list(islice(it, size))
        while chunk:
            yield chunk
            chunk = list(islice(it, size))

    os.makedirs(os.path.dirname(packed_cache), exist_ok=True) if packed_cache else None
    sink = pa.OSFile(packed_cache, 'wb') if packed_cache else pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, pa.schema([('input_ids', pa.list_(id_dtype))]))
    buf, n_blocks, n_docs = [], 0, 0
    try:
        with ThreadPoolExecutor(max_workers=num_proc) as pool:
            for texts in chunk_iter(read_texts(), 20000):
                for ids in pool.map(lambda t: tok(t, add_special_tokens=False)['input_ids'], texts):
                    buf.append(bos_id)
                    buf.extend(ids)
                    buf.append(eos_id)
                    while len(buf) >= max_length:
                        writer.write_batch(pa.record_batch([pa.array([buf[:max_length]], type=pa.list_(id_dtype))],
                                                           names=['input_ids']))
                        buf = buf[max_length:]
                        n_blocks += 1
                n_docs += len(texts)
                print(f'[PretrainDataset] 已处理 {n_docs} 文档 -> {n_blocks} blocks', flush=True)
    finally:
        writer.close()
        sink.close()
    print(f'[PretrainDataset] 完成: {n_docs} 文档, {n_blocks} packed blocks x {max_length} tokens', flush=True)
    if packed_cache:
        return HFDataset.from_file(packed_cache)
    from datasets.table import in_memory_table
    reader = pa.ipc.open_stream(pa.BufferReader(sink.getvalue()))
    return HFDataset(in_memory_table(reader.read_all()))


class PretrainStreamDataset(IterableDataset):
    """流式英文预训练数据集（参考 axolotl streaming 方式）：按 shard 顺序读 parquet 的
    text 列 -> 线程池即时分词（fast tokenizer 释放 GIL）-> 在线拼接成 max_length 定长块。
    不做全量预处理，内存占用仅为预取 buffer，适合 10B+ tokens 级语料全程单遍训练。

    - skip_samples: 跳过前 N 篇文档（原始读取，不分词），用于消费"下一批"数据
    - skip_blocks: 跳过前 N 个 packed 块（需要分词以对齐块边界），用于断点续训
    """

    def __init__(self, data_path, tokenizer, max_length=1024, text_column='text',
                 max_samples=0, min_doc_chars=32, num_threads=32,
                 skip_samples=0, skip_blocks=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.max_samples = max_samples
        self.min_doc_chars = min_doc_chars
        self.num_threads = num_threads
        self.skip_samples = skip_samples
        self.skip_blocks = skip_blocks

        if os.path.isdir(data_path):
            self.files = sorted(glob.glob(os.path.join(data_path, '**', '*.parquet'), recursive=True))
        elif any(ch in data_path for ch in '*?['):
            self.files = sorted(glob.glob(data_path, recursive=True))
        else:
            self.files = [data_path]
        assert self.files and self.files[0].endswith('.parquet'), '流式模式仅支持 parquet'

    def __iter__(self):
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor
        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        tok = self.tokenizer
        max_length = self.max_length

        def read_texts():
            n = 0
            for i, f in enumerate(self.files):
                if self.max_samples and n >= self.max_samples:
                    break
                print(f'[stream] 打开 shard {os.path.basename(f)} ({i + 1}/{len(self.files)}), 已读文档 {n}', flush=True)
                for t in pq.read_table(f, columns=[self.text_column]).column(self.text_column).to_pylist():
                    if self.max_samples and n >= self.max_samples:
                        break
                    n += 1
                    if n <= self.skip_samples:
                        continue
                    if t and len(t) >= self.min_doc_chars:
                        yield t

        def chunk_iter(it, size):
            chunk = list(islice(it, size))
            while chunk:
                yield chunk
                chunk = list(islice(it, size))

        buf, n_blocks = [], 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            for texts in chunk_iter(read_texts(), 20000):
                for ids in pool.map(lambda t: tok(t, add_special_tokens=False)['input_ids'], texts):
                    buf.append(bos_id)
                    buf.extend(ids)
                    buf.append(eos_id)
                    while len(buf) >= max_length:
                        block, buf = buf[:max_length], buf[max_length:]
                        n_blocks += 1
                        if n_blocks > self.skip_blocks:
                            yield torch.tensor(block, dtype=torch.long)
        # 末尾不足一块的 token 丢弃


def estimate_stream_iters(files, batch_size, max_length, max_samples=0, tok_per_doc=1070):
    """按 parquet 元数据快速估算总 step 数（用于 LR 调度；可能有 ±2% 误差）"""
    import pyarrow.parquet as pq
    docs = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if max_samples:
        docs = min(docs, max_samples)
    return max(1, docs * tok_per_doc // max_length // batch_size)


def _load_jsonl_packed(files, ext, text_column, max_samples, min_doc_chars, max_length,
                       tokenizer, num_proc):
    """jsonl 小语料的加载路径（load_dataset + 并行 map + packing）"""
    ds = load_dataset(ext, data_files=files, split='train')
    if max_samples and max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    ds = ds.filter(lambda x: x[text_column] and len(x[text_column]) >= min_doc_chars,
                   num_proc=num_proc, desc='过滤短文档')

    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    def encode(batch):
        enc = tokenizer(batch[text_column], add_special_tokens=False)['input_ids']
        return {'input_ids': [[bos_id] + ids + [eos_id] for ids in enc]}
    ds = ds.map(encode, batched=True, batch_size=1000, num_proc=num_proc,
                remove_columns=ds.column_names, desc='分词')
    ds = ds.cast_column('input_ids', Sequence(Value('int32')))

    # packing：num_proc=1 时各批次按序处理，闭包 buffer 可跨批次保留，最后不足一块的尾token丢弃
    def pack(batch):
        pack.buf.extend(chain.from_iterable(batch['input_ids']))
        blocks = []
        while len(pack.buf) >= max_length:
            blocks.append(pack.buf[:max_length])
            pack.buf = pack.buf[max_length:]
        return {'input_ids': blocks} if blocks else {'input_ids': []}
    pack.buf = []
    ds = ds.map(pack, batched=True, batch_size=2000, num_proc=1,
                remove_columns=ds.column_names, desc='拼接packing')
    return ds


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }


class RLVRDataset(Dataset):
    """可验证奖励数据集：prompt 之外还带标准答案(answer)，供规则奖励函数判分。

    数据格式（每行一条，加载时自动识别并过滤）：
    - {"conversations": [...], "answer": "18"}                       —— answer 字段直接可用
    - {"conversations": [..., {"role": "assistant", "content": "...#### 18"}]}
      —— 末轮 assistant 含终答标记（#### / 答案： / answer:），提取后末轮不进入 prompt
    - {"question": ..., "answer": "...#### 18"}                      —— 转为单轮对话
    无法提取标准答案的行在加载时被跳过（无训练信号）。
    """

    GOLD_PATTERNS = [
        r'####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)',
        r'答案[是为]?\s*[:：]?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)',
        r'(?i)answer\s*[:=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)',
    ]
    PLAIN_NUM_RE = re.compile(r'^-?[\d,]+(?:\.\d+)?$')

    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        raw = load_dataset('json', data_files=jsonl_path, split='train')
        # 预扫描：提取标准答案，仅保留可验证样本
        self.samples, self.golds = [], []
        n_skip = 0
        for sample in raw:
            gold, convs = self._parse(sample)
            if gold is None:
                n_skip += 1
                continue
            self.samples.append(convs)
            self.golds.append(gold)
        if n_skip:
            print(f'[RLVRDataset] 跳过无标准答案样本 {n_skip} 条，保留 {len(self.samples)} 条')

    def __len__(self):
        return len(self.samples)

    @classmethod
    def _normalize_num(cls, s):
        return s.replace(',', '').rstrip('.') if s else None

    @classmethod
    def _extract_gold(cls, text):
        if not text:
            return None
        for pat in cls.GOLD_PATTERNS:
            m = re.findall(pat, text)
            if m:
                return cls._normalize_num(m[-1])
        return None

    @classmethod
    def _parse(cls, sample):
        """返回 (gold, conversations)；无法验证返回 (None, None)。"""
        if 'conversations' in sample:
            convs = sample['conversations']
            ans = sample.get('answer')
            if ans is not None:
                gold = cls._normalize_num(str(ans).strip()) if cls.PLAIN_NUM_RE.match(str(ans).strip()) \
                    else cls._extract_gold(str(ans))
                if gold is not None:
                    return gold, convs
            if convs and convs[-1].get('role') == 'assistant':
                gold = cls._extract_gold(convs[-1].get('content', ''))
                if gold is not None:
                    return gold, convs
            return None, None
        if 'question' in sample and 'answer' in sample:
            gold = cls._extract_gold(sample['answer'])
            if gold is None:
                return None, None
            convs = [{'role': 'user', 'content': sample['question'].strip()}]
            return gold, convs
        return None, None

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        if conversations and conversations[-1]['role'] == 'assistant':
            conversations = conversations[:-1]
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        prompt = self.create_chat_prompt(self.samples[index])

        return {
            'prompt': prompt,
            'answer': self.golds[index]
        }


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass