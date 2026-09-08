import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

device = 'cuda'
lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
tokenizer = AutoTokenizer.from_pretrained('../model')
model = MiniMindForCausalLM(lm_config)
weights = torch.load('../out/pretrain_768.pth', map_location=device)
model.load_state_dict(weights, strict=False)
model = model.to(device).eval()

prompts = [
    '人工智能是',
    '中国的首都是',
    '机器学习是一种',
    '春天来了，',
]

for p in prompts:
    input_ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=80, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f'[Prompt] {p}')
    print(f'[Output] {text}')
    print('-' * 60)
