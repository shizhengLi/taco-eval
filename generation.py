from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import re
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
EOF_STRINGS = ["\nQUESTION", "\n---", "\nANSWER", "<|endoftext|>"]

def truncate_after_eof_strings(text):
    pattern = '|'.join(re.escape(s) for s in EOF_STRINGS)
    match = re.search(pattern, text)
    
    if match:
        return text[:match.start()]
    else:
        return text

import random, numpy as np
def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def decode(tokenizer, raw_text_len, output):
    sents = []
    for tokens in output:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:], skip_special_tokens=True)
        sents.append(sent)
    return sents

def batch_decode(tokenizer, raw_text_lens, outputs):
    sents = []
    for i, tokens in enumerate(outputs):
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_lens[i]:], skip_special_tokens=True)
        sents.append(sent)
    return sents

def predict(device, model, tokenizer, prompt, seed, topp, t, max_length=2048):
    set_random_seed(seed)
    _prompt = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = _prompt["input_ids"]
    raw_text_len = len(input_ids[0])
    with torch.no_grad():
        _prompt = _prompt.to(device)
        output = model.generate(
            **_prompt,
            do_sample=True,
            temperature = t,
            top_p = topp,
            max_new_tokens = max_length,
        )
        output = decode(tokenizer, raw_text_len, output) 
    return output[0]

def batch_predict(device, model, tokenizer, prompts, seeds, topp, t, max_length=2048):
    # 批量处理prompts
    batch_size = min(len(prompts), 4)  # 根据GPU内存调整
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_seeds = seeds[i:i+batch_size]
        
        # 批量tokenization
        _prompts = tokenizer(batch_prompts, truncation=False, return_tensors="pt", padding=True)
        input_ids = _prompts["input_ids"]
        raw_text_lens = [len(ids) for ids in input_ids]
        
        with torch.no_grad():
            _prompts = _prompts.to(device)
            outputs = model.generate(
                **_prompts,
                do_sample=True,
                temperature=t,
                top_p=topp,
                max_new_tokens=max_length,
            )
            
            # 批量解码
            batch_results = batch_decode(tokenizer, raw_text_lens, outputs)
            results.extend(batch_results)
    
    return results

def build_prompt(sample):
    """构建prompt的函数"""
    prompt = "\nQUESTION:\n"
    prompt += sample["question"]
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = (
            None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        )
    except ValueError:
        fn_name = None
    if starter_code:
        prompt += starter_code
    if (not fn_name) and (not starter_code):
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format
    prompt += "\nANSWER:\n"
    return prompt

def process_sample_batch(args):
    device, model, tokenizer, samples, n_samples, temperature, top_p = args
    results = []
    
    for sample in samples:
        # 构建prompts
        prompts = []
        seeds = []
        for i in range(n_samples):
            prompt = build_prompt(sample)
            prompts.append(prompt)
            seeds.append(i)
        
        # 批量生成
        generations = batch_predict(device, model, tokenizer, prompts, seeds, top_p, temperature)
        clean_generations = [truncate_after_eof_strings(gen) for gen in generations]
        
        results.append({
            "task_id": sample["idx"],
            "prompt": prompts[0],
            "output": clean_generations
        })
    
    return results

def parallel_generation(taco, n_samples, temperature, top_p, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 4)
    
    # 分割数据集
    batch_size = len(taco) // num_workers
    sample_batches = [taco[i:i+batch_size] for i in range(0, len(taco), batch_size)]
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(device, model, tokenizer, batch, n_samples, temperature, top_p) 
                    for batch in sample_batches]
        results = list(executor.map(process_sample_batch, args_list))
    
    # 合并结果
    final_results = []
    for batch_result in results:
        final_results.extend(batch_result)
    
    return final_results

# Initialize model and tokenizer
model_name = "/data/lishizheng/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda:0"
model = model.to(device)

# 设置padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Initialize evaluation dataset 
difficulties = ['EASY']
# difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
# skills = ['ALL']
# skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

from datasets import load_dataset
# 使用JSON文件而不是不完整的test目录
taco_dataset = load_dataset('json', data_files='/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/test_easy.json')['train'].filter(lambda entry: entry['difficulty'] in difficulties)

# 添加idx字段
taco_list = []
for idx, sample in enumerate(taco_dataset):
    sample_with_idx = dict(sample)
    sample_with_idx["idx"] = idx
    taco_list.append(sample_with_idx)

output_file = 'generations_optimized.json'

# setting up times of run
n_samples = 3  # 减少样本数用于测试
temperature = 0.2
top_p = 0.95 

# 使用并行生成（只测试前5个样本）
print("Starting parallel generation...")
test_samples = taco_list[:5]  # 只测试前5个样本
output = parallel_generation(test_samples, n_samples, temperature, top_p)

with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(output)} samples, saved to {output_file}")


