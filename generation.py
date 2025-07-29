from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import re
import json
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



# Initialize model and tokenizer
model_name = "/data/lishizheng/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda:3"
model = model.to(device)


# Initialize evaluation dataset 
difficulties = ['EASY']
# difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
# skills = ['ALL']
# skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

from datasets import load_dataset
# 使用JSON文件而不是不完整的test目录
taco = load_dataset('json', data_files='/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/test_easy.json')['train'].filter(lambda entry: entry['difficulty'] in difficulties)

output_file = 'generations.json'

# setting up times of run
n_samples = 20
temperature = 0.2
top_p = 0.95 


output = []
for idx, sample in enumerate(taco):
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
    results = {"task_id": idx, "prompt": prompt}
    generations = []
    for i in range(n_samples):
        seed = i
        generation = predict(device, model, tokenizer, prompt, seed, top_p, temperature, max_length=2048)
        clean_code = truncate_after_eof_strings(generation)
        generations.append(clean_code)
    results["output"] = generations
    output.append(results)

with open(output_file, 'w') as f:
    json.dump(output, f, indent=4)


