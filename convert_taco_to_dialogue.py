#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将TACO数据集转换为对话格式
每个question和每个solution组成一条独立的对话数据
"""

import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer

def parse_solutions(solutions_str):
    """解析solutions字符串，提取多个解决方案"""
    if not solutions_str:
        return []
    
    try:
        # 尝试解析为JSON列表
        solutions = json.loads(solutions_str)
        if isinstance(solutions, list):
            return [str(s) for s in solutions if s]
        else:
            return [str(solutions)]
    except (json.JSONDecodeError, ValueError):
        # 如果解析失败，作为单个字符串处理
        return [solutions_str.strip()]

def generate_dialogue_pairs(item, global_id):
    """为单个TACO项目生成对话对，每个solution生成一条独立数据"""
    pairs = []
    
    question = item.get('question', '')
    solutions_str = item.get('solutions', '')
    
    if not question or not solutions_str:
        return pairs
    
    # 解析多个解决方案
    solutions = parse_solutions(solutions_str)
    
    # 每个解决方案生成一条独立数据
    for i, solution in enumerate(solutions):
        if not solution.strip():
            continue
            
        user_msg = f"Please solve this programming problem:\n\n{question}"
        assistant_msg = solution
        
        dialogue_pair = {
            "id": f"taco_{global_id}_{i}",
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        }
        pairs.append(dialogue_pair)
    
    return pairs

def filter_by_token_length(dialogue_pairs, tokenizer, max_tokens=1024):
    """根据token长度过滤对话数据"""
    filtered_pairs = []
    
    for pair in tqdm(dialogue_pairs, desc="过滤token长度"):
        # 构建完整的对话文本
        full_text = ""
        for message in pair["messages"]:
            if message["role"] == "user":
                full_text += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                full_text += f"Assistant: {message['content']}\n"
        
        # 计算token数量
        tokens = tokenizer.encode(full_text)
        token_count = len(tokens)
        
        # 只保留不超过max_tokens的数据
        if token_count <= max_tokens:
            filtered_pairs.append(pair)
    
    return filtered_pairs

def convert_taco_to_dialogue(input_file, output_prefix, total_num=None, seed=42, max_tokens=1024, model_name="gpt2"):
    """将TACO数据转换为对话格式"""
    
    print(f"正在读取TACO数据: {input_file}")
    
    # 加载tokenizer
    print(f"正在加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 读取TACO数据
    with open(input_file, 'r', encoding='utf-8') as f:
        taco_data = json.load(f)
    
    print(f"读取到 {len(taco_data)} 条TACO数据")
    
    # 生成所有对话样本
    all_dialogue_pairs = []
    global_id = 0
    
    for item in tqdm(taco_data, desc="生成对话对"):
        pairs = generate_dialogue_pairs(item, global_id)
        all_dialogue_pairs.extend(pairs)
        global_id += 1
    
    print(f"生成了 {len(all_dialogue_pairs)} 条对话对")
    
    # 根据token长度过滤
    print(f"正在过滤超过 {max_tokens} 个token的数据...")
    filtered_pairs = filter_by_token_length(all_dialogue_pairs, tokenizer, max_tokens)
    print(f"过滤后保留 {len(filtered_pairs)} 条对话对")
    
    # 如果指定了总数，进行采样
    if total_num and total_num < len(filtered_pairs):
        import random
        random.seed(seed)
        filtered_pairs = random.sample(filtered_pairs, total_num)
        print(f"采样后保留 {len(filtered_pairs)} 条对话对")
    
    # 保存为JSONL格式
    output_file = f"{output_prefix}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in tqdm(filtered_pairs, desc="保存对话数据"):
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 对话数据已保存到: {output_file}")
    print(f"总共 {len(filtered_pairs)} 条对话对")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将TACO数据转换为对话格式")
    parser.add_argument('--input_file', type=str, 
                       default='/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/train_easy.json',
                       help='输入的TACO JSON文件路径')
    parser.add_argument('--output_prefix', type=str, 
                       default='/data/lishizheng/code/peft_study/datasets-peft/TACO/train_easy_dialogue_filtered',
                       help='输出文件前缀')
    parser.add_argument('--total_num', type=int, default=None,
                       help='总采样数量（可选）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='最大token数量')
    parser.add_argument('--model_name', type=str, default='/data/lishizheng/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062',
                       help='用于tokenize的模型名称')
    
    args = parser.parse_args()
    convert_taco_to_dialogue(args.input_file, args.output_prefix, args.total_num, args.seed, args.max_tokens, args.model_name) 