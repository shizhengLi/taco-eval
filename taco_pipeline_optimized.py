#!/usr/bin/env python3
"""
TACO端到端优化脚本
整合生成和评估的完整流程
"""

import json
import time
import torch
from datasets import load_dataset
from generation import parallel_generation, get_model_and_tokenizer
from compute_metric import batch_evaluate_generations, compute_metrics, load_generation

def taco_pipeline_optimized(taco_dataset_path, n_samples=20, temperature=0.2, top_p=0.95):
    """
    优化的TACO评估流水线
    
    Args:
        taco_dataset_path: TACO数据集路径
        n_samples: 每个问题的生成样本数
        temperature: 生成温度
        top_p: 生成top_p参数
    
    Returns:
        metrics: 评估指标
    """
    
    print("=== TACO Optimized Pipeline ===")
    
    # 1. 数据准备
    print("\n1. Preparing dataset...")
    difficulties = ['EASY']
    taco_dataset = load_dataset('json', data_files=taco_dataset_path)['train'].filter(
        lambda entry: entry['difficulty'] in difficulties
    )
    
    # 添加idx字段
    taco_list = []
    for idx, sample in enumerate(taco_dataset):
        sample_with_idx = dict(sample)
        sample_with_idx["idx"] = idx
        taco_list.append(sample_with_idx)
    
    print(f"Loaded {len(taco_list)} samples")
    
    # 2. 模型准备
    print("\n2. Loading model...")
    model_name = "/data/lishizheng/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    model, tokenizer = get_model_and_tokenizer(model_name)
    device = "cuda:0"
    model = model.to(device)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 并行生成
    print(f"\n3. Starting parallel generation ({n_samples} samples per problem)...")
    start_time = time.time()
    
    generation_results = parallel_generation(
        taco_list, n_samples, temperature, top_p, num_workers=4
    )
    
    generation_time = time.time() - start_time
    
    # 保存生成结果
    output_file = 'generations_optimized.json'
    with open(output_file, 'w') as f:
        json.dump(generation_results, f, indent=4)
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated {len(generation_results)} problems, saved to {output_file}")
    
    # 4. 并行评估
    print(f"\n4. Starting optimized evaluation...")
    start_time = time.time()
    
    generations_dict = {res["task_id"]: res["output"] for res in generation_results}
    evaluation_results = batch_evaluate_generations(
        generations_dict, taco_list, batch_size=5, debug=False
    )
    
    evaluation_time = time.time() - start_time
    
    # 5. 计算指标
    print("\n5. Computing metrics...")
    metrics = compute_metrics(evaluation_results)
    
    # 保存结果
    metrics_file = 'taco_metrics_optimized.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    print(f"Metrics saved to {metrics_file}")
    
    # 6. 性能统计
    print("\n=== Performance Summary ===")
    print(f"Total problems: {len(generation_results)}")
    print(f"Total generations: {sum(len(res['output']) for res in generation_results)}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    print(f"Total time: {generation_time + evaluation_time:.2f} seconds")
    
    # 显示主要指标
    if 'pass@1' in metrics:
        print(f"Pass@1: {metrics['pass@1']:.4f}")
    if 'pass@10' in metrics:
        print(f"Pass@10: {metrics['pass@10']:.4f}")
    
    return metrics

def main():
    """主函数"""
    # 配置参数
    taco_dataset_path = '/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/test_easy.json'
    n_samples = 10  # 减少样本数用于快速测试
    temperature = 0.2
    top_p = 0.95
    
    try:
        # 运行优化流水线
        metrics = taco_pipeline_optimized(
            taco_dataset_path, n_samples, temperature, top_p
        )
        
        print("\n=== Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()