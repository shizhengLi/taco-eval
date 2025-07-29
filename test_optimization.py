#!/usr/bin/env python3
"""
TACO优化代码测试脚本
测试generation.py和compute_metric.py的优化效果
"""

import subprocess
import time
import json
import os

def run_command(cmd, timeout=300):
    """运行命令并返回结果"""
    print(f"Running: {cmd}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Command completed in {elapsed_time:.2f} seconds")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0, elapsed_time, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, timeout, "", "Timeout"

def test_original_generation():
    """测试原始生成方法"""
    print("\n=== Testing Original Generation ===")
    
    # 创建原始版本的generation.py备份
    if not os.path.exists('generation_original.py'):
        print("Creating backup of original generation.py...")
        # 这里需要手动创建原始版本或者从git恢复
        return False, 0
    
    # 运行原始版本
    success, time_taken, stdout, stderr = run_command('python generation_original.py')
    
    if success and os.path.exists('generations.json'):
        print(f"Original generation completed in {time_taken:.2f} seconds")
        return True, time_taken
    else:
        print("Original generation failed")
        return False, 0

def test_optimized_generation():
    """测试优化后的生成方法"""
    print("\n=== Testing Optimized Generation ===")
    
    # 运行优化版本
    success, time_taken, stdout, stderr = run_command('python generation.py')
    
    if success and os.path.exists('generations_optimized.json'):
        print(f"Optimized generation completed in {time_taken:.2f} seconds")
        return True, time_taken
    else:
        print("Optimized generation failed")
        return False, 0

def test_original_evaluation():
    """测试原始评估方法"""
    print("\n=== Testing Original Evaluation ===")
    
    if not os.path.exists('generations.json'):
        print("No generations.json found for testing")
        return False, 0
    
    # 运行原始评估
    success, time_taken, stdout, stderr = run_command('python compute_metric_original.py')
    
    if success and os.path.exists('taco_metrics.json'):
        print(f"Original evaluation completed in {time_taken:.2f} seconds")
        return True, time_taken
    else:
        print("Original evaluation failed")
        return False, 0

def test_optimized_evaluation():
    """测试优化后的评估方法"""
    print("\n=== Testing Optimized Evaluation ===")
    
    if not os.path.exists('generations_optimized.json'):
        print("No generations_optimized.json found for testing")
        return False, 0
    
    # 运行优化评估
    success, time_taken, stdout, stderr = run_command('python compute_metric.py')
    
    if success and os.path.exists('taco_metrics_optimized.json'):
        print(f"Optimized evaluation completed in {time_taken:.2f} seconds")
        return True, time_taken
    else:
        print("Optimized evaluation failed")
        return False, 0

def compare_results():
    """比较结果"""
    print("\n=== Comparing Results ===")
    
    # 比较生成结果
    if os.path.exists('generations.json') and os.path.exists('generations_optimized.json'):
        with open('generations.json', 'r') as f:
            original_gen = json.load(f)
        with open('generations_optimized.json', 'r') as f:
            optimized_gen = json.load(f)
        
        print(f"Original generations: {len(original_gen)} samples")
        print(f"Optimized generations: {len(optimized_gen)} samples")
        
        # 比较指标结果
        if os.path.exists('taco_metrics.json') and os.path.exists('taco_metrics_optimized.json'):
            with open('taco_metrics.json', 'r') as f:
                original_metrics = json.load(f)
            with open('taco_metrics_optimized.json', 'r') as f:
                optimized_metrics = json.load(f)
            
            print("Original metrics:", original_metrics)
            print("Optimized metrics:", optimized_metrics)

def main():
    """主测试函数"""
    print("TACO Optimization Test Suite")
    print("=" * 50)
    
    # 测试生成阶段
    #gen_orig_success, gen_orig_time = test_original_generation()
    gen_opt_success, gen_opt_time = test_optimized_generation()
    
    # 测试评估阶段
    #eval_orig_success, eval_orig_time = test_original_evaluation()
    eval_opt_success, eval_opt_time = test_optimized_evaluation()
    
    # 比较结果
    #compare_results()
    
    # 计算加速比
    print("\n=== Performance Summary ===")
    #if gen_orig_success and gen_opt_success:
    #    gen_speedup = gen_orig_time / gen_opt_time
    #    print(f"Generation speedup: {gen_speedup:.2f}x")
    
    #if eval_orig_success and eval_opt_success:
    #    eval_speedup = eval_orig_time / eval_opt_time
    #    print(f"Evaluation speedup: {eval_speedup:.2f}x")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()