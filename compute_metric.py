from metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
import os
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Dict
from datasets import load_dataset

TIMEOUT = 10
# 优化配置
MAX_WORKERS = min(4, multiprocessing.cpu_count() // 2)  # 限制最大进程数
BATCH_SIZE = 8  # 增加批次大小减少进程创建开销
USE_THREADING = True  # 使用线程替代进程减少开销


def check_correctness_optimized(sample, generation, timeout, debug=True):
    """优化的代码检查函数，使用线程池减少开销"""
    try:
        # 直接运行测试，避免嵌套多进程
        result = run_test_with_timeout(sample, generation, timeout, debug)
        return result
    except Exception as e:
        if debug:
            print(f"Exception in check_correctness_optimized: {e}")
        in_outs = json.loads(sample["input_output"])
        return [[-1 for i in range(len(in_outs["inputs"]))]]

def run_test_with_timeout(sample, generation, timeout, debug=True):
    """使用线程的超时测试函数"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def worker():
        try:
            result = run_test(sample, test=generation, debug=debug)
            result_queue.put(result)
        except Exception as e:
            if debug:
                print(f"Worker exception: {e}")
            result_queue.put(None)
    
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        if debug:
            print(f"Timeout after {timeout} seconds")
        in_outs = json.loads(sample["input_output"])
        return [[-1 for i in range(len(in_outs["inputs"]))]]
    
    try:
        result = result_queue.get_nowait()
        return result if result is not None else [[-1]]
    except queue.Empty:
        if debug:
            print("Queue empty, returning timeout")
        in_outs = json.loads(sample["input_output"])
        return [[-1 for i in range(len(in_outs["inputs"]))]]

def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = json.load(f)
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations[task_id] = output
    return generations

def evaluate_generations(generations, samples, idx=None, debug=False):
    assert len(generations.keys()) == len(samples)
    results = {}
    idx = 0
    for task_id, problem_generations in generations.items():
        sample = samples[idx]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                if debug:
                    print(f"\nSuccessful compilation of task {o_idx}!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    if debug:
                        print(f"Results were not True for all test cases")
            except Exception as e:
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
        results[task_id] = res
        idx += 1
    return results

def process_generation(args):
    task_id, sample, problem_generations, debug = args
    res = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    return task_id, res

def evaluate_generations_parallel(generations, samples, idx=None, debug=False):
    assert len(generations.keys()) == len(samples)
    args = [(task_id, samples[i], problem_generations, debug) for i, (task_id, problem_generations) in enumerate(generations.items())]
    import multiprocessing as mp
    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.map(process_generation, args)
    
    results = {task_id: res for task_id, res in results_list}
    return results

def evaluate_generations_parallel_fixed(generations, samples, debug=False):
    """修复版本的并行评估函数 - 优化资源使用"""
    assert len(generations.keys()) == len(samples)
    
    # 根据系统资源动态调整工作进程数
    cpu_count = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    
    # 根据内存和CPU限制工作进程数
    if available_memory < 8:  # 内存小于8GB
        max_workers = min(2, cpu_count // 2)
    elif available_memory < 16:  # 内存小于16GB
        max_workers = min(4, cpu_count // 2)
    else:
        max_workers = min(MAX_WORKERS, cpu_count // 2)
    
    print(f"Using {max_workers} workers (CPU: {cpu_count}, Memory: {available_memory:.1f}GB)")
    
    if USE_THREADING and max_workers <= 4:
        # 对于少量工作进程，使用线程池减少开销
        return evaluate_with_threading(generations, samples, max_workers, debug)
    else:
        # 使用多进程处理CPU密集型任务
        return evaluate_with_processes(generations, samples, max_workers, debug)

def evaluate_with_threading(generations, samples, max_workers, debug=False):
    """使用线程池进行评估"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_generation_threaded, task_id, samples[i], problem_generations, debug): task_id
            for i, (task_id, problem_generations) in enumerate(generations.items())
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                task_id, res = future.result()
                results[task_id] = res
            except Exception as e:
                print(f"Task {task_id} failed: {e}")
                results[task_id] = [[-2]]
    
    return results

def evaluate_with_processes(generations, samples, max_workers, debug=False):
    """使用进程池进行评估"""
    with multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        args = [(task_id, samples[i], problem_generations, debug) 
                for i, (task_id, problem_generations) in enumerate(generations.items())]
        
        # 使用分批处理避免内存问题
        batch_size = BATCH_SIZE
        all_results = []
        
        for i in range(0, len(args), batch_size):
            batch_args = args[i:i+batch_size]
            batch_results = pool.map(process_generation_fixed, batch_args)
            all_results.extend(batch_results)
            
            # 给系统一点喘息时间
            if i + batch_size < len(args):
                import time
                time.sleep(0.1)
    
    results = {task_id: res for task_id, res in all_results}
    return results

def process_generation_threaded(args):
    """线程版本的处理函数"""
    task_id, sample, problem_generations, debug = args
    res = []
    
    for o_idx, o in enumerate(problem_generations):
        try:
            # 使用优化的检查函数
            curr_res = check_correctness_optimized(sample, o, timeout=TIMEOUT, debug=debug)
            
            # 结果处理
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            
            if not np.all(curr_res) and debug:
                print(f"Results were not True for all test cases")
                
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            curr_res = [-2]  # 明确的错误标识
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    
    return task_id, res

def process_generation_fixed(args):
    """修复版本的处理函数 - 优化版本"""
    task_id, sample, problem_generations, debug = args
    res = []
    
    for o_idx, o in enumerate(problem_generations):
        try:
            # 使用优化的检查函数
            curr_res = check_correctness_optimized(sample, o, timeout=TIMEOUT, debug=debug)
            
            # 结果处理
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            
            if not np.all(curr_res) and debug:
                print(f"Results were not True for all test cases")
                
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            curr_res = [-2]  # 明确的错误标识
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    
    return task_id, res

def check_correctness_safe(sample, generation, timeout, debug=True):
    """更安全的代码检查函数 - 使用优化版本"""
    try:
        # 使用优化的检查函数
        result = check_correctness_optimized(sample, generation, timeout, debug)
        return result
    except Exception as e:
        if debug:
            print(f"Exception in check_correctness_safe: {e}")
        return [-1]  # 超时或错误标识

def batch_evaluate_generations(generations, samples, batch_size=4, debug=False):
    """批量评估以减少进程创建开销"""
    # 移除严格断言，支持不同数量的generation
    print(f"Found {len(generations)} generations and {len(samples)} samples")
    
    # 创建task_id到sample的映射
    sample_map = {}
    for i, sample in enumerate(samples):
        # 如果sample有id字段，使用id作为key，否则使用index
        if hasattr(sample, 'id') or (isinstance(sample, dict) and 'id' in sample):
            sample_id = sample['id'] if isinstance(sample, dict) else sample.id
        else:
            sample_id = i
        sample_map[sample_id] = sample
    
    # 只处理有对应sample的generation
    valid_generations = {}
    missing_samples = []
    
    for task_id in generations.keys():
        if task_id in sample_map:
            valid_generations[task_id] = generations[task_id]
        else:
            missing_samples.append(task_id)
    
    if missing_samples:
        print(f"Warning: No matching samples found for task_ids: {missing_samples[:10]}{'...' if len(missing_samples) > 10 else ''}")
    
    if not valid_generations:
        print("Error: No valid generations to evaluate")
        return {}
    
    print(f"Evaluating {len(valid_generations)} valid generations")
    
    # 分批处理
    task_ids = list(valid_generations.keys())
    results = {}
    
    for i in range(0, len(task_ids), batch_size):
        batch_task_ids = task_ids[i:i+batch_size]
        batch_generations = {tid: valid_generations[tid] for tid in batch_task_ids}
        
        # 获取对应的samples
        batch_samples = [sample_map[tid] for tid in batch_task_ids]
        
        # 对每批使用并行评估
        batch_results = evaluate_generations_parallel_fixed(batch_generations, batch_samples, debug)
        results.update(batch_results)
    
    return results

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
        

def compute_metrics(results, k_list=[1, 10, 100]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen>0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k:dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k



def main():
    # Initialize evaluation dataset with the same setup with generation
    difficulties = ['EASY']
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
    # skills = ['ALL']
    # skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

    from datasets import load_dataset
    # 使用与generation相同的数据源
    taco_full = load_dataset('json', data_files='/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/test_easy.json')['train'].filter(lambda entry: entry['difficulty'] in difficulties)
    
    # 要测试的生成文件
    generation_file = "taco_generation_example.json"
    #'generations_optimized.json'
    generations = load_generation(generation_file)
    
    print(f"Loaded {len(generations)} generations from {generation_file}")
    print(f"Loaded {len(taco_full)} samples from dataset")
    
    # 直接传递完整的数据集，让batch_evaluate_generations处理匹配
    print("Starting optimized evaluation...")
    print(f"System info: CPU cores = {multiprocessing.cpu_count()}, Memory = {psutil.virtual_memory().available / (1024**3):.1f}GB")

    print(f"Config: MAX_WORKERS = {MAX_WORKERS}, BATCH_SIZE = {BATCH_SIZE}, USE_THREADING = {USE_THREADING}")
    
    # 使用修复的并行评估
    results = batch_evaluate_generations(generations, taco_full, batch_size=BATCH_SIZE, debug=False)
    
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # 保存优化后的结果
    output_file = 'taco_metrics_optimized.json'
    json.dump(metrics, open(output_file, 'w'), indent=4)
    print(f"Evaluation completed, metrics saved to {output_file}")

if __name__ == "__main__":
    main()
