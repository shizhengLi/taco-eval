from metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

TIMEOUT = 10


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join()
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]

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
    print(len(generations.keys()), len(samples))
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
    """处理单个任务的生成结果"""
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

def evaluate_generations_parallel(generations, samples, debug=False, max_workers=None):
    """
    并行版本的评估函数
    
    Args:
        generations: 生成结果字典
        samples: 样本列表
        debug: 是否开启调试模式
        max_workers: 最大工作进程数，None表示使用CPU核心数
    
    Returns:
        评估结果字典
    """
    print(f"开始并行评估，任务数量: {len(generations.keys())}, 样本数量: {len(samples)}")
    assert len(generations.keys()) == len(samples)
    
    # 准备任务参数
    tasks = []
    idx = 0
    for task_id, problem_generations in generations.items():
        sample = samples[idx]
        tasks.append((task_id, sample, problem_generations, debug))
        idx += 1
    
    results = {}
    
    # 使用进程池并行处理
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(tasks))
    
    print(f"使用 {max_workers} 个进程进行并行处理")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_generation, task): task for task in tasks}
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_task):
            try:
                task_id, res = future.result()
                results[task_id] = res
                completed += 1
                if completed % 10 == 0:
                    print(f"已完成 {completed}/{len(tasks)} 个任务")
            except Exception as e:
                task = future_to_task[future]
                print(f"任务 {task[0]} 处理失败: {e}")
                # 添加默认结果
                results[task[0]] = []
    
    end_time = time.time()
    print(f"并行评估完成，耗时: {end_time - start_time:.2f} 秒")
    
    return results

def evaluate_generations_batch(generations, samples, debug=False, batch_size=10):
    """
    批处理版本的评估函数，可以控制内存使用
    
    Args:
        generations: 生成结果字典
        samples: 样本列表
        debug: 是否开启调试模式
        batch_size: 批处理大小
    
    Returns:
        评估结果字典
    """
    print(f"开始批处理评估，任务数量: {len(generations.keys())}, 样本数量: {len(samples)}")
    assert len(generations.keys()) == len(samples)
    
    # 准备任务参数
    tasks = []
    idx = 0
    for task_id, problem_generations in generations.items():
        sample = samples[idx]
        tasks.append((task_id, sample, problem_generations, debug))
        idx += 1
    
    results = {}
    max_workers = min(multiprocessing.cpu_count(), batch_size)
    
    # 分批处理
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_generation, task): task for task in batch_tasks}
            
            for future in as_completed(future_to_task):
                try:
                    task_id, res = future.result()
                    results[task_id] = res
                except Exception as e:
                    task = future_to_task[future]
                    print(f"任务 {task[0]} 处理失败: {e}")
                    results[task[0]] = []
    
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
    difficulties = ['ALL']
    # difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
    # skills = ['ALL']
    # skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

    from datasets import load_dataset
    taco = load_dataset('json', data_files='//data/lishizheng/code/peft_study/datasets-peft/TACO/sampled_data/test_easy_100.json')['train']
    # taco = load_dataset('BAAI/TACO', split='test', skills=skills)

    generation_file = '/data/lishizheng/code/peft_study/datasets-peft/TACO/generations_lora_easy_100*20.json'
    generations = load_generation(generation_file)

    # 选择评估方式
    print("选择评估方式:")
    print("1. 串行评估 (原始)")
    print("2. 并行评估 (推荐)")
    print("3. 批处理评估 (内存友好)")
    
    choice = input("请输入选择 (1/2/3，默认2): ").strip()
    if choice == "1":
        print("使用串行评估...")
        results = evaluate_generations(generations, taco)
    elif choice == "3":
        print("使用批处理评估...")
        batch_size = int(input("请输入批处理大小 (默认10): ") or "10")
        results = evaluate_generations_batch(generations, taco, batch_size=batch_size)
    else:
        print("使用并行评估...")
        max_workers = input("请输入最大工作进程数 (默认CPU核心数): ").strip()
        max_workers = int(max_workers) if max_workers else None
        results = evaluate_generations_parallel(generations, taco, max_workers=max_workers)
    
    metrics = compute_metrics(results)
    print(metrics)

    output_path = '/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_metrics_lora_new_100*20.json'
    json.dump(metrics, open(output_path, 'w'), indent=4)
    print(f"✅ 指标已保存到: {output_path}")

if __name__ == "__main__":
    main()