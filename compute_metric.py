from metrics.testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset

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
    """修复版本的并行评估函数"""
    assert len(generations.keys()) == len(samples)
    
    # 使用更安全的多进程方式
    with multiprocessing.get_context("spawn").Pool(multiprocessing.cpu_count()) as pool:
        args = [(task_id, samples[i], problem_generations, debug) 
                for i, (task_id, problem_generations) in enumerate(generations.items())]
        
        # 使用imap_unordered提高性能
        results_list = list(pool.imap_unordered(process_generation_fixed, args))
    
    results = {task_id: res for task_id, res in results_list}
    return results

def process_generation_fixed(args):
    """修复版本的处理函数"""
    task_id, sample, problem_generations, debug = args
    res = []
    
    for o_idx, o in enumerate(problem_generations):
        try:
            # 使用更安全的超时处理
            curr_res = check_correctness_safe(sample, o, timeout=TIMEOUT, debug=debug)
            
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
    """更安全的代码检查函数"""
    try:
        # 使用原始的check_correctness函数，但添加更好的错误处理
        result = check_correctness(sample, generation, timeout, debug)
        return result
    except Exception as e:
        if debug:
            print(f"Exception in check_correctness_safe: {e}")
        return [-1]  # 超时或错误标识

def batch_evaluate_generations(generations, samples, batch_size=10, debug=False):
    """批量评估以减少进程创建开销"""
    assert len(generations.keys()) == len(samples)
    
    # 分批处理
    task_ids = list(generations.keys())
    results = {}
    
    for i in range(0, len(task_ids), batch_size):
        batch_task_ids = task_ids[i:i+batch_size]
        batch_generations = {tid: generations[tid] for tid in batch_task_ids}
        batch_samples = samples[i:i+batch_size]
        
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
    
    # 使用优化后的生成文件
    generation_file = 'generations_optimized.json'
    generations = load_generation(generation_file)
    
    # 只评估已生成的样本
    taco_list = []
    for idx, sample in enumerate(taco_full):
        if idx in generations:
            taco_list.append(sample)
    
    print(f"Evaluating {len(generations)} generated samples...")
    # 使用修复的并行评估
    results = batch_evaluate_generations(generations, taco_list, batch_size=5, debug=False)
    
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # 保存优化后的结果
    output_file = 'taco_metrics_optimized.json'
    json.dump(metrics, open(output_file, 'w'), indent=4)
    print(f"Evaluation completed, metrics saved to {output_file}")

if __name__ == "__main__":
    main()
