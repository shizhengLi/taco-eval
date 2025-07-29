# TACO评估性能优化方案

## 现状分析

### 当前性能瓶颈

#### 1. generation.py 问题
- **串行处理**: 逐个处理每个样本，无并行化
- **重复模型加载**: 每次生成都重新加载模型（generation.py:54-58）
- **单样本生成**: 每个prompt单独调用模型，无batch处理
- **内存效率低**: 没有复用CUDA内存

#### 2. compute_metric.py 问题
- **并行版本未被使用**: 第173行使用串行版本`evaluate_generations`
- **并行版本实现问题**: 
  - 多进程间数据共享可能导致竞争条件
  - 异常处理不完善，可能导致返回错误结果
  - 超时机制在多进程环境下不稳定

### 具体问题定位

#### generation.py 性能瓶颈
- **Line 80-108**: 串行循环处理每个样本
- **Line 102-106**: 每个样本生成20个代码版本，逐个处理
- **Line 36-48**: 每次调用都进行完整的tokenization和model.generate

#### compute_metric.py 并行问题
- **Line 173**: 未启用并行评估 `evaluate_generations(generations, taco)`
- **Line 108-116**: 并行实现存在进程安全问题
- **Line 11-31**: 全局超时和信号处理在多进程环境下冲突

## 优化方案

### 方案一: 生成阶段优化 (generation.py)

#### 1.1 实现Batch Generation
```python
def batch_predict(device, model, tokenizer, prompts, seeds, topp, t, max_length=2048):
    # 批量处理prompts
    batch_size = min(len(prompts), 8)  # 根据GPU内存调整
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
```

#### 1.2 实现并行样本处理
```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_sample_batch(args):
    device, model, tokenizer, samples, n_samples, temperature, top_p = args
    results = []
    
    for sample in samples:
        # 构建prompts
        prompts = []
        seeds = []
        for i in range(n_samples):
            prompt = build_prompt(sample)  # 提取prompt构建逻辑
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
```

#### 1.3 优化模型加载和内存管理
```python
# 模型单例模式
_model_instance = None
_tokenizer_instance = None

def get_model_and_tokenizer(model_name):
    global _model_instance, _tokenizer_instance
    if _model_instance is None:
        _tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
        _model_instance = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cuda:0"
        _model_instance = _model_instance.to(device)
        _model_instance.eval()  # 设置为评估模式
    return _model_instance, _tokenizer_instance
```

### 方案二: 评估阶段优化 (compute_metric.py)

#### 2.1 修复并行评估
```python
def evaluate_generations_parallel_fixed(generations, samples, debug=False):
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
        # 使用subprocess而不是multiprocessing避免信号问题
        result = run_test_with_subprocess(sample, generation, timeout, debug)
        return result
    except Exception as e:
        if debug:
            print(f"Exception in check_correctness_safe: {e}")
        return [-1]  # 超时或错误标识

def run_test_with_subprocess(sample, generation, timeout, debug):
    """使用subprocess运行测试避免信号冲突"""
    # 实现subprocess版本的测试运行
    # 避免多进程信号问题
    pass
```

#### 2.2 实现批量评估
```python
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
```

### 方案三: 端到端优化

#### 3.1 流水线处理
```python
def taco_pipeline(taco, n_samples=20, temperature=0.2, top_p=0.95):
    # 1. 并行生成
    print("Starting parallel generation...")
    generation_results = parallel_generation(taco, n_samples, temperature, top_p)
    
    # 保存生成结果
    with open('generations_optimized.json', 'w') as f:
        json.dump(generation_results, f, indent=4)
    
    # 2. 并行评估
    print("Starting parallel evaluation...")
    generations_dict = {res["task_id"]: res["output"] for res in generation_results}
    evaluation_results = batch_evaluate_generations(generations_dict, taco)
    
    # 3. 计算指标
    print("Computing metrics...")
    metrics = compute_metrics(evaluation_results)
    
    # 保存结果
    with open('taco_metrics_optimized.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics
```

## 实施计划

### 阶段一: 生成阶段优化 (优先级: 高)
1. **实现batch generation** - 预期加速3-5倍
2. **实现并行样本处理** - 预期加速2-4倍
3. **优化模型加载** - 减少初始化时间

### 阶段二: 评估阶段优化 (优先级: 高)
1. **修复并行评估** - 解决返回0的问题
2. **实现批量评估** - 减少进程开销
3. **改进错误处理** - 提高稳定性

### 阶段三: 整合优化 (优先级: 中)
1. **端到端流水线** - 统一优化接口
2. **性能监控** - 添加时间和内存监控
3. **参数调优** - 根据硬件调整batch_size和worker数量

## 预期性能提升

### 生成阶段
- **Batch generation**: 3-5倍加速
- **并行样本处理**: 2-4倍加速
- **总体预期**: 6-20倍加速

### 评估阶段
- **修复并行评估**: 解决正确性问题
- **批量评估**: 2-3倍加速
- **总体预期**: 2-3倍加速 + 正确结果

## 环境配置

### Conda环境激活
```bash
conda activate peft_study
```

### 依赖检查
```bash
# 确保必要的包已安装
pip install transformers torch numpy datasets multiprocess
```

### GPU内存优化
```python
# 根据GPU内存调整参数
import torch
if torch.cuda.get_device_properties(0).total_memory < 8e9:  # 8GB
    batch_size = 4
    num_workers = 2
else:
    batch_size = 8
    num_workers = 4
```

## 风险评估

### 技术风险
1. **内存不足**: 大batch处理可能导致OOM
2. **进程间通信**: 多进程可能导致数据竞争
3. **模型兼容性**: 不同模型版本可能影响batch处理

### 缓解措施
1. **动态batch大小**: 根据GPU内存动态调整
2. **进程隔离**: 使用spawn模式避免fork问题
3. **版本固定**: 固定transformers和torch版本

## 测试验证

### 性能测试
```python
import time

def benchmark_original():
    start = time.time()
    # 运行原始版本
    end = time.time()
    return end - start

def benchmark_optimized():
    start = time.time()
    # 运行优化版本
    end = time.time()
    return end - start

# 验证正确性
def verify_correctness(original_results, optimized_results):
    # 比较结果一致性
    pass
```

### 回归测试
1. 小规模数据集验证
2. 结果一致性检查
3. 性能基准对比

## 后续优化方向

1. **模型量化**: 使用int8量化减少内存使用
2. **分布式处理**: 多GPU并行处理
3. **缓存机制**: 缓存已生成的结果
4. **异步处理**: 生成和评估流水线重叠

---

**创建时间**: 2025-07-29  
**环境**: peft_study conda环境  
**目标**: 加速TACO数据集的代码生成和评估流程