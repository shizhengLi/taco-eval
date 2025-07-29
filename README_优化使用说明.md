# TACO优化代码使用说明

## 优化内容

根据优化方案，已对TACO评估流程进行了以下优化：

### 1. generation.py 优化
- **Batch Generation**: 实现批量生成，减少模型调用次数
- **并行样本处理**: 使用ThreadPoolExecutor并行处理多个样本
- **模型单例模式**: 避免重复加载模型
- **Padding Token修复**: 解决tokenizer padding问题

### 2. compute_metric.py 优化
- **修复并行评估**: 解决原有并行版本返回0的问题
- **批量评估**: 减少进程创建开销
- **改进错误处理**: 提高稳定性和容错性
- **更安全的多进程**: 使用spawn模式避免fork问题

### 3. 新增文件
- **taco_pipeline_optimized.py**: 端到端优化流水线
- **test_optimization.py**: 性能测试脚本

## 使用方法

### 方法1: 分别运行优化后的脚本

#### 1. 代码生成
```bash
# 激活环境
conda activate peft_study

# 运行优化后的生成脚本
python generation.py
```
生成结果将保存到 `generations_optimized.json`

#### 2. 代码评估
```bash
# 运行优化后的评估脚本
python compute_metric.py
```
评估结果将保存到 `taco_metrics_optimized.json`

### 方法2: 使用端到端流水线

```bash
# 运行完整的优化流水线
python taco_pipeline_optimized.py
```

### 方法3: 性能对比测试

```bash
# 运行性能测试脚本
python test_optimization.py
```

## 性能优化预期

### 生成阶段
- **Batch generation**: 3-5倍加速
- **并行样本处理**: 2-4倍加速
- **总体预期**: 6-20倍加速

### 评估阶段
- **修复并行评估**: 解决正确性问题
- **批量评估**: 2-3倍加速
- **总体预期**: 2-3倍加速 + 正确结果

## 配置参数

### 生成参数
- `n_samples`: 每个问题的生成样本数（默认: 3-20）
- `temperature`: 生成温度（默认: 0.2）
- `top_p`: 生成top_p参数（默认: 0.95）
- `num_workers`: 并行工作数（默认: 4）

### 评估参数
- `batch_size`: 评估批次大小（默认: 5-10）
- `timeout`: 代码执行超时时间（默认: 10秒）

## 文件说明

### 输入文件
- `taco_dataset/test_easy.json`: TACO测试数据集

### 输出文件
- `generations_optimized.json`: 优化后的生成结果
- `taco_metrics_optimized.json`: 优化后的评估指标

### 代码文件
- `generation.py`: 优化后的生成脚本
- `compute_metric.py`: 优化后的评估脚本
- `taco_pipeline_optimized.py`: 端到端流水线
- `test_optimization.py`: 性能测试脚本

## 注意事项

1. **GPU内存**: 根据GPU内存调整batch_size和num_workers
2. **数据路径**: 确保数据集路径正确
3. **环境依赖**: 确保所有依赖包已安装
4. **超时设置**: 根据问题复杂度调整timeout参数

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或num_workers
2. **Tokenizer padding错误**: 确保已设置padding token
3. **并行评估失败**: 检查多进程环境配置

### 调试建议
1. 先用小规模数据测试
2. 逐步增加样本数量
3. 监控GPU和CPU使用情况

## 扩展优化

### 进一步优化方向
1. **模型量化**: 使用int8量化减少内存使用
2. **分布式处理**: 多GPU并行处理
3. **缓存机制**: 缓存已生成的结果
4. **异步处理**: 生成和评估流水线重叠

---

**创建时间**: 2025-07-29  
**环境**: peft_study conda环境  
**状态**: 已完成优化，测试通过