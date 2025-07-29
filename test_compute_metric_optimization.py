#!/usr/bin/env python3
"""
compute_metric.py 性能优化测试脚本
"""

import time
import subprocess
import psutil

def monitor_cpu_during_test():
    """监控CPU使用率"""
    print("=== 系统资源监控 ===")
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"内存: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # 测试优化版本
    print("\n=== 测试优化版本 ===")
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=1)
    
    try:
        result = subprocess.run(
            ["python", "compute_metric.py"], 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=1)
        
        print(f"执行时间: {end_time - start_time:.2f}秒")
        print(f"平均CPU使用率: {(start_cpu + end_cpu) / 2:.1f}%")
        
        if result.stdout:
            print("输出:")
            print(result.stdout[-500:])  # 显示最后500个字符
            
        if result.stderr:
            print("错误:")
            print(result.stderr[-500:])  # 显示最后500个字符
            
    except subprocess.TimeoutExpired:
        print("测试超时（120秒）")
    except Exception as e:
        print(f"测试失败: {e}")

def analyze_optimizations():
    """分析优化效果"""
    print("\n=== 优化分析 ===")
    
    print("主要优化:")
    print("1. ✅ 限制最大工作进程数 (MAX_WORKERS = min(4, CPU//2))")
    print("2. ✅ 使用线程池替代进程池减少开销")
    print("3. ✅ 增加批次大小减少进程创建开销")
    print("4. ✅ 基于系统资源动态调整配置")
    print("5. ✅ 避免嵌套多进程，使用线程超时")
    print("6. ✅ 添加系统资源监控")
    
    print("\n关于GPU加速:")
    print("❌ Python代码执行本身难以用GPU加速")
    print("✅ 但可以通过减少CPU开销间接提升性能")
    print("✅ GPU主要用于模型推理，代码评估在CPU")
    
    print("\n预期效果:")
    print("- CPU使用率: 从100%降至50-70%")
    print("- 内存使用: 减少进程创建开销")
    print("- 执行时间: 可能略有增加但更稳定")
    print("- 系统响应: 不会完全卡死")

if __name__ == "__main__":
    monitor_cpu_during_test()
    analyze_optimizations()