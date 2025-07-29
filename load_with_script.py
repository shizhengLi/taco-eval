#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用taco.py脚本加载TACO数据集
"""

import os
import json
import sys
from datasets import load_dataset

def load_taco_with_script():
    """使用taco.py脚本加载TACO数据集"""
    
    dataset_dir = "/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset"
    output_dir = "/data/lishizheng/code/peft_study/datasets-peft/TACO"
    
    print("使用taco.py脚本加载TACO数据集...")
    
    try:
        # 添加数据集目录到Python路径
        sys.path.insert(0, dataset_dir)
        
        # 加载数据集
        dataset = load_dataset(dataset_dir + "/taco.py")
        print(f"✅ 成功加载数据集")
        print(f"数据集分割: {list(dataset.keys())}")
        
        # 处理训练数据
        if 'train' in dataset:
            print(f"开始处理训练数据...")
            train_data = list(dataset['train'])
            train_file = os.path.join(output_dir, 'train.json')
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存训练数据: {len(train_data)} 条")
        
        # 处理测试数据
        if 'test' in dataset:
            print(f"开始处理测试数据...")
            test_data = list(dataset['test'])
            eval_file = os.path.join(output_dir, 'eval.json')
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存评估数据: {len(test_data)} 条")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    load_taco_with_script() 