#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取TACO数据集中难度为easy的数据
"""

import os
import json
import sys
from datasets import load_dataset

def extract_easy_data():
    """提取easy难度的TACO数据"""
    
    dataset_dir = "/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset"
    output_dir = "/data/lishizheng/code/peft_study/datasets-peft/TACO"
    
    print("加载TACO数据集并提取easy难度的数据...")
    
    try:
        # 添加数据集目录到Python路径
        sys.path.insert(0, dataset_dir)
        
        # 加载数据集
        dataset = load_dataset(dataset_dir + "/taco.py")
        print(f"✅ 成功加载数据集")
        print(f"数据集分割: {list(dataset.keys())}")
        
        easy_train_data = []
        easy_test_data = []
        
        # 处理训练数据，只保留easy难度
        if 'train' in dataset:
            print("开始筛选easy难度的训练数据...")
            for item in dataset['train']:
                if item.get('difficulty') == 'EASY':
                    easy_train_data.append(item)
            print(f"✅ 找到 {len(easy_train_data)} 条easy难度的训练数据")
        
        # 处理测试数据，只保留easy难度
        if 'test' in dataset:
            print("开始筛选easy难度的测试数据...")
            for item in dataset['test']:
                if item.get('difficulty') == 'EASY':
                    easy_test_data.append(item)
            print(f"✅ 找到 {len(easy_test_data)} 条easy难度的测试数据")
        
        # 保存easy训练数据
        if easy_train_data:
            train_file = os.path.join(output_dir, 'train_easy.json')
            print(f"正在保存训练数据到: {train_file}")
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(easy_train_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存easy训练数据: {len(easy_train_data)} 条")
            
            # 验证文件是否创建成功
            if os.path.exists(train_file):
                print(f"✅ 文件创建成功: {train_file}")
                print(f"文件大小: {os.path.getsize(train_file)} 字节")
            else:
                print(f"❌ 文件创建失败: {train_file}")
        
        # 保存easy测试数据作为eval
        if easy_test_data:
            eval_file = os.path.join(output_dir, 'eval_easy.json')
            print(f"正在保存评估数据到: {eval_file}")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(easy_test_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存easy评估数据: {len(easy_test_data)} 条")
            
            # 验证文件是否创建成功
            if os.path.exists(eval_file):
                print(f"✅ 文件创建成功: {eval_file}")
                print(f"文件大小: {os.path.getsize(eval_file)} 字节")
            else:
                print(f"❌ 文件创建失败: {eval_file}")
        
        print(f"\n提取完成！")
        print(f"Easy训练数据: {len(easy_train_data)} 条")
        print(f"Easy评估数据: {len(easy_test_data)} 条")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    extract_easy_data() 