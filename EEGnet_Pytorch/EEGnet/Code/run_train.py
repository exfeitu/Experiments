#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行训练脚本
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入必要的模块
from train import main
from options import get_config

if __name__ == "__main__":
    print("正在导入模块...")
    # 获取配置
    config = get_config(checkpoint_path=False)
    # 修改root路径为当前项目路径
    config.root = '/home/exfeitu/Experiments/EEGnet_Pytorch/EEGnet'
    print("配置信息:")
    print(f"设备: {config.device}")
    print(f"最大轮数: {config.max_epoch}")
    print(f"学习率: {config.lr}")
    print(f"批次大小: {config.batch_size}")
    print(f"数据路径: {config.root}")
    
    # 开始训练
    main(config)
