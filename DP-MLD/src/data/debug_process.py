import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 创建一个函数来检查数据框中的每一列
def check_column_types(df, column_name=None):
    if column_name:
        col = df[column_name]
        print(f"列 {column_name} 的前5个值: {col.head().tolist()}")
        print(f"列 {column_name} 的类型: {col.dtype}")
        print(f"列 {column_name} 中的唯一值类型: {set([type(x) for x in col.head(20)])}")
    else:
        for i, col in enumerate(df.columns):
            sample_values = df[col].head().tolist()
            print(f"列 {i} 的前5个值: {sample_values}")
            print(f"列 {i} 的类型: {df[col].dtype}")
            print(f"列 {i} 中的唯一值类型: {set([type(x) for x in df[col].head(20)])}")
            print("-" * 50)

# 读取数据
print("开始读取数据...")
data_total = pd.DataFrame()

for task in ["_1","_2","_3"]:
    print(f"读取 task{task}.txt")
    task_data_tmp = pd.read_csv(f"data/raw/task{task}.txt", header=None)
    print(f"task{task}.txt 的形状: {task_data_tmp.shape}")
    data_total = pd.concat([data_total, task_data_tmp], axis=0)

print(f"合并后的数据形状: {data_total.shape}")

# 检查数据类型
print("\n检查前10列的数据类型:")
for i in range(min(10, len(data_total.columns))):
    col = data_total.iloc[:, i]
    print(f"列 {i} 的类型: {col.dtype}")
    print(f"列 {i} 的前5个值: {col.head().tolist()}")
    try:
        # 尝试对该列应用np.round()
        rounded = np.round(col)
        print(f"列 {i} 可以被np.round()处理")
    except Exception as e:
        print(f"列 {i} 无法被np.round()处理: {str(e)}")
    print("-" * 50)

# 尝试修复问题
print("\n尝试修复问题:")
try:
    # 原始代码
    print("尝试原始代码:")
    data = data_total.iloc[:, :-1].apply(lambda x: np.round(x).astype(int))
    print("成功!")
except Exception as e:
    print(f"错误: {str(e)}")
    
    # 修复方案1: 先转换为数值型
    print("\n尝试修复方案1 - 使用pd.to_numeric:")
    try:
        data = data_total.iloc[:, :-1].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).apply(lambda x: np.round(x).astype(int))
        print("修复方案1成功!")
    except Exception as e:
        print(f"修复方案1失败: {str(e)}")
    
    # 修复方案2: 跳过时间戳列
    print("\n尝试修复方案2 - 跳过时间戳列:")
    try:
        # 假设第1列是时间戳列
        numeric_cols = list(range(data_total.shape[1]))
        numeric_cols.pop(1)  # 移除索引为1的列
        numeric_cols.pop(-1)  # 移除最后一列(标签)
        data = data_total.iloc[:, numeric_cols].apply(lambda x: np.round(x).astype(int))
        print("修复方案2成功!")
    except Exception as e:
        print(f"修复方案2失败: {str(e)}")

print("\n完成调试") 