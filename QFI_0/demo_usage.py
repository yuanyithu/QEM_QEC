"""
演示如何使用Q矩阵优化器
包含实际运行示例和结果分析
"""

import numpy as np
import matplotlib.pyplot as plt
from q_matrix_optimizer import QMatrixOptimizer
import time


def demo_real_problem():
    """演示实际问题的优化过程"""
    
    print("=" * 60)
    print("Q矩阵约束优化 - 实际演示")
    print("=" * 60)
    
    # 定义几个不同的目标函数进行测试
    def objective_function_1(Q):
        """目标函数1: 最大化迹"""
        return np.real(np.trace(Q @ Q.conj().T))
    
    def objective_function_2(Q):
        """目标函数2: 最大化 Frobenius 范数的平方"""
        return np.sum(np.abs(Q)**2)
    
    def objective_function_3(Q):
        """目标函数3: 最大化第一列的范数"""
        return np.linalg.norm(Q[:, 0])**2
    
    def objective_function_4(Q):
        """目标函数4: 复杂函数 - 最大化对角线元素的和"""
        QTQ = Q.conj().T @ Q
        return np.real(np.trace(QTQ @ QTQ))
    
    # 测试参数
    test_cases = [
        {"n": 2, "k": 1, "name": "4x2 矩阵"},
        {"n": 3, "k": 2, "name": "8x4 矩阵"},
        {"n": 4, "k": 2, "name": "16x4 矩阵"},
    ]
    
    objective_functions = [
        ("迹最大化", objective_function_1),
        ("F范数最大化", objective_function_2),
        ("第一列范数最大化", objective_function_3),
        ("复合函数", objective_function_4),
    ]
    
    results = {}
    
    # 运行测试
    for obj_name, obj_func in objective_functions:
        print(f"\n{'='*50}")
        print(f"测试目标函数: {obj_name}")
        print(f"{'='*50}")
        
        results[obj_name] = {}
        
        for case in test_cases:
            n, k = case["n"], case["k"]
            rows, cols = 2**n, 2**k
            
            print(f"\n--- {case['name']} ({rows}x{cols}) ---")
            
            # 创建优化器
            optimizer = QMatrixOptimizer(obj_func)
            
            # 测试最速上升法
            print("最速上升法:")
            start_time = time.time()
            Q_steep, history_steep = optimizer.optimize_steepest_ascent(
                n=n, k=k,
                learning_rate=0.01,
                max_iterations=200,
                initial_method='random',
                seed=42,
                verbose=False
            )
            steep_time = time.time() - start_time
            steep_final = history_steep[-1]['best_value']
            steep_violation = optimizer.check_constraint_violation(Q_steep)
            
            print(f"  最终值: {steep_final:.6f}")
            print(f"  用时: {steep_time:.2f}秒")
            print(f"  约束违反: {steep_violation:.2e}")
            
            # 测试流形梯度下降
            print("流形梯度下降:")
            start_time = time.time()
            Q_manifold, history_manifold = optimizer.optimize_manifold_gradient(
                n=n, k=k,
                learning_rate=0.01,
                max_iterations=200,
                initial_method='random',
                seed=42,
                verbose=False
            )
            manifold_time = time.time() - start_time
            manifold_final = history_manifold[-1]['best_value']
            manifold_violation = optimizer.check_constraint_violation(Q_manifold)
            
            print(f"  最终值: {manifold_final:.6f}")
            print(f"  用时: {manifold_time:.2f}秒")
            print(f"  约束违反: {manifold_violation:.2e}")
            
            # 记录结果
            results[obj_name][case['name']] = {
                'steepest_ascent': {
                    'final_value': steep_final,
                    'time': steep_time,
                    'constraint_violation': steep_violation
                },
                'manifold_gradient': {
                    'final_value': manifold_final,
                    'time': manifold_time,
                    'constraint_violation': manifold_violation
                }
            }
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("优化结果总结")
    print(f"{'='*60}")
    
    for obj_name, obj_results in results.items():
        print(f"\n目标函数: {obj_name}")
        print("-" * 40)
        for case_name, case_results in obj_results.items():
            print(f"{case_name}:")
            steep = case_results['steepest_ascent']
            manifold = case_results['manifold_gradient']
            print(f"  最速上升:     值={steep['final_value']:.6f}, 时间={steep['time']:.2f}s")
            print(f"  流形梯度:     值={manifold['final_value']:.6f}, 时间={manifold['time']:.2f}s")
            print(f"  改进幅度:     {((manifold['final_value'] - steep['final_value'])/steep['final_value']*100):+.2f}%")


def create_user_guide():
    """创建用户使用指南"""
    
    guide = """
# Q矩阵约束优化器使用指南

## 概述
这个优化器专门用于优化形如 run(Q) 的函数，其中 Q 是满足正交归一约束的复数矩阵：
∑_{p} Q[p,i].conj() * Q[p,j] = δ_ij

## 快速开始

### 1. 定义您的目标函数
```python
def your_run_function(Q):
    '''
    输入: shape为[2**n, 2**k]的复数numpy数组
    输出: 实数
    '''
    # 在这里实现您的run函数
    return some_real_value
```

### 2. 创建优化器并运行
```python
from q_matrix_optimizer import QMatrixOptimizer

# 创建优化器
optimizer = QMatrixOptimizer(your_run_function)

# 设置矩阵维度参数
n, k = 3, 2  # Q shape为[8, 4]

# 运行优化
Q_optimal, history = optimizer.optimize_steepest_ascent(
    n=n, k=k,
    learning_rate=0.01,
    max_iterations=1000,
    initial_method='random',
    seed=42
)
```

## 主要方法

### 1. 最速上升法 (推荐)
```python
Q, history = optimizer.optimize_steepest_ascent(
    n, k,                    # 矩阵维度参数
    learning_rate=0.01,      # 学习率
    max_iterations=1000,     # 最大迭代次数
    tolerance=1e-6,          # 收敛容忍度
    initial_method='random', # 初始化方法
    seed=42,                 # 随机种子
    verbose=True             # 是否打印进度
)
```

### 2. 流形梯度下降 (更高效)
```python
Q, history = optimizer.optimize_manifold_gradient(
    n, k,
    learning_rate=0.01,
    max_iterations=1000,
    tolerance=1e-6,
    initial_method='random',
    seed=42,
    verbose=True
)
```

## 参数调优建议

### 学习率 (learning_rate)
- 太大: 可能发散
- 太小: 收敛慢
- 建议范围: 0.001 - 0.1
- 默认值: 0.01

### 最大迭代次数 (max_iterations)
- 根据问题复杂度调整
- 简单问题: 200-500
- 复杂问题: 1000-5000

### 收敛容忍度 (tolerance)
- 数值精度要求
- 建议: 1e-6 到 1e-8

## 常见问题

### Q: 如何选择初始化方法？
A: 
- 'random': 随机正交归一矩阵（推荐）
- 'identity': 单位矩阵（如果维度合适）

### Q: 如何判断是否收敛？
A: 
- 检查历史记录中的 improvement 是否很小
- 梯度范数是否足够小
- 约束违反程度是否在可接受范围内

### Q: 结果不理想怎么办？
A:
1. 尝试不同的随机种子
2. 调整学习率
3. 增加迭代次数
4. 尝试不同的优化方法

## 性能监控

### 检查约束满足程度
```python
violation = optimizer.check_constraint_violation(Q_optimal)
print(f"约束违反程度: {violation:.2e}")
```

### 可视化优化过程
```python
optimizer.visualize_optimization('optimization_plot.png')
```

### 查看优化历史
```python
for record in history:
    print(f"迭代 {record['iteration']}: 值={record['current_value']:.6f}")
```
"""
    
    with open('/workspace/code/user_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("用户指南已保存到: /workspace/code/user_guide.md")


if __name__ == "__main__":
    # 运行演示
    demo_real_problem()
    
    # 创建用户指南
    create_user_guide()
    
    print(f"\n{'='*60}")
    print("演示完成！")
    print("查看以下文件:")
    print("- /workspace/code/q_matrix_optimizer.py: 主要优化器")
    print("- /workspace/code/user_guide.md: 详细使用指南")
    print("- /workspace/code/optimization_results.png: 优化过程可视化")
    print(f"{'='*60}")