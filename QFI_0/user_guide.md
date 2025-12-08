
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
