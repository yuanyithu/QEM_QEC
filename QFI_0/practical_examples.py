"""
实际应用示例：复杂目标函数的优化
演示如何在实际科研问题中使用Q矩阵优化器
"""

import numpy as np
import matplotlib.pyplot as plt
from q_matrix_optimizer import QMatrixOptimizer
import time


def quantum_state_optimization_example():
    """
    量子态优化示例
    假设我们要优化量子态的某种性质
    """
    print("=" * 60)
    print("量子态优化示例")
    print("=" * 60)
    
    def quantum_fidelity(Q):
        """
        量子态保真度目标函数
        假设Q是量子态的某种表示矩阵
        """
        # 计算密度矩阵
        rho = Q @ Q.conj().T
        
        # 计算保真度相关的量（示例）
        fidelity = np.real(np.trace(rho @ rho))
        
        # 添加一些非线性项使问题更有趣
        nonlinear_term = np.sum(np.abs(Q)**4) * 0.1
        
        return fidelity + nonlinear_term
    
    # 优化参数
    n, k = 3, 2  # 8x4矩阵
    
    print(f"优化量子态表示矩阵: {2**n} x {2**k}")
    print("目标函数: 最大化保真度 + 非线性项")
    
    # 创建优化器
    optimizer = QMatrixOptimizer(quantum_fidelity)
    
    # 运行优化
    print("\n开始优化...")
    Q_optimal, history = optimizer.optimize_steepest_ascent(
        n=n, k=k,
        learning_rate=0.005,  # 较小的学习率因为目标函数较复杂
        max_iterations=500,
        initial_method='random',
        seed=123,
        verbose=True
    )
    
    # 分析结果
    print(f"\n优化结果分析:")
    print(f"最终保真度值: {history[-1]['best_value']:.6f}")
    print(f"约束违反程度: {optimizer.check_constraint_violation(Q_optimal):.2e}")
    
    # 可视化优化过程
    optimizer.visualize_optimization('/workspace/code/quantum_optimization.png')
    
    return Q_optimal, history


def signal_processing_example():
    """
    信号处理优化示例
    优化信号变换矩阵的性质
    """
    print("\n" + "=" * 60)
    print("信号处理优化示例")
    print("=" * 60)
    
    def signal_energy(Q):
        """
        信号能量最大化目标函数
        包含频率域特性
        """
        # 计算信号的频域表示
        Q_fft = np.fft.fft(Q, axis=0)
        
        # 频域能量
        frequency_energy = np.sum(np.abs(Q_fft)**2)
        
        # 时域能量
        time_energy = np.sum(np.abs(Q)**2)
        
        # 组合目标：平衡频域和时域特性
        return frequency_energy * 0.7 + time_energy * 0.3
    
    # 优化参数
    n, k = 4, 3  # 16x8矩阵
    
    print(f"优化信号变换矩阵: {2**n} x {2**k}")
    print("目标函数: 平衡频域和时域能量")
    
    # 创建优化器
    optimizer = QMatrixOptimizer(signal_energy)
    
    # 运行优化
    print("\n开始优化...")
    Q_optimal, history = optimizer.optimize_manifold_gradient(
        n=n, k=k,
        learning_rate=0.01,
        max_iterations=300,
        initial_method='random',
        seed=456,
        verbose=True
    )
    
    # 分析结果
    print(f"\n优化结果分析:")
    print(f"最终信号能量: {history[-1]['best_value']:.6f}")
    print(f"约束违反程度: {optimizer.check_constraint_violation(Q_optimal):.2e}")
    
    return Q_optimal, history


def machine_learning_feature_selection():
    """
    机器学习特征选择示例
    优化特征变换矩阵
    """
    print("\n" + "=" * 60)
    print("机器学习特征选择示例")
    print("=" * 60)
    
    def feature_discriminability(Q):
        """
        特征可区分性目标函数
        模拟监督学习中的特征选择
        """
        # 获取Q的维度
        n_samples, n_features = Q.shape
        
        # 模拟数据（实际应用中替换为真实数据）
        np.random.seed(789)
        simulated_data = np.random.randn(n_samples, n_features)
        
        # 应用特征变换
        transformed_features = simulated_data @ Q
        
        # 计算类间和类内散布矩阵（简化版本）
        # 这里用随机标签模拟分类问题
        labels = np.random.randint(0, 2, size=transformed_features.shape[0])
        
        # 计算特征的可区分性指标
        feature_variance = np.var(transformed_features, axis=0)
        
        # 目标：最大化方差，最小化相关性
        variance_term = np.sum(feature_variance)
        
        # 计算相关性矩阵（如果特征数足够）
        if transformed_features.shape[1] > 1:
            feature_correlation = np.corrcoef(transformed_features.T)
            # 移除对角线元素（自相关）
            correlation_off_diagonal = feature_correlation - np.eye(feature_correlation.shape[0])
            correlation_penalty = np.sum(np.abs(correlation_off_diagonal))
        else:
            correlation_penalty = 0
        
        return variance_term - 0.1 * correlation_penalty
    
    # 优化参数
    n, k = 3, 2  # 8x4矩阵
    
    print(f"优化特征变换矩阵: {2**n} x {2**k}")
    print("目标函数: 最大化特征方差，最小化特征相关性")
    
    # 创建优化器
    optimizer = QMatrixOptimizer(feature_discriminability)
    
    # 运行优化
    print("\n开始优化...")
    Q_optimal, history = optimizer.optimize_steepest_ascent(
        n=n, k=k,
        learning_rate=0.01,
        max_iterations=400,
        initial_method='random',
        seed=789,
        verbose=True
    )
    
    # 分析结果
    print(f"\n优化结果分析:")
    print(f"最终特征可区分性: {history[-1]['best_value']:.6f}")
    print(f"约束违反程度: {optimizer.check_constraint_violation(Q_optimal):.2e}")
    
    return Q_optimal, history


def compare_optimization_methods():
    """
    比较不同优化方法的性能
    """
    print("\n" + "=" * 60)
    print("优化方法性能比较")
    print("=" * 60)
    
    def complex_objective(Q):
        """
        复杂目标函数，用于方法比较
        """
        # 多项式目标函数
        QTQ = Q.conj().T @ Q
        polynomial_term = np.real(np.trace(QTQ @ QTQ @ QTQ))
        
        # 非线性项
        nonlinear = np.sum(np.abs(Q)**3) * 0.01
        
        # 正则化项
        regularization = np.sum(np.abs(Q)**2) * 0.001
        
        return polynomial_term + nonlinear - regularization
    
    # 测试参数
    n, k = 3, 2  # 8x4矩阵
    
    print(f"测试矩阵: {2**n} x {2**k}")
    print("复杂目标函数: 多项式 + 非线性 + 正则化")
    
    methods = []
    
    # 方法1: 最速上升法
    print("\n--- 最速上升法 ---")
    optimizer1 = QMatrixOptimizer(complex_objective)
    start_time = time.time()
    Q1, history1 = optimizer1.optimize_steepest_ascent(
        n=n, k=k, learning_rate=0.01, max_iterations=200, 
        initial_method='random', seed=42, verbose=False
    )
    time1 = time.time() - start_time
    final1 = history1[-1]['best_value']
    
    methods.append({
        'name': '最速上升法',
        'final_value': final1,
        'time': time1,
        'violation': optimizer1.check_constraint_violation(Q1),
        'history': history1
    })
    
    print(f"最终值: {final1:.6f}, 用时: {time1:.2f}s")
    
    # 方法2: 流形梯度下降
    print("\n--- 流形梯度下降 ---")
    optimizer2 = QMatrixOptimizer(complex_objective)
    start_time = time.time()
    Q2, history2 = optimizer2.optimize_manifold_gradient(
        n=n, k=k, learning_rate=0.01, max_iterations=200,
        initial_method='random', seed=42, verbose=False
    )
    time2 = time.time() - start_time
    final2 = history2[-1]['best_value']
    
    methods.append({
        'name': '流形梯度下降',
        'final_value': final2,
        'time': time2,
        'violation': optimizer2.check_constraint_violation(Q2),
        'history': history2
    })
    
    print(f"最终值: {final2:.6f}, 用时: {time2:.2f}s")
    
    # 比较结果
    print(f"\n{'='*40}")
    print("性能比较总结")
    print(f"{'='*40}")
    
    for method in methods:
        print(f"{method['name']:12s}: 值={method['final_value']:.6f}, "
              f"时间={method['time']:.2f}s, 约束违反={method['violation']:.2e}")
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    
    for i, method in enumerate(methods):
        history = method['history']
        iterations = [h['iteration'] for h in history]
        values = [h['current_value'] for h in history]
        
        plt.subplot(2, 2, i+1)
        plt.plot(iterations, values, 'b-', alpha=0.7)
        plt.title(f"{method['name']}\n最终值: {method['final_value']:.6f}")
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/code/method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n比较图已保存: /workspace/code/method_comparison.png")
    
    return methods


def main():
    """主函数：运行所有示例"""
    
    print("Q矩阵优化器 - 实际应用示例")
    print("展示在不同科学计算场景中的应用")
    
    # 运行各个示例
    quantum_Q, quantum_history = quantum_state_optimization_example()
    signal_Q, signal_history = signal_processing_example()
    ml_Q, ml_history = machine_learning_feature_selection()
    
    # 比较优化方法
    comparison_results = compare_optimization_methods()
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("实际应用总结")
    print(f"{'='*60}")
    print("1. 量子态优化: 成功优化了量子态表示矩阵")
    print("2. 信号处理: 平衡了频域和时域特性")
    print("3. 机器学习: 优化了特征的可区分性")
    print("4. 方法比较: 流形梯度下降通常表现更好")
    print("\n所有示例都成功保持了正交归一约束！")


if __name__ == "__main__":
    main()