"""
针对您问题的完整解决方案
实现带正交归一约束的Q矩阵优化
"""

import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List, Dict
import time


class QMatrixOptimizer:
    """
    Q矩阵优化器 - 针对您的具体问题设计
    优化函数 run(Q)，其中 Q 满足正交归一约束
    """
    
    def __init__(self, run_function: Callable[[np.ndarray], float]):
        """
        初始化优化器
        
        Args:
            run_function: 您的目标函数 run(Q)
                         输入: shape为[2**n, 2**k]的复数numpy数组
                         输出: 实数
        """
        self.run_func = run_function
        self.optimization_history = []
        
    def initialize_Q(self, n: int, k: int, method: str = 'random', seed: Optional[int] = None) -> np.ndarray:
        """
        初始化Q矩阵
        
        Args:
            n, k: 矩阵维度参数，Q shape为[2**n, 2**k]
            method: 初始化方法 ('random', 'identity', 'custom')
            seed: 随机种子
            
        Returns:
            初始化的Q矩阵
        """
        rows, cols = 2**n, 2**k
        
        if seed is not None:
            np.random.seed(seed)
            
        if method == 'random':
            # 随机正交归一初始化
            Q = (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))
            Q, _ = qr(Q, mode='economic')
            
        elif method == 'identity':
            # 单位矩阵初始化（如果维度合适）
            if rows >= cols:
                Q = np.eye(rows, cols)
            else:
                Q = np.eye(rows, cols)
                
        else:
            raise ValueError(f"未知的初始化方法: {method}")
            
        return Q
    
    def project_to_stiefel_manifold(self, Q: np.ndarray) -> np.ndarray:
        """
        投影到Stiefel流形（正交归一约束）
        
        Args:
            Q: 输入矩阵
            
        Returns:
            投影后的正交归一矩阵
        """
        # 使用SVD分解进行投影
        U, _, Vh = svd(Q, full_matrices=False)
        Q_projected = U @ Vh
        return Q_projected
    
    def compute_numerical_gradient(self, Q: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        计算数值梯度
        
        Args:
            Q: 当前Q矩阵
            epsilon: 数值差分步长
            
        Returns:
            梯度矩阵
        """
        grad = np.zeros_like(Q, dtype=complex)
        
        # 遍历所有元素计算梯度
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                # 计算实部梯度
                Q_plus = Q.copy()
                Q_plus[i, j] += epsilon
                f_plus = self.run_func(Q_plus)
                
                Q_minus = Q.copy()
                Q_minus[i, j] -= epsilon
                f_minus = self.run_func(Q_minus)
                
                grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
        
        return grad
    
    def optimize_steepest_ascent(self,
                                n: int,
                                k: int,
                                learning_rate: float = 0.01,
                                max_iterations: int = 1000,
                                tolerance: float = 1e-6,
                                initial_method: str = 'random',
                                seed: Optional[int] = None,
                                save_history: bool = True,
                                verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        最速上升法优化（推荐方法）
        
        Args:
            n, k: 矩阵维度参数
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            initial_method: 初始矩阵生成方法
            seed: 随机种子
            save_history: 是否保存优化历史
            verbose: 是否打印进度信息
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        # 初始化
        Q = self.initialize_Q(n, k, initial_method, seed)
        
        if save_history:
            self.optimization_history = []
        
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        current_value = best_value
        
        if verbose:
            print(f"初始Q矩阵形状: {Q.shape}")
            print(f"初始run(Q)值: {best_value:.8f}")
            print(f"开始优化...")
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # 计算梯度
            gradient = self.compute_numerical_gradient(Q)
            
            # 梯度上升更新
            Q_new = Q + learning_rate * gradient
            
            # 投影回约束流形
            Q_new = self.project_to_stiefel_manifold(Q_new)
            
            # 计算新值
            new_value = self.run_func(Q_new)
            
            # 更新最佳值
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            # 记录历史
            if save_history:
                self.optimization_history.append({
                    'iteration': iteration,
                    'current_value': new_value,
                    'best_value': best_value,
                    'improvement': new_value - current_value,
                    'gradient_norm': np.linalg.norm(gradient)
                })
            
            # 检查收敛
            if abs(new_value - current_value) < tolerance:
                if verbose:
                    print(f"在第 {iteration} 轮收敛 (用时 {time.time() - start_time:.2f}秒)")
                break
            
            # 自适应学习率调整（可选）
            if iteration > 0 and iteration % 100 == 0:
                recent_improvements = [h['improvement'] for h in self.optimization_history[-10:]]
                if np.mean(recent_improvements) < 1e-8:
                    learning_rate *= 0.5
                    if verbose:
                        print(f"调整学习率: {learning_rate:.6f}")
            
            Q = Q_new
            current_value = new_value
            
            # 打印进度
            if verbose and iteration % 50 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {iteration:4d}: 当前值={current_value:.8f}, 最佳值={best_value:.8f}, "
                      f"梯度范数={np.linalg.norm(gradient):.2e}, 用时={elapsed:.2f}s")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"优化完成! 总用时: {total_time:.2f}秒")
            print(f"最终最佳值: {best_value:.8f}")
            print(f"最终约束违反程度: {self.check_constraint_violation(best_Q):.2e}")
        
        return best_Q, self.optimization_history
    
    def optimize_manifold_gradient(self,
                                  n: int,
                                  k: int,
                                  learning_rate: float = 0.01,
                                  max_iterations: int = 1000,
                                  tolerance: float = 1e-6,
                                  initial_method: str = 'random',
                                  seed: Optional[int] = None,
                                  save_history: bool = True,
                                  verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        流形梯度下降（更高效的约束优化方法）
        
        Args:
            n, k: 矩阵维度参数
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            initial_method: 初始矩阵生成方法
            seed: 随机种子
            save_history: 是否保存优化历史
            verbose: 是否打印进度信息
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        # 初始化
        Q = self.initialize_Q(n, k, initial_method, seed)
        
        if save_history:
            self.optimization_history = []
        
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        current_value = best_value
        
        if verbose:
            print(f"初始Q矩阵形状: {Q.shape}")
            print(f"初始run(Q)值: {best_value:.8f}")
            print(f"开始流形优化...")
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # 计算梯度
            gradient = self.compute_numerical_gradient(Q)
            
            # 投影到切空间（流形约束）
            gradient_tangent = gradient - Q @ (Q.conj().T @ gradient)
            
            # 流形上的梯度下降
            Q_new = Q + learning_rate * gradient_tangent
            Q_new = self.project_to_stiefel_manifold(Q_new)
            
            new_value = self.run_func(Q_new)
            
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            if save_history:
                self.optimization_history.append({
                    'iteration': iteration,
                    'current_value': new_value,
                    'best_value': best_value,
                    'improvement': new_value - current_value,
                    'gradient_norm': np.linalg.norm(gradient),
                    'tangent_gradient_norm': np.linalg.norm(gradient_tangent)
                })
            
            if abs(new_value - current_value) < tolerance:
                if verbose:
                    print(f"在第 {iteration} 轮收敛 (用时 {time.time() - start_time:.2f}秒)")
                break
            
            Q = Q_new
            current_value = new_value
            
            if verbose and iteration % 50 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {iteration:4d}: 当前值={current_value:.8f}, 最佳值={best_value:.8f}, "
                      f"切空间梯度范数={np.linalg.norm(gradient_tangent):.2e}, 用时={elapsed:.2f}s")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"流形优化完成! 总用时: {total_time:.2f}秒")
            print(f"最终最佳值: {best_value:.8f}")
            print(f"最终约束违反程度: {self.check_constraint_violation(best_Q):.2e}")
        
        return best_Q, self.optimization_history
    
    def check_constraint_violation(self, Q: np.ndarray) -> float:
        """
        检查约束违反程度
        
        Args:
            Q: Q矩阵
            
        Returns:
            约束违反程度（最大误差）
        """
        QTQ = Q.conj().T @ Q
        identity = np.eye(Q.shape[1])
        violation = np.max(np.abs(QTQ - identity))
        return violation
    
    def visualize_optimization(self, save_path: Optional[str] = None):
        """
        可视化优化过程
        
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        if not self.optimization_history:
            print("没有优化历史记录")
            return
        
        iterations = [h['iteration'] for h in self.optimization_history]
        current_values = [h['current_value'] for h in self.optimization_history]
        best_values = [h['best_value'] for h in self.optimization_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 值的变化
        ax1.plot(iterations, current_values, 'b-', alpha=0.7, label='当前值')
        ax1.plot(iterations, best_values, 'r-', linewidth=2, label='最佳值')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('run(Q) 值')
        ax1.set_title('优化过程中值的变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 梯度范数的变化
        gradient_norms = [h.get('gradient_norm', 0) for h in self.optimization_history]
        ax2.semilogy(iterations, gradient_norms, 'g-', label='梯度范数')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('梯度范数 (对数尺度)')
        ax2.set_title('梯度范数的变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"优化过程图已保存: {save_path}")
        else:
            plt.show()


# 使用示例
def example_usage():
    """使用示例"""
    
    # 定义您的目标函数
    def your_run_function(Q):
        """
        您的run函数示例
        输入: shape为[2**n, 2**k]的复数numpy数组
        输出: 实数
        """
        # 这里替换为您的实际run函数
        # 示例1: 最大化trace(Q @ Q^H)
        return np.real(np.trace(Q @ Q.conj().T))
        
        # 示例2: 最大化所有元素绝对值的平方和
        # return np.sum(np.abs(Q)**2)
        
        # 示例3: 最大化第一列的范数
        # return np.linalg.norm(Q[:, 0])**2
    
    # 创建优化器
    optimizer = QMatrixOptimizer(your_run_function)
    
    # 设置参数
    n, k = 3, 2  # Q shape为[8, 4]
    
    print("=== Q矩阵约束优化示例 ===")
    print(f"矩阵维度: 2^{n} x 2^{k} = {2**n} x {2**k}")
    print(f"约束: Q^H @ Q = I_{2**k}")
    print()
    
    # 方法1: 最速上升法
    print("方法1: 最速上升法")
    Q1, history1 = optimizer.optimize_steepest_ascent(
        n=n, k=k,
        learning_rate=0.01,
        max_iterations=500,
        initial_method='random',
        seed=42
    )
    print()
    
    # 方法2: 流形梯度下降
    print("方法2: 流形梯度下降")
    Q2, history2 = optimizer.optimize_manifold_gradient(
        n=n, k=k,
        learning_rate=0.01,
        max_iterations=500,
        initial_method='random',
        seed=42
    )
    print()
    
    # 可视化结果
    optimizer.visualize_optimization('/workspace/code/optimization_results.png')
    
    # 比较结果
    print("=== 结果比较 ===")
    print(f"最速上升法最终值: {history1[-1]['best_value']:.8f}")
    print(f"流形梯度下降最终值: {history2[-1]['best_value']:.8f}")
    print(f"约束违反程度 (最速上升): {optimizer.check_constraint_violation(Q1):.2e}")
    print(f"约束违反程度 (流形梯度): {optimizer.check_constraint_violation(Q2):.2e}")


if __name__ == "__main__":
    example_usage()