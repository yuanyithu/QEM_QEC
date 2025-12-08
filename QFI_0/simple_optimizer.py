"""
简洁版约束优化算法
提供最实用的优化方法
"""

import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List, Dict


class SimpleQOptimizer:
    """简洁的Q矩阵优化器"""
    
    def __init__(self, run_func: Callable[[np.ndarray], float]):
        """
        初始化优化器
        
        Args:
            run_func: 目标函数，输入shape为[M,N]的复数numpy数组，输出float
        """
        self.run_func = run_func
        self.history = []
    
    def random_orthogonal(self, rows: int, cols: int, seed: Optional[int] = None) -> np.ndarray:
        """生成随机正交归一矩阵"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成随机复数矩阵
        Q = (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))
        
        # QR分解确保正交归一
        Q, _ = qr(Q, mode='economic')
        return Q
    
    def project_orthogonal(self, Q: np.ndarray) -> np.ndarray:
        """投影到正交归一约束"""
        U, _, Vh = svd(Q, full_matrices=False)
        return U @ Vh
    
    def numerical_gradient(self, Q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """计算数值梯度"""
        grad = np.zeros_like(Q, dtype=complex)
        
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                # 实部梯度
                Q_plus = Q.copy()
                Q_plus[i, j] += eps
                f_plus = self.run_func(Q_plus)
                
                Q_minus = Q.copy()
                Q_minus[i, j] -= eps
                f_minus = self.run_func(Q_minus)
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def optimize_projected_gradient(self,
                                  rows: int,
                                  cols: int,
                                  learning_rate: float = 0.01,
                                  max_iter: int = 1000,
                                  tolerance: float = 1e-6,
                                  seed: Optional[int] = None,
                                  verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        投影梯度下降优化（推荐方法）
        
        Args:
            rows, cols: Q矩阵的维度
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tolerance: 收敛容忍度
            seed: 随机种子
            verbose: 是否打印进度
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        # 初始化
        Q = self.random_orthogonal(rows, cols, seed)
        self.history = []
        
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        
        if verbose:
            print(f"初始值: {best_value:.6f}")
        
        for iteration in range(max_iter):
            # 计算梯度
            grad = self.numerical_gradient(Q)
            
            # 投影梯度下降
            Q_temp = Q - learning_rate * grad
            Q_new = self.project_orthogonal(Q_temp)
            
            # 计算新值
            new_value = self.run_func(Q_new)
            
            # 更新最佳值
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            # 记录历史
            self.history.append({
                'iter': iteration,
                'value': new_value,
                'best': best_value,
                'improvement': new_value - self.run_func(Q)
            })
            
            # 检查收敛
            if abs(new_value - self.run_func(Q)) < tolerance:
                if verbose:
                    print(f"第 {iteration} 轮收敛")
                break
            
            Q = Q_new
            
            # 打印进度
            if verbose and iteration % 100 == 0:
                print(f"迭代 {iteration}: 当前={new_value:.6f}, 最佳={best_value:.6f}")
        
        if verbose:
            print(f"最终结果: {best_value:.6f}")
        
        return best_Q, self.history
    
    def optimize_manifold_gradient(self,
                                 rows: int,
                                 cols: int,
                                 learning_rate: float = 0.01,
                                 max_iter: int = 1000,
                                 tolerance: float = 1e-6,
                                 seed: Optional[int] = None,
                                 verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        流形梯度下降（更高效）
        
        Args:
            rows, cols: Q矩阵的维度
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tolerance: 收敛容忍度
            seed: 随机种子
            verbose: 是否打印进度
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        # 初始化
        Q = self.random_orthogonal(rows, cols, seed)
        self.history = []
        
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        
        if verbose:
            print(f"初始值: {best_value:.6f}")
        
        for iteration in range(max_iter):
            # 计算梯度
            grad = self.numerical_gradient(Q)
            
            # 流形上的切空间投影
            grad_proj = grad - Q @ (Q.conj().T @ grad)
            
            # 更新并投影
            Q_temp = Q - learning_rate * grad_proj
            Q_new = self.project_orthogonal(Q_temp)
            
            new_value = self.run_func(Q_new)
            
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            self.history.append({
                'iter': iteration,
                'value': new_value,
                'best': best_value
            })
            
            if abs(new_value - self.run_func(Q)) < tolerance:
                if verbose:
                    print(f"第 {iteration} 轮收敛")
                break
            
            Q = Q_new
            
            if verbose and iteration % 100 == 0:
                print(f"迭代 {iteration}: 当前={new_value:.6f}, 最佳={best_value:.6f}")
        
        if verbose:
            print(f"最终结果: {best_value:.6f}")
        
        return best_Q, self.history
    
    def plot_history(self, save_path: Optional[str] = None):
        """绘制优化历史"""
        if not self.history:
            print("无历史记录")
            return
        
        iterations = [h['iter'] for h in self.history]
        values = [h['value'] for h in self.history]
        best_values = [h['best'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, values, 'b-', alpha=0.7, label='当前值')
        plt.plot(iterations, best_values, 'r-', linewidth=2, label='最佳值')
        plt.xlabel('迭代次数')
        plt.ylabel('run(Q) 值')
        plt.title('优化过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存: {save_path}")
        else:
            plt.show()
    
    def check_constraint(self, Q: np.ndarray) -> float:
        """检查约束满足程度"""
        QTQ = Q.conj().T @ Q
        identity = np.eye(Q.shape[1])
        error = np.max(np.abs(QTQ - identity))
        print(f"约束违反程度: {error:.2e}")
        return error


# 使用示例和测试函数
def create_example_functions():
    """创建示例目标函数"""
    
    def example_func1(Q):
        """示例1: 最大化trace(Q @ Q^H)"""
        return np.real(np.trace(Q @ Q.conj().T))
    
    def example_func2(Q):
        """示例2: 最大化所有元素的平方和"""
        return np.sum(np.abs(Q)**2)
    
    def example_func3(Q):
        """示例3: 最大化第一列的范数"""
        return np.linalg.norm(Q[:, 0])**2
    
    return example_func1, example_func2, example_func3


def demo_optimization():
    """演示优化过程"""
    print("=== Q矩阵约束优化演示 ===\n")
    
    # 创建示例函数
    func1, func2, func3 = create_example_functions()
    
    # 测试参数
    n, k = 3, 2  # 8x4矩阵
    rows, cols = 2**n, 2**k
    
    print(f"矩阵维度: {rows}x{cols}")
    print(f"约束: Q^H @ Q = I_{cols}")
    print()
    
    # 测试不同目标函数
    for i, func in enumerate([func1, func2, func3], 1):
        print(f"--- 测试目标函数 {i} ---")
        
        optimizer = SimpleQOptimizer(func)
        
        # 投影梯度下降
        print("投影梯度下降:")
        Q_pg, history_pg = optimizer.optimize_projected_gradient(
            rows, cols, learning_rate=0.01, max_iter=300, verbose=False
        )
        optimizer.check_constraint(Q_pg)
        
        # 流形梯度下降
        print("流形梯度下降:")
        Q_mg, history_mg = optimizer.optimize_manifold_gradient(
            rows, cols, learning_rate=0.01, max_iter=300, verbose=False
        )
        optimizer.check_constraint(Q_mg)
        print()


if __name__ == "__main__":
    demo_optimization()