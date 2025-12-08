"""
带正交归一约束的矩阵优化算法
用于优化函数 run(Q)，其中 Q 满足 ∑Q[p,i].conj()*Q[p,j] = δ_ij
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
import time


class ConstrainedQOptimizer:
    """
    带正交归一约束的Q矩阵优化器
    """
    
    def __init__(self, run_func: Callable, n: int, k: int):
        """
        初始化优化器
        
        Args:
            run_func: 目标函数 run(Q)，输入shape为[2**n, 2**k]的复数numpy数组，输出实数
            n: 指数参数，决定矩阵的行数 2**n
            k: 指数参数，决定矩阵的列数 2**k
        """
        self.run_func = run_func
        self.n = n
        self.k = k
        self.rows = 2**n
        self.cols = 2**k
        self.history = []
        
    def random_orthogonal_init(self, seed: Optional[int] = None) -> np.ndarray:
        """
        生成随机正交归一矩阵作为初始值
        
        Args:
            seed: 随机种子
            
        Returns:
            shape为[rows, cols]的正交归一复数矩阵
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 生成随机复数矩阵
        Q = (np.random.randn(self.rows, self.cols) + 
             1j * np.random.randn(self.rows, self.cols))
        
        # QR分解确保正交归一
        Q, _ = qr(Q, mode='economic')
        
        return Q
    
    def project_to_orthogonal(self, Q: np.ndarray) -> np.ndarray:
        """
        将矩阵投影到正交归一约束空间
        
        Args:
            Q: 输入矩阵
            
        Returns:
            投影后的正交归一矩阵
        """
        # 使用SVD分解进行投影
        U, _, Vh = svd(Q, full_matrices=False)
        Q_proj = U @ Vh
        return Q_proj
    
    def gradient_descent_optimize(self, 
                                learning_rate: float = 0.01,
                                max_iterations: int = 1000,
                                tolerance: float = 1e-6,
                                initial_Q: Optional[np.ndarray] = None,
                                save_history: bool = True) -> Tuple[np.ndarray, dict]:
        """
        梯度下降优化方法
        
        Args:
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            initial_Q: 初始Q矩阵，如果为None则随机生成
            save_history: 是否保存历史记录
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        if initial_Q is None:
            Q = self.random_orthogonal_init()
        else:
            Q = initial_Q.copy()
        
        if save_history:
            self.history = []
        
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        
        print(f"初始值: {best_value:.6f}")
        
        for iteration in range(max_iterations):
            # 计算梯度（数值方法）
            epsilon = 1e-8
            grad = np.zeros_like(Q, dtype=complex)
            
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
            
            # 梯度下降更新
            Q_new = Q - learning_rate * grad
            
            # 投影回正交归一约束
            Q_new = self.project_to_orthogonal(Q_new)
            
            # 计算新值
            new_value = self.run_func(Q_new)
            
            # 更新最佳值
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            if save_history:
                self.history.append({
                    'iteration': iteration,
                    'value': new_value,
                    'best_value': best_value,
                    'improvement': new_value - self.run_func(Q)
                })
            
            # 检查收敛
            if abs(new_value - self.run_func(Q)) < tolerance:
                print(f"在第 {iteration} 轮收敛")
                break
            
            Q = Q_new
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}: 当前值 = {new_value:.6f}, 最佳值 = {best_value:.6f}")
        
        print(f"最终结果: {best_value:.6f}")
        return best_Q, self.history
    
    def projected_gradient_descent(self,
                                 learning_rate: float = 0.01,
                                 max_iterations: int = 1000,
                                 tolerance: float = 1e-6,
                                 initial_Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        投影梯度下降方法
        
        Args:
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            initial_Q: 初始Q矩阵
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        if initial_Q is None:
            Q = self.random_orthogonal_init()
        else:
            Q = initial_Q.copy()
        
        self.history = []
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        
        print(f"初始值: {best_value:.6f}")
        
        for iteration in range(max_iterations):
            # 计算梯度
            epsilon = 1e-8
            grad = np.zeros_like(Q, dtype=complex)
            
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    Q_plus = Q.copy()
                    Q_plus[i, j] += epsilon
                    f_plus = self.run_func(Q_plus)
                    
                    Q_minus = Q.copy()
                    Q_minus[i, j] -= epsilon
                    f_minus = self.run_func(Q_minus)
                    
                    grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
            
            # 投影梯度下降
            Q_temp = Q - learning_rate * grad
            Q_new = self.project_to_orthogonal(Q_temp)
            
            new_value = self.run_func(Q_new)
            
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            self.history.append({
                'iteration': iteration,
                'value': new_value,
                'best_value': best_value
            })
            
            if abs(new_value - self.run_func(Q)) < tolerance:
                print(f"在第 {iteration} 轮收敛")
                break
            
            Q = Q_new
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}: 当前值 = {new_value:.6f}, 最佳值 = {best_value:.6f}")
        
        print(f"最终结果: {best_value:.6f}")
        return best_Q, self.history
    
    def manifold_gradient_descent(self,
                                learning_rate: float = 0.01,
                                max_iterations: int = 1000,
                                tolerance: float = 1e-6,
                                initial_Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        流形梯度下降方法（更高效的约束优化）
        
        Args:
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            initial_Q: 初始Q矩阵
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        if initial_Q is None:
            Q = self.random_orthogonal_init()
        else:
            Q = initial_Q.copy()
        
        self.history = []
        best_Q = Q.copy()
        best_value = self.run_func(Q)
        
        print(f"初始值: {best_value:.6f}")
        
        for iteration in range(max_iterations):
            # 计算梯度
            epsilon = 1e-8
            grad = np.zeros_like(Q, dtype=complex)
            
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    Q_plus = Q.copy()
                    Q_plus[i, j] += epsilon
                    f_plus = self.run_func(Q_plus)
                    
                    Q_minus = Q.copy()
                    Q_minus[i, j] -= epsilon
                    f_minus = self.run_func(Q_minus)
                    
                    grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
            
            # 流形上的梯度下降
            # 计算切空间投影
            grad_proj = grad - Q @ (Q.conj().T @ grad)
            
            # 更新
            Q_new = Q - learning_rate * grad_proj
            Q_new = self.project_to_orthogonal(Q_new)
            
            new_value = self.run_func(Q_new)
            
            if new_value > best_value:
                best_value = new_value
                best_Q = Q_new.copy()
            
            self.history.append({
                'iteration': iteration,
                'value': new_value,
                'best_value': best_value
            })
            
            if abs(new_value - self.run_func(Q)) < tolerance:
                print(f"在第 {iteration} 轮收敛")
                break
            
            Q = Q_new
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}: 当前值 = {new_value:.6f}, 最佳值 = {best_value:.6f}")
        
        print(f"最终结果: {best_value:.6f}")
        return best_Q, self.history
    
    def scipy_optimize(self,
                      method: str = 'L-BFGS-B',
                      max_iterations: int = 1000,
                      initial_Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        使用scipy优化器进行约束优化
        
        Args:
            method: 优化方法
            max_iterations: 最大迭代次数
            initial_Q: 初始Q矩阵
            
        Returns:
            优化后的Q矩阵和优化历史
        """
        if initial_Q is None:
            Q_init = self.random_orthogonal_init()
        else:
            Q_init = initial_Q.copy()
        
        # 将复数矩阵转换为实数向量
        def objective(x):
            Q = x[:Q_init.size//2].reshape(Q_init.shape) + 1j * x[Q_init.size//2:].reshape(Q_init.shape)
            return -self.run_func(Q)  # 负号因为scipy是最小化
        
        # 初始向量
        x0 = np.concatenate([Q_init.real.flatten(), Q_init.imag.flatten()])
        
        # 约束条件（正交归一）
        def constraint(x):
            Q = x[:Q_init.size//2].reshape(Q_init.shape) + 1j * x[Q_init.size//2:].reshape(Q_init.shape)
            return np.diag(Q.conj().T @ Q) - 1  # 对角线元素应为1
        
        constraints = [{'type': 'eq', 'fun': constraint}]
        
        # 优化
        result = minimize(
            objective, 
            x0, 
            method=method,
            constraints=constraints,
            options={'maxiter': max_iterations}
        )
        
        # 恢复Q矩阵
        Q_opt = result.x[:Q_init.size//2].reshape(Q_init.shape) + 1j * result.x[Q_init.size//2:].reshape(Q_init.shape)
        Q_opt = self.project_to_orthogonal(Q_opt)
        
        final_value = self.run_func(Q_opt)
        print(f"Scipy优化结果: {final_value:.6f}")
        
        return Q_opt, {'optimization_result': result}
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        if not self.history:
            print("没有优化历史记录")
            return
        
        iterations = [h['iteration'] for h in self.history]
        values = [h['value'] for h in self.history]
        best_values = [h['best_value'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, values, 'b-', label='当前值', alpha=0.7)
        plt.plot(iterations, best_values, 'r-', label='最佳值', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('run(Q) 值')
        plt.title('优化过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
    
    def verify_constraint(self, Q: np.ndarray) -> bool:
        """
        验证Q是否满足正交归一约束
        
        Args:
            Q: 待验证的矩阵
            
        Returns:
            是否满足约束
        """
        # 计算 Q^H * Q
        QTQ = Q.conj().T @ Q
        
        # 检查是否接近单位矩阵
        identity = np.eye(self.cols)
        error = np.max(np.abs(QTQ - identity))
        
        print(f"约束违反程度: {error:.2e}")
        return error < 1e-10


def example_usage():
    """
    使用示例
    """
    # 定义示例目标函数
    def example_run_func(Q):
        """示例目标函数：最大化trace(Q @ Q^H)"""
        return np.real(np.trace(Q @ Q.conj().T))
    
    # 创建优化器
    n, k = 3, 2  # 8x4矩阵
    optimizer = ConstrainedQOptimizer(example_run_func, n, k)
    
    # 生成初始矩阵
    Q_init = optimizer.random_orthogonal_init(seed=42)
    print("初始约束验证:")
    optimizer.verify_constraint(Q_init)
    
    print("\n=== 梯度下降优化 ===")
    Q_gd, history_gd = optimizer.gradient_descent_optimize(
        learning_rate=0.01,
        max_iterations=500,
        initial_Q=Q_init.copy()
    )
    
    print("\n=== 流形梯度下降优化 ===")
    Q_mgd, history_mgd = optimizer.manifold_gradient_descent(
        learning_rate=0.01,
        max_iterations=500,
        initial_Q=Q_init.copy()
    )
    
    print("\n=== 最终约束验证 ===")
    optimizer.verify_constraint(Q_gd)
    optimizer.verify_constraint(Q_mgd)
    
    # 绘制优化历史
    optimizer.plot_optimization_history('/workspace/code/optimization_history.png')


if __name__ == "__main__":
    example_usage()