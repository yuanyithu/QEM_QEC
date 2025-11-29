import jax.numpy as jnp
import numpy as np
from jax import random
from jax.random import PRNGKey
import matplotlib.pyplot as plt
from QFI_single_round import max_eig, trivial_Q
from QFI_optimize import optimize_riemannian, show_state
import time

def expand_Q(Q_old, n_old, n_new):
    """
    扩展Q矩阵从n_old到n_new
    Q_old: shape (2**n_old, 2**k)
    返回: shape (2**n_new, 2**k) 的新矩阵
    """
    k = int(jnp.log2(Q_old.shape[1]))
    old_size = 2**n_old
    new_size = 2**n_new
    
    # 创建新的零矩阵
    Q_new = jnp.zeros((new_size, 2**k), dtype=jnp.complex64)
    # 复制旧矩阵到新矩阵的前old_size行
    Q_new = Q_new.at[:old_size, :].set(Q_old)
    
    return Q_new

def run_progressive_optimization(p, k=1, n_list=None, optimize_steps=100, 
                                 lr=0.02, gamma=0.005, decay=0.9, 
                                 key_seed=42, verbose=False):
    """
    执行渐进式优化
    
    参数:
        p: 噪声参数
        k: 固定为1
        n_list: n值列表
        optimize_steps: 每次优化的步数
        lr: 学习率
        gamma: 随机力强度
        decay: 速度衰减率
        key_seed: 随机种子
        verbose: 是否打印详细信息
    
    返回:
        n_values: n值列表
        eig_max_values: 对应的最大特征值列表
        Q_list: 每个n对应的优化后Q矩阵列表
    """
    # print("entering run_progressive_optimization with n_list = ",n_list)
    if n_list is None:
        n_list = [1, 2, 3, 4, 5]
    
    n_values = []
    eig_max_values = []
    Q_list = []
    
    # 从n=1开始，trivial初始化
    n = n_list[0]
    Q = trivial_Q(n)[:, :2**k]  # 确保只取前2**k列
    # print("Q.shape = ",Q.shape)
    # 计算初始特征值
    Q_real = jnp.real(Q)
    Q_imag = jnp.imag(Q)
    # print("第一次调用max_eig()函数")
    # print("k = ",k)
    # print("n = ",n)
    # print("p = ",p)
    # print("Q_real.shape = ",Q_real.shape)
    # print("Q_imag.shape = ",Q_imag.shape)
    eig_max = max_eig(Q_real, Q_imag, n, k, p)
    # print("第一次调用max_eig()函数结束")
    n_values.append(n)
    eig_max_values.append(float(eig_max))
    Q_list.append(Q)
    
    if verbose:
        print(f"n={n}, Initial eig_max={eig_max:.6f}")
    
    # 逐步扩展和优化
    for i in range(1, len(n_list)):
        n_old = n_list[i-1]
        n_new = n_list[i]
        
        # 扩展Q矩阵
        Q = expand_Q(Q, n_old, n_new)
        # print("准备开始优化 n_new = ",n_new," n_old = ",n_old)
        # 优化扩展后的Q矩阵
        Q_opt, costs = optimize_riemannian(
            Q_init=Q,
            p=p,
            n_steps=optimize_steps,
            lr=lr,
            gamma=gamma,
            decay=decay,
            verbose=False,
            key_seed=key_seed + i  # 每次使用不同的种子
        )
        
        # 计算优化后的特征值
        Q_real_opt = jnp.real(Q_opt)
        Q_imag_opt = jnp.imag(Q_opt)
        eig_max = max_eig(Q_real_opt, Q_imag_opt, n_new, k, p)
        
        n_values.append(n_new)
        eig_max_values.append(float(eig_max))
        Q_list.append(Q_opt)
        
        if verbose:
            print(f"n={n_new}, Optimized eig_max={eig_max:.6f}, "
                  f"Cost improvement={costs[-1]-costs[0]:.6f}")
        
        # 更新Q为优化后的版本，用于下一轮扩展
        Q = Q_opt
    
    return n_values, eig_max_values, Q_list

def plot_results(results_dict, save_path=None):
    """
    绘制所有结果
    
    参数:
        results_dict: 字典，键为p值，值为(n_values, eig_max_values_list)
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 8))
    
    # 定义颜色映射：从浅蓝到深蓝
    p_values = sorted(results_dict.keys())
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(p_values)))
    
    for i, p in enumerate(p_values):
        n_values, eig_max_lists = results_dict[p]
        
        # 计算每个n的平均值和标准差
        eig_max_array = np.array(eig_max_lists)
        mean_values = np.mean(eig_max_array, axis=0)
        std_values = np.std(eig_max_array, axis=0)
        
        # 绘制均值线
        plt.plot(n_values, mean_values, 'o-', color=colors[i], 
                 linewidth=2, markersize=8, label=f'p={p:.1f}')
        
        # 绘制误差带（标准差）
        plt.fill_between(n_values, 
                         mean_values - std_values, 
                         mean_values + std_values,
                         color=colors[i], alpha=0.2)
        
        # 绘制所有试验的散点（半透明）
        for trial_values in eig_max_lists:
            plt.scatter(n_values, trial_values, color=colors[i], 
                       alpha=0.3, s=20)
    
    plt.xlabel('Number of qubits (n)', fontsize=14)
    plt.ylabel('Maximum Eigenvalue', fontsize=14)
    plt.title('Progressive Q Expansion and Optimization', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数
    """
    # 参数设置
    p_list = [0,0.1,0.2,0.3,0.4]
    n_list = [1,2,3,4]  # 可以根据需要调整
    k = 1  # 固定k=1
    num_trials = 1  # 每个p执行的次数
    
    # 优化参数
    optimize_steps = 100  # 每次优化的步数
    lr = 0.02
    gamma = 0.005
    decay = 0.9
    
    # 存储所有结果
    all_results = {}
    
    print("=" * 60)
    print("开始渐进式Q扩展优化实验")
    print(f"参数: p_list={p_list}, n_list={n_list}, k={k}")
    print(f"每个p执行{num_trials}次试验")
    print("=" * 60)
    
    # 对每个p值进行实验
    for p in p_list:
        print(f"\n处理 p={p}...")
        eig_max_trials = []
        
        # 进行多次试验
        for trial in range(num_trials):
            print(f"  试验 {trial+1}/{num_trials}...")
            
            # 执行渐进式优化
            n_values, eig_max_values, Q_list = run_progressive_optimization(
                p=p,
                k=k,
                n_list=n_list,
                optimize_steps=optimize_steps,
                lr=lr,
                gamma=gamma,
                decay=decay,
                key_seed=42 + trial * 100,  # 每次试验使用不同的种子
                verbose=False
            )
            
            eig_max_trials.append(eig_max_values)
            
            # 打印最终结果
            print(f"    最终 (n={n_list[-1]}): eig_max={eig_max_values[-1]:.6f}")
        
        # 存储这个p的所有结果
        all_results[p] = (n_values, eig_max_trials)
        
        # 打印统计信息
        mean_final = np.mean([trial[-1] for trial in eig_max_trials])
        std_final = np.std([trial[-1] for trial in eig_max_trials])
        print(f"  p={p} 最终平均: {mean_final:.6f} ± {std_final:.6f}")
    
    print("\n" + "=" * 60)
    print("实验完成！正在绘制结果...")
    
    # 绘制结果
    plot_results(all_results, save_path='progressive_optimization_results.png')
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("最终汇总:")
    for p in p_list:
        n_values, eig_max_trials = all_results[p]
        final_values = [trial[-1] for trial in eig_max_trials]
        mean_final = np.mean(final_values)
        std_final = np.std(final_values)
        print(f"p={p:.1f}: 最终平均特征值 = {mean_final:.6f} ± {std_final:.6f}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    main()