"""
并行版本：基于 order_1_GHZ_preparation.py，使用 multiprocessing 并行计算每个 T 的 sample。
保留原有逻辑，仅加速独立的 protocol2_sample_cost 计算；可通过 max_processes 参数控制最大并行进程数。
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from time import time

from order_1_GHZ_preparation import protocol2_sample_cost, pure_PEC_sample_cost
from order_1_plotting import set_publication_style, plot_vs_T, plot_vs_n


def compute_samples_parallel(T_list, n, p_single, p_double, max_processes=None):
    """
    并行计算不同 T 对应的 protocol2_sample_cost，返回与 T_list 顺序一致的结果列表。
    max_processes 用于限制最大并行进程数；None 或小于 1 时取 min(len(T_list), cpu_count())。
    """
    if (max_processes is None) or (max_processes < 1):
        pool_size = min(len(T_list), cpu_count())
    else:
        pool_size = min(len(T_list), max_processes)

    tasks = [(n, p_single, p_double, int(T)) for T in T_list]
    with Pool(processes=pool_size) as pool:
        results = pool.starmap(protocol2_sample_cost, tasks)
    return results


def compute_samples_over_n_parallel(n_list, p_single, p_double, T, max_processes=None):
    """
    并行计算不同 n 对应的 protocol2_sample_cost（固定 T）。
    返回与 n_list 顺序一致的结果列表。
    max_processes 用于限制最大并行进程数；None 或小于 1 时取 min(len(n_list), cpu_count())。
    """
    if (max_processes is None) or (max_processes < 1):
        pool_size = min(len(n_list), cpu_count())
    else:
        pool_size = min(len(n_list), max_processes)

    tasks = [(int(n), p_single, p_double, T) for n in n_list]
    with Pool(processes=pool_size) as pool:
        results = pool.starmap(protocol2_sample_cost, tasks)
    return results


if __name__ == "__main__":
    p_single = 1e-4
    p_double = 1e-5
    set_publication_style()

    # 设定最大并行进程数，None 表示使用 cpu_count() 但不超过任务数
    max_processes = None

    # 绘图 1: 固定 n，扫描 T（并行加速）
    n_fixed = 100
    T_min = 1
    T_max = int(0.5/(3*n_fixed*p_single + 9*(n_fixed-1)*p_double)) // 4
    T_list = np.arange(T_min, max(T_min+1, T_max))
    start_time = time()
    protocol2_sample_T = compute_samples_parallel(T_list, n_fixed, p_single, p_double, max_processes=max_processes)
    pure_sample_T = [pure_PEC_sample_cost(n_fixed - 2, p_single, p_double)] * len(T_list)
    end_time = time()
    print("run time (sweep T, parallel) = ", end_time - start_time, " s")
    plot_vs_T(T_list, protocol2_sample_T, pure_sample_T, n_fixed, p_single, p_double, logy=True, show=False)

    # 绘图 2: 固定 T，扫描 n（并行加速）
    T_fixed = 1
    n_list = np.arange(50, 201, 10)  # 自行调整 n 范围
    start_time = time()
    protocol2_sample_n = compute_samples_over_n_parallel(n_list, p_single, p_double, T_fixed, max_processes=max_processes)
    pure_sample_n = [pure_PEC_sample_cost(int(n) - 2, p_single, p_double) for n in n_list]
    end_time = time()
    print("run time (sweep n, parallel) = ", end_time - start_time, " s")
    plot_vs_n(n_list, protocol2_sample_n, pure_sample_n, T_fixed, p_single, p_double, logy=True, show=False)

    plt.show()
