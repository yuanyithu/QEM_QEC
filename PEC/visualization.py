import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker as ticker

def setup_plot_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'stix', 
        'font.sans-serif': ['Arial'] 
    })

def plot_survival_probability(L_list, p_list, probs_matrix):
    """
    绘制图1：后选择概率误差 (1 - Probability)
    保持不变，或根据需要同样应用 log 坐标
    """
    L_list = np.array(L_list)
    p_list = np.array(p_list)
    data_matrix = 1 - np.array(probs_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=p_list.min(), vmax=p_list.max())
    
    for i, p in enumerate(p_list):
        color = cmap(norm(p))
        ax.plot(L_list, data_matrix[i, :], color=color, marker='o', alpha=0.8)


    ax.set_xlabel(r"Circuit Depth ($L$)")
    ax.set_ylabel(r"$1 - P_{success}$")
    ax.set_title("Post-selection Failure Rate vs. Depth")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'Physical Error Rate ($p$)')

    plt.tight_layout()
    plt.show()

def add_direct_labels(ax, x_vals, y_vals, p_val, color):
    """
    辅助函数：在曲线末端添加带有颜色的数值标签
    """
    x_end = x_vals[-1]
    y_end = y_vals[-1]
    
    text = f"p={p_val:.1e}"

    ax.text(
        x_end + (x_vals[-1] - x_vals[0]) * 0.02,
        y_end,
        text, 
        color=color, 
        fontsize=10, 
        verticalalignment='center',
        fontweight='bold'
    )

def plot_cost_comparison(L_list, p_list, pec_matrix, total_matrix):
    """
    修改版：使用真实的对数坐标轴 (Log Scale Axis)
    """
    L_list = np.array(L_list)
    p_list = np.array(p_list)
    

    pec_matrix = np.array(pec_matrix)
    total_matrix = np.array(total_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)) # 去掉 sharey=True，防止两边范围差异过大影响观察，如果需要对齐可保留
    
    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=p_list.min(), vmax=p_list.max())

    indices_to_label = [0, len(p_list)//2, len(p_list)-1]

    for i, p in enumerate(p_list):
        color = cmap(norm(p))
        
        if i in indices_to_label:
            lw = 2.5
            alpha = 1.0
            zorder = 10
        else:
            lw = 1.0
            alpha = 0.6
            zorder = 1
            
        ax1.plot(L_list, pec_matrix[i, :], color=color, linewidth=lw, alpha=alpha, marker='.', zorder=zorder)
        ax2.plot(L_list, total_matrix[i, :], color=color, linewidth=lw, alpha=alpha, marker='.', zorder=zorder)

        if i in indices_to_label:
            add_direct_labels(ax1, L_list, pec_matrix[i, :], p, color)
            add_direct_labels(ax2, L_list, total_matrix[i, :], p, color)


    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.set_title(r"(a) Intrinsic PEC Sample Cost", loc='left', fontweight='bold')
    ax1.set_xlabel(r"Circuit Depth ($L$)")

    ax1.set_ylabel(r"Sample Cost (Log Scale)") 
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    
    ax2.set_title(r"(b) Total Cost (w/ Overhead)", loc='left', fontweight='bold')
    ax2.set_xlabel(r"Circuit Depth ($L$)")
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)


    xlim_max = L_list[-1] + (L_list[-1] - L_list[0]) * 0.25 
    ax1.set_xlim(L_list[0], xlim_max)
    ax2.set_xlim(L_list[0], xlim_max)


    fig.subplots_adjust(right=0.85, wspace=0.2)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    

    cbar.set_ticks([p_list.min(), np.median(p_list), p_list.max()])
    cbar.set_ticklabels([f'{p_list.min():.1e}', f'{np.median(p_list):.1e}', f'{p_list.max():.1e}'])
    cbar.set_label(r'Physical Error Rate ($p$)')

    plt.suptitle("Scaling Analysis of Subspace PEC", fontsize=16, y=0.98)
    plt.show()


def plot_layered_heatmap(L_list, L_check_list, total_samples_matrix, p_val):
    """
    新增函数：绘制热力图，展示在固定 p 下，L 和 L_check 对总 Sample Cost 的影响。
    total_samples_matrix shape: (len(L_list), len(L_check_list))
    """
    L_grid, L_check_grid = np.meshgrid(L_list, L_check_list)
    
    # 转置矩阵以匹配 meshgrid (y, x)
    data = total_samples_matrix.T 
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # 使用 LogNorm 因为 Sample Cost 变化范围通常巨大
    pcm = ax.pcolormesh(L_grid, L_check_grid, data, 
                        norm=mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data)),
                        cmap='viridis', shading='auto')
    
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Total Sample Cost')
    
    ax.set_xlabel(r"Total Circuit Depth ($L$)")
    ax.set_ylabel(r"Check Interval ($L_{check}$)")
    ax.set_title(f"Cost Landscape (p={p_val:.1e})\nDarker is Better", fontweight='bold')
    
    # 可以在图上画出一条“最优路径”
    optimal_indices = np.argmin(total_samples_matrix, axis=1)
    optimal_L_checks = np.array([L_check_list[i] for i in optimal_indices])
    ax.plot(L_list, optimal_L_checks, 'r--', label='Optimal $L_{check}$', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_check_interval_cuts(L_check_list, total_samples_matrix, L_list, p_val, select_L_indices):
    """
    新增函数：在固定 p 下，选取几个特定的总深度 L，画出 Cost vs L_check 的曲线。
    用于观察是否存在最优的检查频率。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cmap = cm.autumn
    norm = mcolors.Normalize(vmin=0, vmax=len(select_L_indices)-1)

    for idx, L_idx in enumerate(select_L_indices):
        L_val = L_list[L_idx]
        cost_curve = total_samples_matrix[L_idx, :] # (len(L_check_list),)
        
        color = cmap(norm(idx))
        ax.plot(L_check_list, cost_curve, color=color, label=f"Total Depth L={L_val}", marker='o', markersize=3)
        
        # 标记最小值
        min_idx = np.argmin(cost_curve)
        ax.scatter(L_check_list[min_idx], cost_curve[min_idx], s=100, facecolors='none', edgecolors='blue', zorder=10)

    ax.set_yscale('log')
    ax.set_xlabel(r"Check Interval ($L_{check}$)")
    ax.set_ylabel("Total Sample Cost")
    ax.set_title(f"Impact of Check Interval (p={p_val:.1e})")
    ax.legend()
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()