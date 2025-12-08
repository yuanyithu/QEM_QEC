import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit


def setup_plot_style():
    """Lightweight plot style used by early scripts."""
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
        'font.sans-serif': ['Arial'],
    })


def setup_nature_style():
    """Matplotlib style similar to Nature journal aesthetics."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.figsize': (5, 4),
    })


def linear_func(x, a, b):
    """Linear function for curve fitting: y = a*x + b."""
    return a * x + b


def _select_p_slice(matrix, name="matrix", p_index=0):
    """
    Ensure matrices share the shape (len(L_check), len(L_list)).
    If a probability axis exists, slice the requested index.
    """
    arr = np.asarray(matrix)
    if arr.ndim == 3:
        if arr.shape[1] <= p_index:
            raise IndexError(f"{name} second axis too small for p_index={p_index}")
        if arr.shape[1] > 1 and p_index == 0:
            print(f"Note: {name} has multiple probability entries; using p_index={p_index}.")
        arr = arr[:, p_index, :]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")
    return arr


def plot_survival_probability(L_list, p_list, probs_matrix):
    """Plot post-selection failure rate vs depth for different physical error rates."""
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
    """Add labels to the end of curves to highlight specific p values."""
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
        fontweight='bold',
    )


def plot_cost_comparison(L_list, p_list, pec_matrix, total_matrix):
    """Plot intrinsic PEC cost and total cost on log scale for different error rates."""
    L_list = np.array(L_list)
    p_list = np.array(p_list)
    pec_matrix = np.array(pec_matrix)
    total_matrix = np.array(total_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=p_list.min(), vmax=p_list.max())
    indices_to_label = [0, len(p_list) // 2, len(p_list) - 1]

    for i, p in enumerate(p_list):
        color = cmap(norm(p))
        if i in indices_to_label:
            lw, alpha, zorder = 2.5, 1.0, 10
        else:
            lw, alpha, zorder = 1.0, 0.6, 1

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
    """Heatmap of total sample cost vs L and L_check for a fixed p."""
    L_grid, L_check_grid = np.meshgrid(L_list, L_check_list)
    data = np.asarray(total_samples_matrix).T

    fig, ax = plt.subplots(figsize=(9, 7))
    pcm = ax.pcolormesh(
        L_grid,
        L_check_grid,
        data,
        norm=mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data)),
        cmap='viridis',
        shading='auto',
    )

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Total Sample Cost')

    ax.set_xlabel(r"Total Circuit Depth ($L$)")
    ax.set_ylabel(r"Check Interval ($L_{check}$)")
    ax.set_title(f"Cost Landscape (p={p_val:.1e})\nDarker is Better", fontweight='bold')

    optimal_indices = np.argmin(total_samples_matrix, axis=1)
    optimal_L_checks = np.array([L_check_list[i] for i in optimal_indices])
    ax.plot(L_list, optimal_L_checks, 'r--', label='Optimal $L_{check}$', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_check_interval_cuts(L_check_list, total_samples_matrix, L_list, p_val, select_L_indices):
    """Plot cost vs L_check for selected total depths."""
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = cm.autumn
    norm = mcolors.Normalize(vmin=0, vmax=len(select_L_indices) - 1)

    for idx, L_idx in enumerate(select_L_indices):
        L_val = L_list[L_idx]
        cost_curve = total_samples_matrix[L_idx, :]
        color = cmap(norm(idx))
        ax.plot(L_check_list, cost_curve, color=color, label=f"Total Depth L={L_val}", marker='o', markersize=3)

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


def plot_inverse_probability(L_list, L_check_list, prob_matrix, save_path='inverse_probability.png', p_index=0):
    """Plot 1/probability vs L with log-scale y-axis; supports 2D or 3D input."""
    setup_nature_style()
    fig, ax = plt.subplots()

    data = _select_p_slice(prob_matrix, "prob_matrix", p_index)
    L_list = np.array(L_list)
    L_check_list = np.array(L_check_list)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())

    for i, L_check in enumerate(L_check_list):
        inv_prob = 1.0 / data[i, :]
        color = cmap(norm(L_check))
        ax.scatter(L_list, inv_prob, color=color, s=15, alpha=0.7, zorder=3)

        log_inv_prob = np.log10(inv_prob)
        valid_mask = np.isfinite(log_inv_prob)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_inv_prob[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10 ** log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('$1/P$')
    ax.set_title('Inverse Probability vs Circuit Depth')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '')
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pec_sample(L_list, L_check_list, pec_matrix, save_path='pec_sample.png', p_index=0):
    """Plot PEC samples vs L with log-scale y-axis; supports 2D or 3D input."""
    setup_nature_style()
    fig, ax = plt.subplots()

    data = _select_p_slice(pec_matrix, "pec_matrix", p_index)
    L_list = np.array(L_list)
    L_check_list = np.array(L_check_list)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())

    for i, L_check in enumerate(L_check_list):
        pec = data[i, :]
        color = cmap(norm(L_check))
        ax.scatter(L_list, pec, color=color, s=15, alpha=0.7, zorder=3)

        log_pec = np.log10(pec)
        valid_mask = np.isfinite(log_pec)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_pec[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10 ** log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('PEC Sample')
    ax.set_title('PEC Sample vs Circuit Depth')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '')
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_total_sample(L_list, L_check_list, total_matrix, save_path='total_sample.png', p_index=0):
    """Plot total sample vs L with log-scale y-axis; supports 2D or 3D input."""
    setup_nature_style()
    fig, ax = plt.subplots()

    data = _select_p_slice(total_matrix, "total_matrix", p_index)
    L_list = np.array(L_list)
    L_check_list = np.array(L_check_list)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())

    for i, L_check in enumerate(L_check_list):
        total = data[i, :]
        color = cmap(norm(L_check))
        ax.scatter(L_list, total, color=color, s=15, alpha=0.7, zorder=3)

        log_total = np.log10(total)
        valid_mask = np.isfinite(log_total)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_total[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10 ** log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('Total Sample')
    ax.set_title('Total Sample vs Circuit Depth')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else '')
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_all(L_list, L_check_list, prob_matrix, pec_matrix, total_matrix, prefix='', p_index=0):
    """Convenience wrapper to generate all three log-scale plots."""
    plot_inverse_probability(L_list, L_check_list, prob_matrix, save_path=f'{prefix}inverse_probability.png', p_index=p_index)
    plot_pec_sample(L_list, L_check_list, pec_matrix, save_path=f'{prefix}pec_sample.png', p_index=p_index)
    plot_total_sample(L_list, L_check_list, total_matrix, save_path=f'{prefix}total_sample.png', p_index=p_index)


def save_results_to_csv(L_list, L_check_list, prob_matrix, pec_matrix, total_matrix, save_path='results.csv', p_index=0):
    """Save log-scale values of probability/PEC/total samples to CSV."""
    prob = _select_p_slice(prob_matrix, "prob_matrix", p_index)
    pec = _select_p_slice(pec_matrix, "pec_matrix", p_index)
    total = _select_p_slice(total_matrix, "total_matrix", p_index)

    L_list = np.array(L_list)
    L_check_list = np.array(L_check_list)

    rows = []
    for i, L_check in enumerate(L_check_list):
        for j, L in enumerate(L_list):
            prob_val = prob[i, j]
            pec_val = pec[i, j]
            total_val = total[i, j]
            rows.append({
                'L_check': L_check,
                'L': L,
                'log10_inv_probability': np.log10(1.0 / prob_val) if prob_val > 0 else np.nan,
                'log10_pec_sample': np.log10(pec_val) if pec_val > 0 else np.nan,
                'log10_total_sample': np.log10(total_val) if total_val > 0 else np.nan,
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df
