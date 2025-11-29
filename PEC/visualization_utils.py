import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import pandas as pd
from scipy.optimize import curve_fit


def linear_func(x, a, b):
    """Linear function for fitting: y = a*x + b"""
    return a * x + b


def setup_nature_style():
    """Set up matplotlib style similar to Nature journal aesthetics."""
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


def plot_inverse_probability(L_list, L_check_list, probs_matrix, save_path='inverse_probability.png'):
    """
    Plot 1/probability vs L with log scale y-axis.
    
    Args:
        L_list: array of L values (x-axis)
        L_check_list: array of L_check values (different curves)
        probs_matrix: shape (len(L_check_list), 1, len(L_list))
        save_path: output file path
    """
    setup_nature_style()
    fig, ax = plt.subplots()
    
    # Gradient colormap (Nature-style: viridis or plasma work well)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())
    
    for i, L_check in enumerate(L_check_list):
        inv_prob = 1.0 / probs_matrix[i, 0, :]
        color = cmap(norm(L_check))
        
        # Plot data points
        ax.scatter(L_list, inv_prob, color=color, s=15, alpha=0.7, zorder=3)
        
        # Fit in log space: log10(y) = a*x + b
        log_inv_prob = np.log10(inv_prob)
        valid_mask = np.isfinite(log_inv_prob)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_inv_prob[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10**log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('$1/P$')
    ax.set_title('Inverse Probability vs Circuit Depth')
    
    # Scientific notation for y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pec_sample(L_list, L_check_list, pec_samples_matrix, save_path='pec_sample.png'):
    """
    Plot PEC sample vs L with log scale y-axis.
    
    Args:
        L_list: array of L values (x-axis)
        L_check_list: array of L_check values (different curves)
        pec_samples_matrix: shape (len(L_check_list), 1, len(L_list))
        save_path: output file path
    """
    setup_nature_style()
    fig, ax = plt.subplots()
    
    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())
    
    for i, L_check in enumerate(L_check_list):
        pec = pec_samples_matrix[i, 0, :]
        color = cmap(norm(L_check))
        
        # Plot data points
        ax.scatter(L_list, pec, color=color, s=15, alpha=0.7, zorder=3)
        
        # Fit in log space: log10(y) = a*x + b
        log_pec = np.log10(pec)
        valid_mask = np.isfinite(log_pec)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_pec[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10**log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('PEC Sample')
    ax.set_title('PEC Sample vs Circuit Depth')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_total_sample(L_list, L_check_list, total_samples_matrix, save_path='total_sample.png'):
    """
    Plot total sample vs L with log scale y-axis.
    
    Args:
        L_list: array of L values (x-axis)
        L_check_list: array of L_check values (different curves)
        total_samples_matrix: shape (len(L_check_list), 1, len(L_list))
        save_path: output file path
    """
    setup_nature_style()
    fig, ax = plt.subplots()
    
    cmap = plt.cm.viridis
    norm = Normalize(vmin=L_check_list.min(), vmax=L_check_list.max())
    
    for i, L_check in enumerate(L_check_list):
        total = total_samples_matrix[i, 0, :]
        color = cmap(norm(L_check))
        
        # Plot data points
        ax.scatter(L_list, total, color=color, s=15, alpha=0.7, zorder=3)
        
        # Fit in log space: log10(y) = a*x + b
        log_total = np.log10(total)
        valid_mask = np.isfinite(log_total)
        if np.sum(valid_mask) > 2:
            popt, _ = curve_fit(linear_func, L_list[valid_mask], log_total[valid_mask])
            L_fit = np.linspace(L_list.min(), L_list.max(), 200)
            log_fit = linear_func(L_fit, *popt)
            ax.plot(L_fit, 10**log_fit, color=color, linewidth=1.5, linestyle='-', zorder=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('$L$')
    ax.set_ylabel('Total Sample')
    ax.set_title('Total Sample vs Circuit Depth')
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('$L_{check}$')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_all(L_list, L_check_list, probs_matrix, pec_samples_matrix, total_samples_matrix, prefix=''):
    """Convenience function to plot all three figures."""
    plot_inverse_probability(L_list, L_check_list, probs_matrix, 
                            save_path=f'{prefix}inverse_probability.png')
    plot_pec_sample(L_list, L_check_list, pec_samples_matrix, 
                   save_path=f'{prefix}pec_sample.png')
    plot_total_sample(L_list, L_check_list, total_samples_matrix, 
                     save_path=f'{prefix}total_sample.png')


def save_results_to_csv(L_list, L_check_list, probs_matrix, pec_samples_matrix, 
                        total_samples_matrix, save_path='results.csv'):
    """
    Save results to CSV file organized by L_check.
    
    Args:
        L_list: array of L values
        L_check_list: array of L_check values
        probs_matrix: shape (len(L_check_list), 1, len(L_list))
        pec_samples_matrix: shape (len(L_check_list), 1, len(L_list))
        total_samples_matrix: shape (len(L_check_list), 1, len(L_list))
        save_path: output CSV file path
    """
    rows = []
    
    for i, L_check in enumerate(L_check_list):
        for l, L in enumerate(L_list):
            prob = probs_matrix[i, 0, l]
            pec = pec_samples_matrix[i, 0, l]
            total = total_samples_matrix[i, 0, l]
            
            rows.append({
                'L_check': L_check,
                'L': L,
                'log10_inv_probability': np.log10(1.0 / prob) if prob > 0 else np.nan,
                'log10_pec_sample': np.log10(pec) if pec > 0 else np.nan,
                'log10_total_sample': np.log10(total) if total > 0 else np.nan
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df
