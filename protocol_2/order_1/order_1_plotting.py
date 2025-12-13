"""
绘图工具：提供期刊级（简化版）绘图样式和两类绘图接口：
1) plot_vs_T: 固定 n、p_single、p_double，随 T 绘制 sample。
2) plot_vs_n: 固定 T、p_single、p_double，随 n 绘制 sample。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterMathtext


def sci_mathtext(v: float, sig: int = 2) -> str:
    if v == 0 or not np.isfinite(v):
        return r"$0$"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10 ** exp)
    mant_str = f"{mant:.{sig}g}"
    if exp == 0:
        return rf"${mant_str}$"
    return rf"${mant_str}\times 10^{{{exp}}}$"


def set_publication_style():
    """设置更接近期刊的绘图风格（不依赖 LaTeX）。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.titlesize"] = 12


def plot_vs_T(T_list, protocol_sample, pure_sample, n, p_single, p_double, logy=True, show=False, save_path=None):
    """
    绘制随 T 变化的 sample 曲线。
    protocol_sample, pure_sample: 与 T_list 等长的序列。
    """
    T_int = np.array(T_list, dtype=int)
    x = np.arange(len(T_int))
    bar_width = 0.36
    fig, ax = plt.subplots()

    bars_qedcpec = ax.bar(
        x - bar_width / 2,
        protocol_sample,
        width=bar_width,
        facecolor="#66C2A5",
        edgecolor="#2F6F62",
        linewidth=0.8,
        alpha=0.85,
        label="QEDC+PEC",
    )
    bars_purepec = ax.bar(
        x + bar_width / 2,
        pure_sample,
        width=bar_width,
        facecolor="#BFE6D8",
        edgecolor="#5BAA96",
        linewidth=0.8,
        alpha=0.65,
        label="pure PEC",
    )
    trend_line, = ax.plot(
        x - bar_width / 2,
        protocol_sample,
        color="#2F6F62",
        linewidth=1.6,
        linestyle="-",
        marker="o",
        markersize=5,
        markerfacecolor="#2F6F62",
        markeredgewidth=0,
        label="QEDC+PEC trend",
        zorder=3,
    )

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xlabel("Logical cycle T", color="#222222")
    ax.set_ylabel("Sample overhead", color="#222222")
    ax.set_title("Sample overhead vs T", color="#222222")
    meta_text = f"GHZ prep, n={n}, p1={sci_mathtext(p_single)}, p2={sci_mathtext(p_double)}"
    ax.text(0.98, 0.98, meta_text, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(T_int)

    ax.grid(True, which="major", axis="y", color="#D9D9D9", linewidth=0.9, alpha=0.9)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(colors="#222222")

    ax.legend(
        handles=[bars_qedcpec, bars_purepec, trend_line],
        frameon=False,
        loc="upper left",
        handlelength=2.2,
        borderaxespad=0.6,
    )

    for bars in (bars_qedcpec, bars_purepec):
        labels = [sci_mathtext(b.get_height(), sig=2) for b in bars]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9, color="#222222")

    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_vs_n(n_list, protocol_sample, pure_sample, T, p_single, p_double, logy=True, show=False, save_path=None):
    """
    绘制随 n 变化的 sample 曲线（固定 T）。
    protocol_sample, pure_sample: 与 n_list 等长的序列。
    """
    fig, ax = plt.subplots()
    main_color = "#1B9E77"
    base_color = "#7A7A7A"
    mark_every = 2 if len(n_list) > 6 else 1

    ax.plot(
        n_list - 2,
        protocol_sample,
        color=main_color,
        linewidth=2.2,
        marker="o",
        markersize=6,
        markerfacecolor=main_color,
        markeredgewidth=0,
        alpha=0.95,
        zorder=3,
        markevery=mark_every,
        label="QEDC+PEC",
    )
    ax.plot(
        n_list - 2,
        pure_sample,
        color=base_color,
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=5.5,
        markerfacecolor="none",
        markeredgecolor=base_color,
        alpha=0.75,
        zorder=2,
        markevery=mark_every,
        label="pure PEC",
    )

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.set_xlabel("logical qubits n")
    ax.set_ylabel("Sample overhead")
    ax.set_title("Sample overhead vs n")
    meta_text = f"GHZ prep, T={T}, p1={sci_mathtext(p_single)}, p2={sci_mathtext(p_double)}"
    ax.text(0.99, 0.98, meta_text, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#222222")

    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, colors="#222222")

    def format_sci_tex(val: float, sig: int = 2) -> str:
        if val == 0 or not np.isfinite(val):
            return r"$0$"
        sign = "-" if val < 0 else ""
        v = abs(val)
        exp = int(np.floor(np.log10(v)))
        mant = v / (10 ** exp)
        mant_str = f"{mant:.{sig}g}"
        if exp == 0:
            return rf"${sign}{mant_str}$"
        return rf"${sign}{mant_str}\times 10^{{{exp}}}$"

    mid = len(n_list) // 2
    x_mid = (n_list - 2)[mid]
    y_mid_qedc = protocol_sample[mid]
    y_mid_pure = pure_sample[mid]
    ax.annotate(
        f"QEDC+PEC: {format_sci_tex(y_mid_qedc)}",
        xy=(x_mid, y_mid_qedc),
        xytext=(6, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
    )
    ax.annotate(
        f"pure PEC: {format_sci_tex(y_mid_pure)}",
        xy=(x_mid, y_mid_pure),
        xytext=(6, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
    )

    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
