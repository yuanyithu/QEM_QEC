# A13 期望值估计

本项目探索了量子系统中期望值估计的方法，重点关注误差缓解技术。

## 项目结构

项目结构如下：

-   `.gitignore`: 指定 Git 应该忽略的未跟踪文件。
-   `报告.md`: (待总结)
-   `动力学解释.md`: 包含误差缓解下开放量子系统动力学的报告，包括采样复杂度分析。
-   `log.md`: 研究进展、想法和问题的日志。
-   `requirements.txt`: 列出了运行项目所需的 Python 包。
-   `test_code.ipynb`: 用于测试代码的 Jupyter Notebook。
-   `PEC/`: 包含与概率误差消除 (PEC) 技术相关的 Python 脚本。
    -   `layered_nopostselection_subspace_PEC.py`
    -   `layered_subspace_PEC.py`
    -   `layered_total_subspace_PEC.py`
    -   `monitor.py`
    -   `PEC_base.py`
    -   `pure_PEC_.py`
    -   `subspace_PEC.py`
    -   `visualization.py`
    -   `results/`: 用于存储 PEC 模拟结果的目录。
-   `QFI/`: 包含与量子费舍尔信息 (QFI) 相关的 Python 脚本。
    -   `QFI_main.py`
    -   `QFI_optimize.py`
    -   `QFI_single_round.py`
-   `reference paper/`: 包含参考论文。
    -   `IBM progressive_optimization_results.pdf`
    -   `IBM.pdf`

## 关键概念

-   **概率误差消除 (PEC)**: 一种量子误差缓解技术，用于减少量子计算中噪声的影响。
-   **量子费舍尔信息 (QFI)**: 一种衡量量子态携带的关于未知参数的信息量的指标。

## 研究方向

本项目探索以下研究方向：

-   理解不同插入测量策略对采样成本的影响。
-   研究在所有测量之后执行总 PEC 的好处。
-   探索利用辛德罗姆测量结果来改进误差缓解的方法。
-   考虑非 Clifford 门对量子电路中泡利噪声的影响。
-   噪声参数的基准测试。
-   将在线噪声学习与误差缓解技术相结合。

## 动力学模型 (来自 动力学解释.md)

系统演化由去极化噪声（速率 $\gamma$）和针对泄漏空间的连续投影测量（速率 $\kappa$）共同决定。

系统的非迹守恒 Lindblad 主方程为：

$$
\frac{d\rho(t)}{dt} = \gamma \left( \frac{\text{Tr}(\rho)}{n}\mathbb{I} - \rho \right) - \frac{\kappa}{2} \{ Q, \rho \}
$$

## 下一步

-   总结 `报告.md` 的内容。
-   进一步探索 `log.md` 中概述的研究方向。
-   实施和测试 `QFI/` 中与 QFI 相关的代码。
-   分析 `PEC/results/` 中 PEC 模拟的结果。