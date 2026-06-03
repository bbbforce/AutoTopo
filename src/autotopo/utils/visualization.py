"""可视化工具：密度场绘图与结果导出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_density_field(
    densities: np.ndarray,
    save_path: str | None = None,
    *,
    title: str = "Topology Optimization Result",
    dpi: int = 300,
    cmap: str = "gray_r",
    show: bool = False,
) -> str | None:
    """绘制 2D 密度场分布图。

    Parameters
    ----------
    densities : 2D 密度数组 (nely x nelx)
    save_path : 保存路径，None 则不保存
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(densities, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_convergence_history(
    compliance_history: list[float],
    volume_history: list[float],
    save_path: str | None = None,
    *,
    dpi: int = 300,
    show: bool = False,
) -> str | None:
    """绘制收敛历史图（双 Y 轴）。

    Parameters
    ----------
    compliance_history : 各迭代柔度值列表
    volume_history : 各迭代体积分数列表
    save_path : 保存路径，None 则不保存
    """
    iterations = range(1, len(compliance_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左轴：柔度
    color1 = "#2563eb"
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Compliance", color=color1, fontsize=12)
    ax1.plot(iterations, compliance_history, color=color1, linewidth=1.5, label="Compliance")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xlim(1, len(compliance_history))

    # 右轴：体积分数
    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("Volume Fraction", color=color2, fontsize=12)
    ax2.plot(iterations, volume_history, color=color2, linewidth=1.5, linestyle="--", label="Volume")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 标题和图例
    fig.suptitle("Convergence History", fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path
