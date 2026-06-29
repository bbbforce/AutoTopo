"""可视化工具：密度场绘图与结果导出。

支持结构化矩形网格 (imshow) 和非结构三角网格 (tripcolor)。
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotopo-matplotlib")

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适用于 Docker 无 GUI 环境
import matplotlib.pyplot as plt
import numpy as np


def plot_density_field(
    densities: np.ndarray,
    save_path: str | None = None,
    *,
    title: str = "Topology Optimization Result",
    dpi: int = 300,
    cmap: str = "gray_r",
    binary_threshold: float | None = None,
    show_axes: bool = True,
    show: bool = False,
) -> str | None:
    """绘制 2D 密度场分布图。

    Parameters
    ----------
    densities : 2D 密度数组 (nely x nelx)
    save_path : 保存路径，None 则不保存
    """
    field = np.asarray(densities, dtype=float)
    if binary_threshold is not None:
        field = (field >= binary_threshold).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(field, cmap=cmap, origin="upper", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_axis_off()
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


def plot_fenics_density(
    coordinates: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    save_path: str | None = None,
    *,
    title: str = "Topology Optimization Result",
    dpi: int = 300,
    cmap: str = "gray_r",
    show: bool = False,
) -> str | None:
    """绘制非结构三角网格上的密度场。

    Parameters
    ----------
    coordinates : 节点坐标 (N, 2)
    cells : 三角形单元连接 (M, 3)
    values : 密度值 (M,) 或 (N,)
    save_path : 保存路径
    """
    from matplotlib.tri import Triangulation

    triang = Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if len(values) == len(cells):
        # 单元密度 → tripcolor
        tc = ax.tripcolor(triang, values, cmap=cmap, vmin=0, vmax=1, shading="flat")
    else:
        # 节点密度 → tripcolor (Gouraud)
        tc = ax.tripcolor(triang, values, cmap=cmap, vmin=0, vmax=1, shading="gouraud")

    fig.colorbar(tc, ax=ax, shrink=0.8)
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
    title: str = "Convergence History",
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
    if len(compliance_history) > 1:
        ax1.set_xlim(1, len(compliance_history))
    else:
        ax1.set_xlim(0.5, 1.5)

    # 右轴：体积分数
    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("Volume Fraction", color=color2, fontsize=12)
    ax2.plot(iterations, volume_history, color=color2, linewidth=1.5, linestyle="--", label="Volume")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 标题和图例
    fig.suptitle(title, fontsize=14, fontweight="bold")
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
