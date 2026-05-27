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
