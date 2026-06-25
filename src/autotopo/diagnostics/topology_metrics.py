"""密度场拓扑质量指标。"""

from __future__ import annotations

from collections import deque

import numpy as np


def active_values(density: np.ndarray, passive_mask: np.ndarray | None = None) -> np.ndarray:
    """返回设计域内的密度值。"""

    arr = np.asarray(density, dtype=float)
    if passive_mask is None:
        return arr.ravel()
    mask = ~np.asarray(passive_mask, dtype=bool)
    return arr[mask]


def grayness_index(density: np.ndarray, passive_mask: np.ndarray | None = None) -> float:
    """灰度指标，0 表示纯 0/1，1 表示全部 0.5。"""

    vals = np.clip(active_values(density, passive_mask), 0.0, 1.0)
    if vals.size == 0:
        return 1.0
    return float(np.mean(4.0 * vals * (1.0 - vals)))


def checkerboard_score(density: np.ndarray, passive_mask: np.ndarray | None = None) -> float:
    """检测局部交替密度的简化棋盘格指标。"""

    arr = np.asarray(density, dtype=float)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    score = np.abs((arr[:-1, :-1] + arr[1:, 1:]) - (arr[1:, :-1] + arr[:-1, 1:])) / 2.0
    if passive_mask is not None:
        active_block = ~(
            passive_mask[:-1, :-1]
            | passive_mask[1:, 1:]
            | passive_mask[1:, :-1]
            | passive_mask[:-1, 1:]
        )
        if not np.any(active_block):
            return 0.0
        score = score[active_block]
    return float(np.mean(score)) if score.size else 0.0


def connectivity_score(
    density: np.ndarray,
    passive_mask: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> float:
    """最大实体连通分量占所有实体单元的比例。"""

    arr = np.asarray(density, dtype=float)
    solid = arr >= threshold
    if passive_mask is not None:
        solid = solid & ~np.asarray(passive_mask, dtype=bool)
    total = int(np.sum(solid))
    if total == 0:
        return 0.0

    visited = np.zeros_like(solid, dtype=bool)
    best = 0
    rows, cols = solid.shape
    for r in range(rows):
        for c in range(cols):
            if not solid[r, c] or visited[r, c]:
                continue
            count = 0
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                count += 1
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and solid[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            best = max(best, count)
    return float(best / total)


def volume_error(
    density: np.ndarray,
    target_volume: float,
    passive_mask: np.ndarray | None = None,
) -> float:
    """设计域体积分数误差。"""

    vals = active_values(density, passive_mask)
    if vals.size == 0:
        return 1.0
    return float(abs(np.mean(vals) - target_volume))
