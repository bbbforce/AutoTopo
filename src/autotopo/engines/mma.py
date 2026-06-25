"""最小可替换 MMA 更新器。

本模块实现的是面向第一轮研究实验的确定性 MMA-style 更新：
使用移动渐近线限制每步变量范围，再对体积分数约束做盒约束投影。
它不是 OC 更新，也不会把 OC 作为 fallback。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MMAState:
    """MMA 更新状态。"""

    iteration: int
    move: float
    lower_asymptote: np.ndarray
    upper_asymptote: np.ndarray
    previous_x: np.ndarray


def initialize_mma_state(x: np.ndarray, *, move: float = 0.2) -> MMAState:
    """根据初始设计变量创建 MMA 状态。"""

    x = np.asarray(x, dtype=float)
    return MMAState(
        iteration=0,
        move=move,
        lower_asymptote=np.maximum(0.0, x - move),
        upper_asymptote=np.minimum(1.0, x + move),
        previous_x=x.copy(),
    )


def _project_to_volume(
    y: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    dv: np.ndarray,
    target_volume: float,
    active_mask: np.ndarray,
) -> np.ndarray:
    """将变量投影到线性体积约束和盒约束内。"""

    xnew = np.clip(y, lower, upper)
    if not np.any(active_mask):
        return xnew

    weights = np.maximum(np.asarray(dv, dtype=float), 1e-12)
    target_sum = float(target_volume * np.sum(weights[active_mask]))

    def weighted_sum(lam: float) -> float:
        candidate = np.clip(y - lam * weights, lower, upper)
        return float(np.sum(weights[active_mask] * candidate[active_mask]))

    lo, hi = -1.0, 1.0
    for _ in range(80):
        if weighted_sum(lo) >= target_sum:
            break
        lo *= 2.0
    for _ in range(80):
        if weighted_sum(hi) <= target_sum:
            break
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if weighted_sum(mid) > target_sum:
            lo = mid
        else:
            hi = mid

    return np.clip(y - 0.5 * (lo + hi) * weights, lower, upper)


def mma_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    *,
    volfrac: float,
    state: MMAState | None = None,
    active_mask: np.ndarray | None = None,
    passive_void_mask: np.ndarray | None = None,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> tuple[np.ndarray, MMAState]:
    """执行一次 MMA-style 有界更新。"""

    x = np.asarray(x, dtype=float)
    dc = np.asarray(dc, dtype=float)
    dv = np.asarray(dv, dtype=float)
    if state is None:
        state = initialize_mma_state(x)

    if active_mask is None:
        active_mask = np.ones_like(x, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)

    if passive_void_mask is None:
        passive_void_mask = np.zeros_like(x, dtype=bool)
    else:
        passive_void_mask = np.asarray(passive_void_mask, dtype=bool)
        active_mask = active_mask & ~passive_void_mask

    # 根据上一轮方向轻微调节移动限制，形成可替换的 MMA 状态接口。
    direction = x - state.previous_x
    stable_direction = np.sign(direction) == np.sign(state.previous_x - np.clip(state.previous_x, xmin, xmax))
    move = float(np.clip(state.move * (1.05 if np.any(stable_direction) else 0.95), 0.02, 0.25))
    lower = np.maximum(xmin, x - move)
    upper = np.minimum(xmax, x + move)
    lower[passive_void_mask] = 0.0
    upper[passive_void_mask] = 0.0

    scale = np.percentile(np.abs(dc[active_mask]), 75) if np.any(active_mask) else 1.0
    scale = max(float(scale), 1e-9)
    gradient_step = -0.15 * dc / scale / np.maximum(dv, 1e-12)
    y = x + np.clip(gradient_step, -move, move)

    xnew = _project_to_volume(
        y,
        lower=lower,
        upper=upper,
        dv=dv,
        target_volume=volfrac,
        active_mask=active_mask,
    )
    xnew[passive_void_mask] = 0.0
    xnew = np.clip(xnew, xmin, xmax)

    next_state = MMAState(
        iteration=state.iteration + 1,
        move=move,
        lower_asymptote=lower,
        upper_asymptote=upper,
        previous_x=x.copy(),
    )
    return xnew, next_state
