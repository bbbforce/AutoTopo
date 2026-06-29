"""结构化网格过滤器辅助函数。"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def build_density_filter(nelx: int, nely: int, rmin: float):
    """构建与 PythonSimpMMAEngine 一致的稀疏密度过滤矩阵。"""

    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(max(nfilter, 1))
    jH = np.zeros(max(nfilter, 1))
    sH = np.zeros(max(nfilter, 1))
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc += 1
    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.asarray(H.sum(1)).reshape(-1)
    Hs[Hs <= 0] = 1.0
    return H, Hs

