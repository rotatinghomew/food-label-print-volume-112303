"""删失感知的需求分布估计。

历史班次有三种信息含量：
- 正常班次（stockout=0, 未取消）：真实需求被完整观察到，produced 即需求实现；
- 缺标班次（stockout=1）：因缺标没开满，只知道真实需求 >= produced（右删失）；
- 取消班次（plan_cancelled=1）：订单调整，produced=0 不携带需求信息，直接剔除。

把"用量低"和"缺标没开出来"混为一谈会系统性低估需求，因此所有估计都基于
Tobit 形式的删失似然而不是简单均值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize as scipy_optimize_minimize

# 数值下限，防止 log(0) 与除零。
_EPS = 1e-9


@dataclass
class CensoredSample:
    """一个班次对需求残差（需求 - 计划量）的观察。"""

    residual: float       # 正常班次 = produced - planned；删失班次 = produced - planned
    censored: bool        # True 表示只知道残差 >= residual


@dataclass
class NormalFit:
    """残差分布 N(bias, sigma^2) 的最大似然估计。"""

    bias: float
    sigma: float
    n_observed: int
    n_censored: int
    converged: bool

    @property
    def n_total(self) -> int:
        return self.n_observed + self.n_censored


def _neg_log_likelihood(
    params: tuple[float, float], samples: list[CensoredSample]
) -> float:
    bias, log_sigma = params
    sigma = math.exp(log_sigma)
    ll = 0.0
    for sample in samples:
        if sample.censored:
            # P(需求 >= produced) = 1 - Phi((c-bias)/sigma)
            z = (sample.residual - bias) / sigma
            ll += math.log(max(stats.norm.sf(z), _EPS))
        else:
            z = (sample.residual - bias) / sigma
            ll += stats.norm.logpdf(z) - math.log(sigma)
    return -ll


def fit_normal(samples: list[CensoredSample]) -> NormalFit:
    """对残差拟合正态分布，右删失观察走生存函数。

    用完整观察的矩估计做初值，再用 scipy 的 Nelder-Mead 在 (bias, log_sigma)
    二维上直接最大化删失似然；不依赖嵌套一维搜索，收敛稳定。
    优化失败时退回矩估计并把 converged 置 False，调用方据此保守使用。
    """
    observed = np.array([s.residual for s in samples if not s.censored], dtype=float)
    censored = [s for s in samples if s.censored]
    n_obs, n_cen = len(observed), len(censored)
    if n_obs < 2:
        raise ValueError("完整观察样本不足，无法估计需求分布")

    bias0 = float(observed.mean())
    sigma0 = max(float(observed.std(ddof=1)), 1.0)
    fallback = NormalFit(
        bias=bias0, sigma=sigma0, n_observed=n_obs, n_censored=n_cen,
        converged=False,
    )

    x0 = np.array([bias0, math.log(sigma0)])
    try:
        result = scipy_optimize_minimize(
            lambda x: _neg_log_likelihood((float(x[0]), float(x[1])), samples),
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000},
        )
    except Exception:
        return fallback
    if not result.success:
        return fallback
    bias_hat, log_sigma_hat = float(result.x[0]), float(result.x[1])
    sigma_hat = max(math.exp(log_sigma_hat), 0.5)
    return NormalFit(
        bias=bias_hat,
        sigma=sigma_hat,
        n_observed=n_obs,
        n_censored=n_cen,
        converged=True,
    )


def empirical_normal(samples: list[CensoredSample]) -> NormalFit:
    """基准矩估计：只用完整观察，忽略删失（会低估，作为对照基线）。"""
    observed = np.array([s.residual for s in samples if not s.censored], dtype=float)
    if len(observed) < 2:
        raise ValueError("完整观察样本不足")
    return NormalFit(
        bias=float(observed.mean()),
        sigma=max(float(observed.std(ddof=1)), 0.5),
        n_observed=len(observed),
        n_censored=sum(1 for s in samples if s.censored),
        converged=True,
    )


def quantile(fit: NormalFit, q: float) -> float:
    """残差分布的 q 分位数。"""
    return float(stats.norm.ppf(q, loc=fit.bias, scale=fit.sigma))


def survival_probability(fit: NormalFit, threshold: float) -> float:
    """P(残差 > threshold)，即真实需求超过 planned+threshold 的概率。"""
    return float(stats.norm.sf((threshold - fit.bias) / fit.sigma))
