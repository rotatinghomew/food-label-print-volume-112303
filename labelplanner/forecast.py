"""需求预测模型、滚动回测与结构变化判断。

两个可解释方案在同一条滚动时间线上比较：

- ``naive``（基准）：只看完整观察班次的残差矩估计，缺标班次当缺失，
  等价于"用量低就少备"的经验做法，会被删失带偏；
- ``tobit``（改进）：对右删失班次使用生存函数的最大似然估计，
  明确区分"需求本来就低"与"缺标导致没开出来"。

回测严格按 planned_at 滚动：每个原点只用当时已经完成的班次，
对成熟窗口（completed_at 已过）里的班次做事后评估，不使用未来信息。
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .distributions import (
    CensoredSample,
    NormalFit,
    empirical_normal,
    fit_normal,
    quantile,
)
from .models import HistoryRow

SKUS = ("a", "b", "c")


def _residual_samples(
    rows: Iterable[HistoryRow], sku: str
) -> list[CensoredSample]:
    samples: list[CensoredSample] = []
    for row in rows:
        if row.sku != sku or row.plan_cancelled:
            continue
        samples.append(
            CensoredSample(
                residual=float(row.produced - row.planned),
                censored=row.stockout,
            )
        )
    return samples


@dataclass
class SkuForecast:
    """单 SKU 的需求分布：需求 | 计划量 ~ N(planned+bias, sigma^2)。"""

    sku: str
    method: str
    fit: NormalFit
    training_rows: int

    def mean(self, planned: int) -> float:
        return planned + self.fit.bias

    def quantile_demand(self, planned: int, q: float) -> float:
        return planned + quantile(self.fit, q)


@dataclass
class ForecastBundle:
    """一个计划时点上三个 SKU 的边缘分布与同日联合协方差。"""

    as_of: dt.date
    method: str
    by_sku: dict[str, SkuForecast]
    # 同日残差相关矩阵（由完整观察日估计），顺序固定为 SKUS。
    correlation: np.ndarray
    structural: "StructuralChangeReport | None" = None

    def joint_parameters(
        self, planned: dict[str, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回联合需求的均值向量与协方差矩阵（保留三厂合计关系）。"""
        means = np.array(
            [planned[sku] + self.by_sku[sku].fit.bias for sku in SKUS], dtype=float
        )
        sigmas = np.array([self.by_sku[sku].fit.sigma for sku in SKUS])
        cov = self.correlation * np.outer(sigmas, sigmas)
        return means, cov

    def total_quantile(self, planned: dict[str, int], q: float) -> float:
        """三 SKU 合计需求的 q 分位数（由联合参数解析得到，不做独立近似）。"""
        means, cov = self.joint_parameters(planned)
        total_mean = float(means.sum())
        total_sd = math.sqrt(float(cov.sum()))
        from scipy import stats

        return float(stats.norm.ppf(q, loc=total_mean, scale=total_sd))


def estimate_correlation(rows: Iterable[HistoryRow]) -> np.ndarray:
    """用三厂都完整观察（无缺标、无取消）的工作日估计同日残差相关。

    促销窗口三厂同步，残差相关体现需求联动；缺标日的删失行不进入相关估计。
    """
    by_day: dict[dt.date, dict[str, float]] = {}
    for row in rows:
        if row.plan_cancelled or row.stockout:
            continue
        by_day.setdefault(row.day, {})[row.sku] = float(
            row.produced - row.planned
        )
    matrix = []
    for day, per_sku in sorted(by_day.items()):
        if all(sku in per_sku for sku in SKUS):
            matrix.append([per_sku[sku] for sku in SKUS])
    if len(matrix) < 5:
        return np.eye(len(SKUS))
    arr = np.array(matrix)
    corr = np.corrcoef(arr, rowvar=False)
    # 数值保护：保证正定且对角为 1。
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() <= 1e-6:
        corr = corr + np.eye(len(SKUS)) * (1e-6 - eigvals.min())
    return corr


# --------------------------------------------------------------------------- #
# 结构变化
# ---------------------------------------------------------------------------


@dataclass
class StructuralChangeReport:
    """计划/售量结构变化的证据分级。"""

    segments: list[dict] = field(default_factory=list)
    # sku -> 结论：comparable（仍有可比证据）/ interval（仅区间可用）/ conservative（保守）
    verdicts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def detect_structural_change(
    rows: list[HistoryRow], window_days: int = 28
) -> StructuralChangeReport:
    """按月度分段比较计划量水平，识别结构变化并给出证据分级。

    - comparable：近段与上一段水平差异在噪声范围内，边缘分布仍具可比证据；
    - interval：水平发生漂移，点估计需按近段重估，旧段只支持区间/方差证据；
    - conservative：近段样本太短或删失比例过高，只能保守使用上界。
    """
    report = StructuralChangeReport()
    months = sorted({row.day.strftime("%Y-%m") for row in rows})
    segment_planned: dict[str, dict[str, list[int]]] = {m: {} for m in months}
    for row in rows:
        segment_planned[row.day.strftime("%Y-%m")].setdefault(row.sku, []).append(
            row.planned
        )

    for sku in SKUS:
        levels = [
            (m, float(np.mean(segment_planned[m].get(sku, [np.nan]))))
            for m in months
        ]
        report.segments.append({"sku": sku, "levels": levels})
        recent = [r for r in rows if r.sku == sku and r.plan_cancelled == 0]
        cutoff = max(r.day for r in rows) - dt.timedelta(days=window_days)
        recent_rows = [r for r in recent if r.day > cutoff]
        if len(recent_rows) < 10:
            report.verdicts[sku] = "conservative"
            report.notes.append(
                f"sku={sku}: 近{window_days}天完整班次不足10条，只给保守上界"
            )
            continue
        censor_rate = sum(r.stockout for r in recent_rows) / len(recent_rows)
        if censor_rate > 0.35:
            report.verdicts[sku] = "conservative"
            report.notes.append(
                f"sku={sku}: 近段删失比例 {censor_rate:.0%} 过高，点估计不可靠，保守使用"
            )
            continue
        # 与前一段等长窗口比较计划水平，漂移超过一个合并标准差即判结构变化。
        older_rows = [r for r in recent if r.day <= cutoff]
        if len(older_rows) < 10:
            report.verdicts[sku] = "interval"
            continue
        old_mean = float(np.mean([r.planned for r in older_rows]))
        new_mean = float(np.mean([r.planned for r in recent_rows]))
        pooled_sd = math.sqrt(
            (float(np.var([r.planned for r in older_rows]))
             + float(np.var([r.planned for r in recent_rows]))) / 2
        )
        if abs(new_mean - old_mean) > max(pooled_sd, 1.0):
            report.verdicts[sku] = "interval"
            report.notes.append(
                f"sku={sku}: 计划水平 {old_mean:.1f}->{new_mean:.1f} 发生漂移，"
                "近段重估点预测，旧段仅支持区间证据"
            )
        else:
            report.verdicts[sku] = "comparable"
    return report


# --------------------------------------------------------------------------- #
# 模型训练与回测
# ---------------------------------------------------------------------------


def train(
    rows: list[HistoryRow],
    method: str = "tobit",
    as_of: dt.date | None = None,
) -> ForecastBundle:
    """以 as_of（默认取历史中最大 completed_at）为计划时刻训练一组预测。"""
    if as_of is None:
        as_of = max(row.completed_at for row in rows)
    known = [row for row in rows if row.completed_at <= as_of]
    by_sku: dict[str, SkuForecast] = {}
    for sku in SKUS:
        samples = _residual_samples(known, sku)
        if method == "naive":
            fit = empirical_normal(samples)
        elif method == "tobit":
            fit = fit_normal(samples)
        else:
            raise ValueError(f"未知预测方法 {method}")
        by_sku[sku] = SkuForecast(
            sku=sku, method=method, fit=fit, training_rows=len(samples)
        )
    correlation = estimate_correlation(known)
    return ForecastBundle(
        as_of=as_of,
        method=method,
        by_sku=by_sku,
        correlation=correlation,
        structural=detect_structural_change(known),
    )


@dataclass
class BacktestRow:
    origin: dt.date
    sku: str
    shift: str
    planned: int
    actual: int
    censored: bool
    pred_mean: float
    pred_sigma: float
    pred_q85: float
    pred_q95: float


@dataclass
class BacktestSummary:
    method: str
    origins: list[dt.date]
    rows: list[BacktestRow]
    # 完整观察班次上的点预测指标。
    mae: float
    rmse: float
    bias: float
    # 完整观察班次上的区间覆盖率（应接近名义 85%/95%）。
    coverage85: float
    coverage95: float
    # 删失班次上的下界检查：actual（需求下界）低于预测分位的比例。
    coverage85_censored: float
    coverage95_censored: float
    crps: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "n_origins": len(self.origins),
            "n_eval": len(self.rows),
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "bias": round(self.bias, 3),
            "coverage85": round(self.coverage85, 3),
            "coverage95": round(self.coverage95, 3),
            "coverage85_lowerbound_on_stockouts": round(self.coverage85_censored, 3),
            "coverage95_lowerbound_on_stockouts": round(self.coverage95_censored, 3),
            "crps": round(self.crps, 3),
        }


def _crps_normal(mu: float, sigma: float, value: float) -> float:
        """闭合形式 CRPS（正态预测 vs 单值实现）。"""
        z = (value - mu) / sigma
        return float(
            sigma
            * (
                z * (2 * _norm_cdf(z) - 1)
                + 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                - 1 / math.sqrt(math.pi)
            )
        )


def _norm_cdf(z: float) -> float:
    from scipy import stats

    return float(stats.norm.cdf(z))


def rolling_backtest(
    rows: list[HistoryRow],
    method: str,
    *,
    origin_step_days: int = 14,
    min_training_days: int = 28,
) -> BacktestSummary:
    """滚动原点回测。

    每个原点 t：仅用 completed_at<=t 的班次训练；评估 (t, t+origin_step] 内
    完成的成熟窗口班次。原点等距滚动，首个原点留足 min_training_days 天训练。
    """
    if not rows:
        raise ValueError("历史为空")
    first_day = min(row.day for row in rows)
    last_complete = max(row.completed_at for row in rows)
    first_origin = first_day + dt.timedelta(days=min_training_days)
    if first_origin >= last_complete:
        raise ValueError("历史长度不足以做一次完整回测")

    origins: list[dt.date] = []
    cursor = first_origin
    while cursor < last_complete:
        origins.append(cursor)
        cursor += dt.timedelta(days=origin_step_days)

    eval_rows: list[BacktestRow] = []
    for origin, next_origin in zip(origins, origins[1:] + [last_complete]):
        bundle = train(rows, method=method, as_of=origin)
        for row in rows:
            if not (origin < row.completed_at <= next_origin):
                continue
            if row.plan_cancelled:
                continue
            fc = bundle.by_sku[row.sku]
            eval_rows.append(
                BacktestRow(
                    origin=origin,
                    sku=row.sku,
                    shift=row.shift,
                    planned=row.planned,
                    actual=row.produced,
                    censored=row.stockout,
                    pred_mean=fc.mean(row.planned),
                    pred_sigma=fc.fit.sigma,
                    pred_q85=fc.quantile_demand(row.planned, 0.85),
                    pred_q95=fc.quantile_demand(row.planned, 0.95),
                )
            )

    full = [r for r in eval_rows if not r.censored]
    errors = np.array([r.pred_mean - r.actual for r in full])
    cen = [r for r in eval_rows if r.censored]
    crps_values = [
        _crps_normal(r.pred_mean, r.pred_sigma, r.actual)
        for r in full
    ]
    # 完整观察班次上的区间覆盖率应接近名义水平（85%/95%），
    # 删失班次只做下界检查：actual 低于分位的比例应不低于名义覆盖。
    coverage85 = sum(1 for r in full if r.actual <= r.pred_q85) / len(full)
    coverage95 = sum(1 for r in full if r.actual <= r.pred_q95) / len(full)
    return BacktestSummary(
        method=method,
        origins=origins,
        rows=eval_rows,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        bias=float(np.mean(errors)),
        coverage85=coverage85,
        coverage95=coverage95,
        coverage85_censored=(
            sum(1 for r in cen if r.actual <= r.pred_q85) / len(cen) if cen else float("nan")
        ),
        coverage95_censored=(
            sum(1 for r in cen if r.actual <= r.pred_q95) / len(cen) if cen else float("nan")
        ),
        crps=float(np.mean(crps_values)),
    )


def compare_methods(
    rows: list[HistoryRow], **kwargs
) -> dict[str, BacktestSummary]:
    """在同一批滚动原点上比较基准与改进方案。"""
    return {
        method: rolling_backtest(rows, method, **kwargs)
        for method in ("naive", "tobit")
    }
