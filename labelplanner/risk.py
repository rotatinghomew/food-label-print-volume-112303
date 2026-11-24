"""联合缺标概率与高分位余量仿真。

三厂处于同一促销/产线安排，需求残差同日相关，因此不能把各厂安全量相加来
推断整批安全。这里对需求向量做联合蒙特卡洛（保留合计关系），在每个仿真
路径上按时间线执行同一套确定性分配规则，统计：

- 任一生产单发生缺标的概率（政策口径：窗口内任一需求日缺标）；
- 各需求单的缺标概率与期望缺口；
- 切换后失去合格去处的纸标量（报废风险）分布；
- 合计需求的高分位余量（由联合分布解析给出，与仿真口径一致）。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .eligibility import EligibilityIndex
from .forecast import SKUS, ForecastBundle
from .inventory import SupplyLedger, build_ledger
from .models import Dataset, FutureNeed, PrintOrder

_EPS = 1e-9


def _home_market_sku(dataset: Dataset, plant: str, label: str) \
        -> tuple[str, str] | None:
    """某厂某纸标版本在本厂对应的 (市场, SKU)。

    一个厂服务固定的市场集合（由历史产线归属确定）；取该版本合格关系里
    与该厂市场匹配的第一条。旧标因此只会映射到它自己市场的 SKU，
    共用版则可能在多个市场命中——此时返回来源厂历史上实际生产的组合。
    """
    plant_markets = {
        row.market for row in dataset.history if row.plant == plant
    }
    plant_skus = {
        row.sku for row in dataset.history if row.plant == plant
    }
    for rule in dataset.eligibility:
        if (rule.label == label and rule.market in plant_markets
                and rule.sku in plant_skus):
            return rule.market, rule.sku
    return None


def _rule_covers(
    dataset: Dataset, market: str, sku: str, label: str, day: dt.date
) -> bool:
    """是否存在覆盖 (市场, SKU, 版本) 与需求日的普通合格规则（非渠道例外）。"""
    return any(
        rule.market == market and rule.sku == sku and rule.label == label
        and not rule.exception_lot and rule.covers(day)
        for rule in dataset.eligibility
    )


def _is_shared_label(dataset: Dataset, label: str) -> bool:
    """纸标是否为显式获准的跨市场共用版（委托合格关系索引判定）。

    市场专属 old/new 即使两个市场同名也不能跨市场调拨；只有像 "shared"
    这样同窗口绑定到不同市场不同 SKU 的版本才是共用纸标。
    """
    from .eligibility import EligibilityIndex

    return EligibilityIndex(dataset).is_cross_market_shared(label)


@dataclass
class NeedOutcome:
    need: FutureNeed
    shortage_probability: float
    expected_shortage: float
    q95_shortage: float
    mean_demand: float
    q95_demand: float
    usable_supply_mean: float


@dataclass
class SimulationResult:
    n_paths: int
    per_need: list[NeedOutcome]
    any_shortage_probability: float
    # 每个需求单在各路径上的实际满足量/缺口，矩阵形状 (路径数, 需求数)。
    shortage_matrix: np.ndarray
    demand_matrix: np.ndarray
    # 切换后不可用纸标量的路径分布与高分位。
    unusable_distribution: np.ndarray
    unusable_q95: float
    total_demand_mean: float
    total_demand_q95: float
    total_buffer_q95: float
    allocations: list[dict] = field(default_factory=list)
    infeasible_reasons: list[str] = field(default_factory=list)

    def passes_policy(self, dataset: Dataset) -> tuple[bool, list[str]]:
        """对照合同风险阈值与报废容限，给出是否可签及不可兼得项。"""
        failures: list[str] = []
        if self.any_shortage_probability > dataset.policy.max_stockout_probability + _EPS:
            failures.append(
                f"联合缺标概率 {self.any_shortage_probability:.1%} 超过阈值 "
                f"{dataset.policy.max_stockout_probability:.0%}"
            )
        if self.unusable_q95 > dataset.policy.max_unusable + _EPS:
            failures.append(
                f"切换后不可用纸标 P95={self.unusable_q95:.0f} 张超过容限 "
                f"{dataset.policy.max_unusable} 张"
            )
        return not failures, failures


def _allocate_path(
    dataset: Dataset,
    needs: list[FutureNeed],
    demands: np.ndarray,
    ledger: SupplyLedger,
    eligibility: EligibilityIndex,
    as_of: dt.date,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict], int]:
    """单条路径上的逐日分配，返回 (满足量, 缺口, 分配明细, 期末余量, 不可用量)。"""
    satisfied = np.zeros(len(needs))
    shortages = np.zeros(len(needs))
    path_allocations: list[dict] = []
    unusable = 0

    # 记录每个 (市场,sku) 的最后需求日，用于判断期末旧标是否还有合格去处。
    last_need_day: dict[tuple[str, str], dt.date] = {}
    for need in needs:
        key = (need.market, need.sku)
        last_need_day[key] = max(last_need_day.get(key, need.day), need.day)

    for i, need in enumerate(sorted(needs, key=lambda n: (n.day, n.id))):
        demand = int(round(demands[i]))
        options = eligibility.options(
            need.market, need.sku, need.day, as_of=as_of
        )
        remaining = demand
        # 先用普通合格版（本版优先、再共用调拨），渠道例外批号留到最后，
        # 因为例外标只能去指定市场，过早消耗会抬高报废。
        ordered = sorted(options, key=lambda o: o.is_exception)

        # 来源实物批号/到货按其本厂所属 (市场, SKU) 建档；跨厂调拨要求来源版
        # 对"目标 (市场, SKU)"在需求日确实合格，而不是只看版本名字相同。
        # 注意台账是每条仿真路径重建的，extra 推荐印单也在其中，必须用
        # 批次自带的 plant/label 现场映射，不能引用 dataset 里的静态订单表。
        lot_home = {
            p.lot: _home_market_sku(dataset, p.plant, p.label)
            for p in ledger.lots
        }
        batch_home = {
            b.order: _home_market_sku(dataset, b.plant, b.label)
            for b in ledger.incoming
        }

        def transfer_legal(from_plant, to_plant, lot_name, order_id,
                           label_version):
            # 跨厂调拨合法的充要条件：
            # 1) 该版本是跨市场共用版（市场专属 old/new 即使同名也不能跨市场）；
            # 2) 来源厂与目标厂的 (市场, SKU) 在需求日都有该版本的普通合格规则；
            # 渠道例外批号不参与跨厂调拨。
            if not _is_shared_label(dataset, label_version):
                return False
            if lot_name is not None:
                home = lot_home.get(lot_name)
            else:
                home = batch_home.get(order_id)
            if home is None:
                return False
            source_market, source_sku = home
            if (source_market, source_sku) == (need.market, need.sku):
                return True  # 同市场同 SKU 的厂间转运
            shared_source = _rule_covers(
                dataset, source_market, source_sku, label_version, need.day
            )
            shared_target = _rule_covers(
                dataset, need.market, need.sku, label_version, need.day
            )
            return shared_source and shared_target

        for option in ordered:
            if remaining <= 0:
                break
            supply = ledger.free_supply_on(
                option.label,
                need.plant,
                need.day,
                exception_lot=option.exception_lot,
                target_market=need.market,
                target_sku=need.sku,
                transfer_filter=lambda fp, tp, lot, oid, lv=option.label:
                    transfer_legal(fp, tp, lot, oid, lv),
            )
            # 本厂来源优先于跨厂调拨，减少在途占用。
            supply.sort(key=lambda s: (s[2] != need.plant, s[0]))
            for source_key, qty, _plant in supply:
                if remaining <= 0:
                    break
                take = min(qty, remaining)
                rec = ledger.consume(
                    source_key, take, need.id, option.label, need.plant
                )
                path_allocations.append(rec)
                remaining -= take
        satisfied[i] = demand - remaining
        shortages[i] = remaining

    # 期末报废（切换后不可用）口径：只统计"换代链旧标"在切换后失去合格去处的
    # 余量。共用版与新版在政策窗口后仍有合格规则（资料中其 to_date 延至
    # 2027-01-01），属于可继续周转的安全库存，不计报废；需求已满足而自然剩余
    # 的库存同样不计。这样报废与"切换"直接挂钩，而不是把所有剩余都算损失。
    chain_labels = {"old"}

    def _has_later_use(
        label: str,
        lot_name: str | None,
        ready_day,
        home: tuple[str, str] | None,
    ) -> bool:
        if home is None:
            return False
        market, sku = home
        day = last_need_day.get((market, sku))
        if day is None or day < ready_day:
            return False
        for option in eligibility.options(market, sku, day, as_of=as_of):
            if option.label != label:
                continue
            if lot_name is not None and option.exception_lot not in (None, lot_name):
                continue
            return True
        return False

    for pos in ledger.lots:
        if pos.label not in chain_labels or pos.free_qty <= 0:
            continue
        home = _home_market_sku(dataset, pos.plant, pos.label)
        if not _has_later_use(pos.label, pos.lot, pos.available, home):
            unusable += pos.free_qty
    for batch in ledger.incoming:
        if batch.label not in chain_labels or batch.qty <= 0:
            continue
        home = _home_market_sku(dataset, batch.plant, batch.label)
        if not _has_later_use(batch.label, None, batch.arrives, home):
            unusable += batch.qty
    return satisfied, shortages, path_allocations, ledger.remaining_by_label(), unusable


def simulate(
    dataset: Dataset,
    bundle: ForecastBundle,
    *,
    n_paths: int = 4000,
    rng: np.random.Generator | None = None,
    extra_orders: list[PrintOrder] | None = None,
    as_of: dt.date | None = None,
) -> SimulationResult:
    """对当前资料与一组（可能是试算的）印单做联合风险仿真。"""
    rng = rng or np.random.default_rng(20261124)
    # 计划"现在"取需求与切换通知中最晚的登记时刻：切换通知一旦登记就改变
    # 旧标可用性，不能因为它登记得晚就被需求登记日排除在外。
    known_moments = [n.recorded_at for n in dataset.needs]
    known_moments += [s.recorded_at for s in dataset.switches]
    as_of = as_of or max(known_moments)
    eligibility = EligibilityIndex(dataset)
    needs = sorted(dataset.needs, key=lambda n: (n.day, n.id))
    if not needs:
        raise ValueError("没有待保障需求")

    planned = {sku: 0 for sku in SKUS}
    # 每个需求单的计划量取生产单数量；联合向量按需求单顺序展开，
    # 同日同 SKU 的需求共享同一相关结构（用分组相关驱动）。
    sku_index = {sku: k for k, sku in enumerate(SKUS)}
    means = np.zeros(len(needs))
    sigmas = np.zeros(len(needs))
    sku_of = np.array([sku_index[n.sku] for n in needs])
    for i, need in enumerate(needs):
        fit = bundle.by_sku[need.sku].fit
        means[i] = need.quantity + fit.bias
        sigmas[i] = max(fit.sigma, _EPS)

    # 用 SKU 级相关矩阵构造需求单相关矩阵：同 SKU 完全相关，跨 SKU 用估计相关。
    corr = np.eye(len(needs))
    for i in range(len(needs)):
        for j in range(i + 1, len(needs)):
            if sku_of[i] == sku_of[j]:
                rho = 1.0
            else:
                rho = float(bundle.correlation[sku_of[i], sku_of[j]])
            corr[i, j] = corr[j, i] = rho
    cov = corr * np.outer(sigmas, sigmas)
    # 蒙特卡洛需求路径（半正定保护）。
    eig = np.linalg.eigvalsh(cov)
    if eig.min() <= 0:
        cov += np.eye(len(needs)) * (abs(eig.min()) + 1e-6)
    demand_paths = rng.multivariate_normal(means, cov, size=n_paths)
    demand_paths = np.maximum(demand_paths, 0.0)

    shortage_matrix = np.zeros((n_paths, len(needs)))
    demand_matrix = np.zeros((n_paths, len(needs)))
    unusable_dist = np.zeros(n_paths)
    allocation_counter: dict[tuple, int] = {}

    for p in range(n_paths):
        ledger = build_ledger(dataset, extra_orders=extra_orders)
        _, shortages, allocs, _leftover, unusable = _allocate_path(
            dataset, needs, demand_paths[p], ledger, eligibility, as_of
        )
        shortage_matrix[p] = shortages
        demand_matrix[p] = demand_paths[p]
        unusable_dist[p] = unusable
        for rec in allocs:
            key = (rec["need"], rec["label"], rec["source"],
                   rec["from_plant"], rec["to_plant"])
            allocation_counter[key] = allocation_counter.get(key, 0) + rec["qty"]

    per_need: list[NeedOutcome] = []
    for i, need in enumerate(needs):
        col = shortage_matrix[:, i]
        per_need.append(
            NeedOutcome(
                need=need,
                shortage_probability=float(np.mean(col > _EPS)),
                expected_shortage=float(np.mean(col)),
                q95_shortage=float(np.quantile(col, 0.95)),
                mean_demand=float(np.mean(demand_matrix[:, i])),
                q95_demand=float(np.quantile(demand_matrix[:, i], 0.95)),
                usable_supply_mean=float(
                    np.mean(demand_matrix[:, i]) - np.mean(col)
                ),
            )
        )

    any_short = float(np.mean(np.any(shortage_matrix > _EPS, axis=1)))
    total_demand = demand_matrix.sum(axis=1)

    # 代表性分配明细（取路径平均量，标注跨厂调拨）。
    allocations = [
        {
            "need": k[0], "label": k[1], "source": k[2],
            "from_plant": k[3], "to_plant": k[4],
            "avg_qty": round(qty / n_paths, 1),
            "transferred": k[3] != k[4],
        }
        for k, qty in sorted(allocation_counter.items())
    ]

    return SimulationResult(
        n_paths=n_paths,
        per_need=per_need,
        any_shortage_probability=any_short,
        shortage_matrix=shortage_matrix,
        demand_matrix=demand_matrix,
        unusable_distribution=unusable_dist,
        unusable_q95=float(np.quantile(unusable_dist, 0.95)),
        total_demand_mean=float(total_demand.mean()),
        total_demand_q95=float(np.quantile(total_demand, 0.95)),
        total_buffer_q95=float(
            np.quantile(total_demand, 0.95) - total_demand.mean()
        ),
        allocations=allocations,
    )


def high_quantile_buffer(
    bundle: ForecastBundle, planned: dict[str, int], q: float = 0.95
) -> dict[str, float]:
    """解析法高分位余量：合计与分 SKU 同时给出，合计由联合协方差推出。"""
    means, cov = bundle.joint_parameters(planned)
    total_sd = float(np.sqrt(cov.sum()))
    from scipy import stats

    z = float(stats.norm.ppf(q))
    return {
        sku: round(z * float(np.sqrt(cov[k, k])), 2)
        for k, sku in enumerate(SKUS)
    } | {"joint_total": round(z * total_sd, 2)}
