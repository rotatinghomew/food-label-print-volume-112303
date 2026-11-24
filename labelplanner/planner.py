"""印量计划：逐日承诺、印厂窗口与印版共用约束。

计划原则（对应业务口径）：
1. 逐个需求日判断各市场能用哪版纸标、库存是否已被之前的确认占用；
2. 共用版在几个厂之间做联合调拨，而不是各厂各订各的安全量；
3. 已开印/已发运承诺必须计入预计到货与报废风险；未确认推荐整体重算；
4. 同一共用印版不能同时承诺给两个交期（区间重叠即冲突）；
5. 采购只在合同给出的缺标风险与报废容限内签字，超限时列出不可兼得项；
6. 所有候选都超风险或超印厂窗口时，输出缺口与可协商的需求日/印版。

搜索方式：按需求日先后，对当天每个缺保障的纸标版本做印量网格搜索，候选印单
可落在任一印版上（不局限于该版历史印过的标签），逐个网格点跑联合仿真，取
满足政策阈值的最小印量；印版冲突的候选直接剔除并记录为可协商项。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .eligibility import EligibilityIndex
from .forecast import ForecastBundle, train
from .inventory import committed_orders
from .models import Dataset, PrintOrder
from .risk import SimulationResult, simulate


@dataclass
class RecommendedOrder:
    label: str
    plate: str
    qty: int
    starts: dt.date
    window_end: dt.date
    arrives: dt.date
    plant: str
    reason_codes: list[str]
    replaces: list[str] = field(default_factory=list)

    def as_print_order(self, order_id: str) -> PrintOrder:
        return PrintOrder(
            order=order_id,
            plate=self.plate,
            label=self.label,
            qty=self.qty,
            starts=self.starts,
            window_end=self.window_end,
            arrives=self.arrives,
            plant=self.plant,
            status="推荐",
        )


@dataclass
class PlanResult:
    bundle: ForecastBundle
    baseline: SimulationResult           # 只含已承诺印单的底线风险
    recommended_orders: list[RecommendedOrder]
    planned: SimulationResult | None    # 采纳推荐后的风险
    feasible: bool
    failures: list[str]
    plate_conflicts: list[tuple[str, str, str]]
    reason_breakdown: dict[str, dict]
    negotiable: list[dict]
    grid_evidence: list[dict]


def _plate_windows(
    dataset: Dataset,
) -> dict[str, tuple[dt.date, dt.date, dt.timedelta]]:
    """归纳每个印版的标准排产窗口：(最早开工, 最晚 window_end, 到货提前期)。"""
    windows: dict[str, tuple[dt.date, dt.date, dt.timedelta]] = {}
    for order in dataset.orders:
        lead = order.arrives - order.window_end
        if order.plate not in windows:
            windows[order.plate] = (order.starts, order.window_end, lead)
        else:
            starts, end, old_lead = windows[order.plate]
            windows[order.plate] = (
                min(starts, order.starts),
                max(end, order.window_end),
                lead,
            )
    return windows


def _candidate_orders(
    dataset: Dataset,
    label: str,
    quantities: list[int],
    arrive_by: dt.date,
    plant: str,
    *,
    extra_committed: list[PrintOrder] | None = None,
    rec_need_days: dict[str, dt.date] | None = None,
) -> tuple[list[PrintOrder], list[dict]]:
    """在所有印版窗口内生成某版本的候选印单。

    印版共用限制：同一印版的占用区间不得与 *已承诺* 印单重叠。推荐印单之间
    可以在同一印版上前后排产（窗口不重叠即可），因此候选与既有推荐的冲突
    通过收窄本次 window_end 来化解，而不是整块印版弃用。
    rec_need_days 记录每个既有推荐印单服务的需求日，用于区分同版串行与同日共用。
    """
    candidates: list[PrintOrder] = []
    blocked: list[dict] = []
    committed = committed_orders(dataset)
    prior_recs = list(extra_committed or [])
    rec_need_days = rec_need_days or {}
    windows = _plate_windows(dataset)
    for plate, (starts, latest_end, lead) in sorted(windows.items()):
        # 印版是不可共享的物理资源：本次可用区间 = 印版标准窗口减去
        # 已承诺印单与既有推荐印单的全部占用区间，按时间先后顺移。
        window_end = min(latest_end, arrive_by - lead)
        next_start = starts
        hard_block: list[str] = []
        occupied = sorted(
            [o for o in committed if o.plate == plate]
            + [o for o in prior_recs if o.plate == plate],
            key=lambda o: (o.starts, o.window_end),
        )
        for occ in occupied:
            if occ.starts <= window_end and occ.window_end >= next_start:
                hard_block.append(occ.order)
                window_end = min(window_end, occ.starts - dt.timedelta(days=1))
                next_start = max(
                    next_start, occ.window_end + dt.timedelta(days=1)
                )
        if window_end < next_start:
            blocked.append({
                "label": label, "plate": plate,
                "reason": "印版无空出窗口（已承诺/已推荐占用）",
                "conflicts_with": hard_block,
                "arrive_by": arrive_by.isoformat(),
            })
            continue
        if window_end < next_start:
            blocked.append({
                "label": label, "plate": plate,
                "reason": "印版剩余窗口早于所需到货日",
                "arrive_by": arrive_by.isoformat(),
            })
            continue
        # 到货日 = window_end + 印版标准提前期，并预留跨厂调拨的 transfer_days，
        # 保证印单在需求日当天能实际到达目标厂（含在途）。
        transfer_days = dataset.policy.transfer_days
        arrives = window_end + lead
        if arrives + dt.timedelta(days=transfer_days) > arrive_by:
            blocked.append({
                "label": label, "plate": plate,
                "reason": "含调拨在途后到货晚于需求日",
                "arrive_by": arrive_by.isoformat(),
                "arrives": arrives.isoformat(),
            })
            continue
        for qty in quantities:
            candidates.append(
                PrintOrder(
                    order=f"rec:{plate}:{label}:{qty}",
                    plate=plate, label=label, qty=qty,
                    starts=next_start, window_end=window_end, arrives=arrives,
                    plant=plant, status="推荐",
                )
            )
    return candidates, blocked


def _labels_by_need_day(
    dataset: Dataset, as_of: dt.date
) -> list[tuple[dt.date, str, list[str]]]:
    """按需求日汇总当天需要保障的纸标版本（去重、保序）。"""
    eligibility = EligibilityIndex(dataset)
    order: list[tuple[dt.date, str, list[str]]] = []
    for need in sorted(dataset.needs, key=lambda n: (n.day, n.id)):
        labels = [
            option.label
            for option in eligibility.options(
                need.market, need.sku, need.day, as_of=as_of
            )
        ]
        if not order or order[-1][0] != need.day:
            order.append((need.day, need.plant, list(labels)))
        else:
            for label in labels:
                if label not in order[-1][2]:
                    order[-1][2].append(label)
    return order


def _reason_breakdown(dataset: Dataset, bundle: ForecastBundle) -> dict[str, dict]:
    """解释每个需求单为什么需要保障：总需求上升 / 需求联动 / 旧标无合格去处。"""
    eligibility = EligibilityIndex(dataset)
    breakdown: dict[str, dict] = {}
    for need in sorted(dataset.needs, key=lambda n: (n.day, n.id)):
        codes: list[str] = []
        fit = bundle.by_sku[need.sku].fit
        demand_mean = need.quantity + fit.bias
        if demand_mean > need.quantity:
            codes.append("总需求上升")
        shared_demand = [
            n for n in dataset.needs
            if n.day == need.day and n.id != need.id
            and any(
                o.label == "shared"
                for o in eligibility.options(n.market, n.sku, n.day)
            )
        ]
        if shared_demand:
            codes.append("需求联动使共用备量不足")
        options = eligibility.options(need.market, need.sku, need.day)
        has_old = any(o.label == "old" and not o.is_exception for o in options)
        if not has_old:
            codes.append("旧标已无合格去处")
        breakdown[need.id] = {
            "demand_mean": round(demand_mean, 1),
            "q95_demand": round(
                need.quantity + fit.bias + 1.64485 * fit.sigma, 1
            ),
            "usable_labels": [o.label for o in options],
            "exception_only": [o.exception_lot for o in options if o.is_exception],
            "reason_codes": codes,
        }
    return breakdown


def make_plan(
    dataset: Dataset,
    *,
    n_paths: int = 2500,
    grid_step: int = 25,
    max_extra_per_label: int = 600,
    seed: int = 20261124,
) -> PlanResult:
    """生成一轮印量计划，并给出底线/采纳后两组风险证据。"""
    bundle = train(dataset.history, method="tobit")
    # 计划时刻取需求与切换通知中最晚的登记时刻（切换通知登记即改变旧标可用性）。
    known_moments = [n.recorded_at for n in dataset.needs]
    known_moments += [s.recorded_at for s in dataset.switches]
    as_of = max(known_moments)
    needs = sorted(dataset.needs, key=lambda n: (n.day, n.id))
    policy = dataset.policy

    baseline = _simulate_committed_only(dataset, bundle, n_paths, seed)

    grid_evidence: list[dict] = []
    recommendations: list[RecommendedOrder] = []
    current_extra: list[PrintOrder] = []
    rec_need_days: dict[str, dt.date] = {}
    blocked_all: list[dict] = []
    reasons = _reason_breakdown(dataset, bundle)

    # 当天各版本在联合口径下的共同基线：只含承诺印单 + 此前日期已确认的推荐。
    # 同一天可能有多个版本竞争同一印版空窗，因此先把所有候选放在同一基线上
    # 比较边际贡献，再决定把稀缺印版窗口给哪个版本（联合调拨口径，不各订各的）。
    for day, plant, labels in _labels_by_need_day(dataset, as_of):
        label_needs = [
            n for n in needs
            if n.day == day and any(
                o.label in labels
                for o in EligibilityIndex(dataset).options(
                    n.market, n.sku, n.day, as_of=as_of
                )
            )
        ]
        eligibility = EligibilityIndex(dataset)
        before = simulate(
            dataset, bundle, n_paths=n_paths,
            rng=np.random.default_rng(seed),
            extra_orders=current_extra, as_of=as_of,
        )
        before_total_gap = sum(o.expected_shortage for o in before.per_need)

        # 收集当天每个版本的全部候选印单及其在共同基线上的风险结果。
        day_options: list[dict] = []
        for label in labels:
            serving_needs = [
                n for n in label_needs
                if any(
                    o.label == label
                    for o in eligibility.options(
                        n.market, n.sku, n.day, as_of=as_of
                    )
                )
            ]
            upper = max(
                max_extra_per_label,
                sum(n.quantity for n in serving_needs) + grid_step,
            )
            quantities = list(range(grid_step, upper + 1, grid_step))
            candidates, blocked = _candidate_orders(
                dataset, label, quantities, day, plant,
                extra_committed=current_extra,
                rec_need_days=rec_need_days,
            )
            blocked_all.extend(blocked)
            if not candidates:
                grid_evidence.append({
                    "label": label, "day": day.isoformat(),
                    "status": "无可用印版窗口", "blocked": blocked,
                })
                continue

            served_ids = {
                n.id for n in needs
                if any(
                    o.label == label
                    for o in eligibility.options(
                        n.market, n.sku, n.day, as_of=as_of
                    )
                )
            }
            before_served = sum(
                o.expected_shortage for o in before.per_need
                if o.need.id in served_ids
            )
            for cand in sorted(candidates, key=lambda c: (c.qty, c.plate)):
                result = simulate(
                    dataset, bundle, n_paths=n_paths,
                    rng=np.random.default_rng(seed),
                    extra_orders=current_extra + [cand], as_of=as_of,
                )
                ok, _failures = result.passes_policy(dataset)
                served_gap = sum(
                    o.expected_shortage for o in result.per_need
                    if o.need.id in served_ids
                )
                marginal_gap = before_served - served_gap
                total_gap_drop = before_total_gap - sum(
                    o.expected_shortage for o in result.per_need
                )
                grid_evidence.append({
                    "label": label, "day": day.isoformat(),
                    "plate": cand.plate, "qty": cand.qty,
                    "any_shortage_p": round(result.any_shortage_probability, 4),
                    "unusable_q95": round(result.unusable_q95, 1),
                    "passes": ok,
                    "marginal_gap_reduction": round(marginal_gap, 2),
                    "total_gap_reduction": round(total_gap_drop, 2),
                })
                day_options.append({
                    "label": label, "cand": cand, "result": result,
                    "ok": ok, "marginal_gap": marginal_gap,
                    "total_gap_drop": total_gap_drop,
                    "serving_needs": serving_needs,
                })

        if not day_options:
            continue

        # 决策：先看是否有候选能让整体直接过政策（取最小印量）；
        # 否则在印版空窗稀缺时，选总缺口下降最多的版本，避免把窗口给一个
        # 只救单一需求、却挤掉能同时救两个厂共用需求的版本。
        passing = [d for d in day_options if d["ok"]]
        if passing:
            chosen = min(passing, key=lambda d: d["cand"].qty)
        else:
            beneficial = [
                d for d in day_options
                if d["total_gap_drop"] > max(0.5, 0.01 * max(before_total_gap, 1.0))
            ]
            if not beneficial:
                for label in labels:
                    grid_evidence.append({
                        "label": label, "day": day.isoformat(),
                        "status": "全部候选超风险或超窗口",
                    })
                continue
            # 同印版只能选一个：按总缺口下降排序，同幅取小印量。
            # 报废容限是独立约束：切换导致的旧标报废不因加印而变化（加印的是
            # 新/共用版），因此不应因为报废已超限就否决能实质降缺标的推荐。
            chosen = max(
                beneficial,
                key=lambda d: (d["total_gap_drop"], -d["cand"].qty),
            )

        cand = chosen["cand"]
        label = chosen["label"]
        current_extra.append(cand)
        rec_need_days[cand.order] = day
        recommendations.append(
            RecommendedOrder(
                label=label,
                plate=cand.plate,
                qty=cand.qty,
                starts=cand.starts,
                window_end=cand.window_end,
                arrives=cand.arrives,
                plant=cand.plant,
                reason_codes=sorted({
                    code
                    for n in chosen["serving_needs"]
                    for code in reasons[n.id]["reason_codes"]
                }),
                replaces=[
                    o.order for o in dataset.orders
                    if not o.committed and o.label == label
                ],
            )
        )
        # 同一印版窗口已被占用，当天其余版本若再无空窗则记录为可协商阻塞。
        for other in day_options:
            if other["cand"].plate == cand.plate and other["label"] != label:
                blocked_all.append({
                    "label": other["label"], "plate": cand.plate,
                    "reason": "印版空窗已分配给边际贡献更高的版本",
                    "arrive_by": day.isoformat(),
                })

    planned_result = simulate(
        dataset, bundle, n_paths=n_paths,
        rng=np.random.default_rng(seed), extra_orders=current_extra, as_of=as_of,
    )
    feasible, failures = planned_result.passes_policy(dataset)

    conflicts = list(dataset.plate_conflicts())
    for rec in recommendations:
        probe = rec.as_print_order("probe")
        for order in dataset.orders:
            if order.committed and probe.overlaps(order):
                conflicts.append((rec.plate, "推荐", order.order))

    negotiable = _negotiation_options(
        dataset, bundle, planned_result, seed, n_paths, blocked_all
    )
    return PlanResult(
        bundle=bundle,
        baseline=baseline,
        recommended_orders=recommendations,
        planned=planned_result,
        feasible=feasible,
        failures=failures,
        plate_conflicts=sorted(set(conflicts)),
        reason_breakdown=reasons,
        negotiable=negotiable,
        grid_evidence=grid_evidence,
    )


def _simulate_committed_only(
    dataset: Dataset,
    bundle: ForecastBundle,
    n_paths: int,
    seed: int,
    extra: list[PrintOrder] | None = None,
) -> SimulationResult:
    """只保留已承诺印单的底线视图（备选/推荐全部剔除后重算）。"""
    committed = committed_orders(dataset)
    from .models import Dataset as _Dataset

    stripped = _Dataset(
        history=dataset.history,
        eligibility=dataset.eligibility,
        stock=dataset.stock,
        orders=committed,
        needs=dataset.needs,
        switches=dataset.switches,
        policy=dataset.policy,
        source_dir=dataset.source_dir,
    )
    return simulate(
        stripped, bundle, n_paths=n_paths,
        rng=np.random.default_rng(seed), extra_orders=extra,
    )


def _negotiation_options(
    dataset: Dataset,
    bundle: ForecastBundle,
    result: SimulationResult,
    seed: int,
    n_paths: int,
    blocked: list[dict],
) -> list[dict]:
    """仍有缺口或印版受限时，列出可以协商的需求日与印版。"""
    options: list[dict] = []
    all_plates = sorted(_plate_windows(dataset))
    for outcome in result.per_need:
        if outcome.shortage_probability <= dataset.policy.max_stockout_probability:
            continue
        need = outcome.need
        options.append({
            "need": need.id,
            "day": need.day.isoformat(),
            "market": need.market,
            "sku": need.sku,
            "shortage_p": round(outcome.shortage_probability, 4),
            "expected_gap": round(outcome.expected_shortage, 1),
            "negotiable_plates": all_plates,
            "plate_blocks": [b for b in blocked
                             if b.get("label") in
                             [o.label for o in EligibilityIndex(dataset).options(
                                 need.market, need.sku, need.day)]],
            "suggestion": "协商提前印版 window_end、释放印版占用或后移需求日",
        })
    return options
