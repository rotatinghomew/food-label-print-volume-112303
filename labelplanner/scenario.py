"""追加排期、切换通知与印厂收单状态的反复试算。

包材组在确认前会不断改资料：追加生产需求、提前/推迟切换日、更新印单状态。
每次试算都从同一套已校验资料派生一个临时视图，重算未确认推荐；
已开印/已发运的承诺在任何视图里都保留，并继续在业务账上可查。
"""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass

from .models import (
    COMMITTED_STATUSES,
    Dataset,
    FutureNeed,
    PrintOrder,
    SwitchNotice,
)
from .planner import PlanResult, make_plan


@dataclass
class ScenarioChange:
    description: str
    kind: str
    payload: dict


class ScenarioBuilder:
    """以不可变方式在原资料上叠加试算改动。"""

    def __init__(self, dataset: Dataset):
        self._base = dataset

    def add_need(self, need: FutureNeed) -> "ScenarioBuilder":
        if any(n.id == need.id for n in self._base.needs):
            raise ValueError(f"需求 {need.id} 已存在")
        need.validate()
        self._base = self._clone(needs=[*self._base.needs, need])
        return self

    def add_switch(self, switch: SwitchNotice) -> "ScenarioBuilder":
        if any(s.id == switch.id for s in self._base.switches):
            raise ValueError(f"切换 {switch.id} 已存在")
        switch.validate()
        self._base = self._clone(switches=[*self._base.switches, switch])
        return self

    def move_switch(
        self, market: str, sku: str, new_day: dt.date
    ) -> "ScenarioBuilder":
        """临时提前/推迟切换日：旧标可用性与跨厂调拨优先级随之改变。"""
        moved = []
        hit = False
        for switch in self._base.switches:
            if switch.market == market and switch.sku == sku:
                hit = True
                moved.append(
                    SwitchNotice(
                        id=switch.id, market=market, sku=sku, day=new_day,
                        label=switch.label, recorded_at=switch.recorded_at,
                    )
                )
            else:
                moved.append(switch)
        if not hit:
            raise ValueError(f"没有 {market}/{sku} 的切换通知可改期")
        self._base = self._clone(switches=moved)
        return self

    def update_order_status(
        self, order_id: str, status: str
    ) -> "ScenarioBuilder":
        """更新印厂收单状态。

        备选 -> 开印/已发运属于把推荐转成承诺：此后该印单不可再被重算替换。
        已承诺印单不允许回退到非承诺状态。
        """
        if status not in COMMITTED_STATUSES and status not in {"备选", "推荐"}:
            raise ValueError(f"未知印单状态 {status}")
        updated: list[PrintOrder] = []
        hit = False
        candidate: PrintOrder | None = None
        for order in self._base.orders:
            if order.order == order_id:
                hit = True
                if order.committed and status not in COMMITTED_STATUSES:
                    raise ValueError(
                        f"印单 {order_id} 已承诺，不能回退到 {status}"
                    )
                candidate = PrintOrder(
                    order=order.order, plate=order.plate, label=order.label,
                    qty=order.qty, starts=order.starts,
                    window_end=order.window_end, arrives=order.arrives,
                    plant=order.plant, status=status,
                )
                updated.append(candidate)
            else:
                updated.append(order)
        if not hit:
            raise ValueError(f"印单 {order_id} 不存在")
        # 转承诺时必须先通过印版冲突检查：同一印版不能同时承诺两个交期。
        if candidate is not None and candidate.committed:
            conflicts = [
                other.order
                for other in updated
                if other.order != order_id and other.committed
                and candidate.overlaps(other)
            ]
            if conflicts:
                raise ValueError(
                    f"印单 {order_id} 转承诺会与已承诺印单 {conflicts} "
                    "在同一印版上交期重叠，不能同时承诺"
                )
        self._base = self._clone(orders=updated)
        return self

    def _clone(self, **overrides) -> Dataset:
        params = {
            "history": self._base.history,
            "eligibility": self._base.eligibility,
            "stock": self._base.stock,
            "orders": self._base.orders,
            "needs": self._base.needs,
            "switches": self._base.switches,
            "policy": self._base.policy,
            "source_dir": self._base.source_dir,
            "extras": copy.deepcopy(self._base.extras),
        }
        params.update(overrides)
        dataset = Dataset(**params)
        dataset.validate()
        return dataset

    @property
    def dataset(self) -> Dataset:
        return self._base

    def run(self, **plan_kwargs) -> PlanResult:
        return make_plan(self._base, **plan_kwargs)


def explain_window(
    base_result: PlanResult, scenario_result: PlanResult
) -> dict:
    """对比两个试算结果，回答"为什么要多印"。"""
    base_p95 = {o.need.id: o.q95_demand for o in base_result.planned.per_need} \
        if base_result.planned else {}
    changes = []
    target = scenario_result.planned or scenario_result.baseline
    for outcome in target.per_need:
        before = base_p95.get(outcome.need.id)
        changes.append({
            "need": outcome.need.id,
            "q95_demand_before": round(before, 1) if before is not None else None,
            "q95_demand_now": round(outcome.q95_demand, 1),
            "shortage_p_now": round(outcome.shortage_probability, 4),
            "reasons": scenario_result.reason_breakdown[outcome.need.id][
                "reason_codes"
            ],
        })
    return {
        "baseline_any_shortage_p": round(
            base_result.baseline.any_shortage_probability, 4
        ),
        "scenario_any_shortage_p": round(
            (scenario_result.planned or scenario_result.baseline)
            .any_shortage_probability, 4
        ),
        "scenario_feasible": scenario_result.feasible,
        "failures": scenario_result.failures,
        "per_need": changes,
        "recommended_orders": [
            {
                "label": r.label, "plate": r.plate, "qty": r.qty,
                "arrives": r.arrives.isoformat(),
                "reasons": r.reason_codes, "replaces": r.replaces,
            }
            for r in scenario_result.recommended_orders
        ],
    }
