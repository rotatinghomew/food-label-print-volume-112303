"""库存位置、在途印单与跨厂联合调拨。

关键不变量：
- 库存的 free = qty - held，held 是先前确认已经占用的量，不可再分配；
- 已开印/已发运/签收的印单是承诺，必须计入预计到货，不能删除或改期；
- 备选/推荐印单不是承诺，计划重算时整体替换；
- 共用版纸标在几个厂之间联合调拨，调拨需要 transfer_days 的在途时间；
- 同一实物批号在一次计划里只能沿时间线被分配一次（先到先得，逐日推进）；
- 跨厂调拨只在纸标对目标 (市场, SKU) 确实合格时发生：旧标按市场/SKU 各自
  的版本窗口管理，不能把南市场 b 线的旧标调去北市场 a 线。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .models import (
    COMMITTED_STATUSES,
    Dataset,
    PrintOrder,
)


@dataclass
class LotPosition:
    """计划时间线上某个实物批号的可分配余量。"""

    lot: str
    plant: str
    label: str
    available: dt.date
    free_qty: int
    exception_lot: bool = False  # 该批号是否为渠道例外批号

    def can_serve(self, day: dt.date, transfer_days: int,
                  target_plant: str) -> bool:
        lead = 0 if target_plant == self.plant else transfer_days
        return self.available + dt.timedelta(days=lead) <= day and self.free_qty > 0


@dataclass
class IncomingBatch:
    """预计到货：来自印单承诺或被确认的推荐印单。"""

    order: str
    plate: str
    label: str
    qty: int
    arrives: dt.date
    plant: str
    committed: bool
    starts: dt.date
    window_end: dt.date


@dataclass
class SupplyLedger:
    """计划期内全部供给（现存批号 + 在途/推荐到货）的时间线视图。"""

    lots: list[LotPosition]
    incoming: list[IncomingBatch]
    transfer_days: int
    # 已做分配：(批号或印单号, 需求id, 纸标, 数量, 来源厂, 目标厂)
    allocations: list[dict] = field(default_factory=list)

    def free_supply_on(
        self,
        label: str,
        target_plant: str,
        day: dt.date,
        *,
        exception_lot: str | None = None,
        target_market: str | None = None,
        target_sku: str | None = None,
        transfer_filter=None,
    ) -> list[tuple[str, int, str]]:
        """当天目标厂可用于某版本的供给，按 (来源标识, 可用量, 来源厂) 返回。

        exception_lot 给定时只允许该批号（渠道例外）。
        transfer_filter(来源厂, 目标厂, 批号或None, 印单号或None) 用于判断跨厂
        调拨在该需求日是否合法（日期窗口、切换通知、例外批号都由它裁决）。
        跨厂调拨按 transfer_days 提前就位，因此提前切换日会改变调拨优先级。
        """
        supply: list[tuple[str, int, str]] = []
        for pos in self.lots:
            if pos.label != label or pos.free_qty <= 0:
                continue
            if exception_lot is not None and pos.lot != exception_lot:
                continue
            if not pos.can_serve(day, self.transfer_days, target_plant):
                continue
            if pos.plant != target_plant and transfer_filter is not None and \
                    not transfer_filter(pos.plant, target_plant, pos.lot, None):
                continue
            supply.append((f"lot:{pos.lot}", pos.free_qty, pos.plant))
        for batch in self.incoming:
            if batch.label != label or batch.qty <= 0:
                continue
            lead = 0 if batch.plant == target_plant else self.transfer_days
            if batch.arrives + dt.timedelta(days=lead) > day:
                continue
            if batch.plant != target_plant and transfer_filter is not None and \
                    not transfer_filter(batch.plant, target_plant, None, batch.order):
                continue
            supply.append((f"order:{batch.order}", batch.qty, batch.plant))
        return supply

    def consume(
        self,
        source_key: str,
        qty: int,
        need_id: str,
        label: str,
        target_plant: str,
    ) -> dict:
        """从某来源扣减可用量并登记分配，返回分配明细。"""
        if source_key.startswith("lot:"):
            lot = source_key[4:]
            pos = next(p for p in self.lots if p.lot == lot)
            if qty > pos.free_qty:
                raise ValueError(f"批号 {lot} 余量不足")
            pos.free_qty -= qty
            from_plant = pos.plant
        else:
            order_id = source_key[6:]
            batch = next(b for b in self.incoming if b.order == order_id)
            if qty > batch.qty:
                raise ValueError(f"印单 {order_id} 到货余量不足")
            batch.qty -= qty
            from_plant = batch.plant
        record = {
            "need": need_id,
            "label": label,
            "source": source_key,
            "qty": qty,
            "from_plant": from_plant,
            "to_plant": target_plant,
            "transferred": from_plant != target_plant,
        }
        self.allocations.append(record)
        return record

    def remaining_by_label(self) -> dict[str, int]:
        """计划期末各版本仍未分配的供给量（报废风险的计算基础）。"""
        totals: dict[str, int] = {}
        for pos in self.lots:
            totals[pos.label] = totals.get(pos.label, 0) + pos.free_qty
        for batch in self.incoming:
            totals[batch.label] = totals.get(batch.label, 0) + batch.qty
        return totals


def build_ledger(
    dataset: Dataset,
    *,
    extra_orders: list[PrintOrder] | None = None,
) -> SupplyLedger:
    """从资料构建供给台账。

    extra_orders 传入试算中新增/替换的推荐印单：传入后旧的非承诺印单被整体
    替换，只保留已承诺印单 + 新推荐。跨厂调拨合法性由 risk 层的 transfer_filter
    按需求日的合格关系动态裁决，台账本身只记录批号/到货的物理位置与余量。
    """
    exception_lots = {
        rule.exception_lot
        for rule in dataset.eligibility
        if rule.exception_lot
    }

    lots = [
        LotPosition(
            lot=lot.lot,
            plant=lot.plant,
            label=lot.label,
            available=lot.available,
            free_qty=lot.free,
            exception_lot=lot.lot in exception_lots,
        )
        for lot in dataset.stock
    ]
    incoming: list[IncomingBatch] = []
    order_pool = list(dataset.orders)
    if extra_orders is not None:
        # 试算时用新推荐整体替换旧的非承诺印单。
        order_pool = [o for o in order_pool if o.committed] + list(extra_orders)
    for order in order_pool:
        incoming.append(
            IncomingBatch(
                order=order.order,
                plate=order.plate,
                label=order.label,
                qty=order.qty,
                arrives=order.arrives,
                plant=order.plant,
                committed=order.committed,
                starts=order.starts,
                window_end=order.window_end,
            )
        )
    return SupplyLedger(
        lots=lots,
        incoming=incoming,
        transfer_days=dataset.policy.transfer_days,
    )


def committed_orders(dataset: Dataset) -> list[PrintOrder]:
    return [o for o in dataset.orders if o.status in COMMITTED_STATUSES]


def recommended_orders(dataset: Dataset) -> list[PrintOrder]:
    return [o for o in dataset.orders if o.status not in COMMITTED_STATUSES]
