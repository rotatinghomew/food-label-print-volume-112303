"""领域记录与资料容器。

所有数量以张计，日期为工作日历上的当地日期（``datetime.date``）。
记录来自合成 CSV，读取时做最小约束校验，明显矛盾直接抛 ``ValueError``，
避免把坏数据带进预测和承诺计算。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value.strip())


def parse_int(value: str) -> int:
    value = value.strip()
    if value == "":
        raise ValueError("空整数列")
    return int(value)


@dataclass(frozen=True)
class HistoryRow:
    """一条产线班次记录（滚动预测历史中的一行）。

    stockout=1 表示该班次因缺标没开满，真实需求只知道 >= produced（右删失）；
    plan_cancelled=1 表示订单调整取消，produced=0 不是需求实现，不参与需求估计。
    """

    shift: str
    day: dt.date
    planned_at: dt.date
    plant: str
    market: str
    sku: str
    planned: int
    produced: int
    stockout: bool
    plan_cancelled: bool
    promo: bool
    received: int
    issued: int
    returned: int
    completed_at: dt.date

    @property
    def net_issued(self) -> int:
        """实际用纸标 = 实领 - 退库，恒等于出品量。"""
        return self.issued - self.returned

    def validate(self) -> None:
        if self.planned_at > self.day:
            raise ValueError(f"{self.shift}: 计划时刻晚于生产日")
        if self.plan_cancelled and self.produced != 0:
            raise ValueError(f"{self.shift}: 取消班次不应有出品")
        if self.net_issued != self.produced:
            raise ValueError(
                f"{self.shift}: 实领-退库={self.net_issued} 与出品={self.produced} 不一致"
            )
        for name, value in (
            ("planned", self.planned),
            ("produced", self.produced),
            ("received", self.received),
            ("issued", self.issued),
            ("returned", self.returned),
        ):
            if value < 0:
                raise ValueError(f"{self.shift}: {name} 为负")
        if self.stockout and self.plan_cancelled:
            raise ValueError(f"{self.shift}: 缺标与取消互斥")


@dataclass(frozen=True)
class EligibilityRule:
    """某市场某 SKU 在 [from_date, to_date] 内允许使用的纸标版本。

    exception_lot 非空时，该窗口是渠道例外：只有指定实物批号可以用，
    且仍受批号本身的可用日和占用约束。
    """

    market: str
    sku: str
    label: str
    from_date: dt.date
    to_date: dt.date
    exception_lot: str | None

    def covers(self, day: dt.date) -> bool:
        return self.from_date <= day <= self.to_date

    def validate(self) -> None:
        if self.from_date > self.to_date:
            raise ValueError(f"合格区间倒置: {self}")


@dataclass(frozen=True)
class StockLot:
    """某厂内的纸标实物批号。qty 为现存量，held 为已被先前确认占用的量。"""

    lot: str
    plant: str
    label: str
    qty: int
    held: int
    available: dt.date

    @property
    def free(self) -> int:
        return self.qty - self.held

    def validate(self) -> None:
        if self.qty < 0 or self.held < 0 or self.held > self.qty:
            raise ValueError(f"批号 {self.lot}: 库存或占用量非法")


# 已开印/已发运属于不可撤回的承诺；备选/推荐只是建议，可重算。
COMMITTED_STATUSES = frozenset({"开印", "已发运", "签收"})
RECOMMENDED_STATUSES = frozenset({"备选", "推荐"})
ALL_STATUSES = COMMITTED_STATUSES | RECOMMENDED_STATUSES


@dataclass(frozen=True)
class PrintOrder:
    """印厂整批印单。

    window_end 是印版资源的最后占用日；同一印版的两个占用区间不得重叠。
    arrives 为预计到货日，到货后进入指定工厂的同版库存。
    """

    order: str
    plate: str
    label: str
    qty: int
    starts: dt.date
    window_end: dt.date
    arrives: dt.date
    plant: str
    status: str

    @property
    def committed(self) -> bool:
        return self.status in COMMITTED_STATUSES

    def overlaps(self, other: "PrintOrder") -> bool:
        if self.plate != other.plate or self.order == other.order:
            return False
        return self.starts <= other.window_end and other.starts <= self.window_end

    def validate(self) -> None:
        if self.status not in ALL_STATUSES:
            raise ValueError(f"印单 {self.order}: 未知状态 {self.status}")
        if self.qty <= 0:
            raise ValueError(f"印单 {self.order}: 印量非正")
        if not (self.starts <= self.window_end < self.arrives):
            raise ValueError(f"印单 {self.order}: 交期顺序应为 starts<=window_end<arrives")


@dataclass(frozen=True)
class FutureNeed:
    """待保障的生产需求（生产单）。"""

    id: str
    plant: str
    market: str
    sku: str
    day: dt.date
    quantity: int
    recorded_at: dt.date

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"需求 {self.id}: 数量非正")
        if self.recorded_at > self.day:
            raise ValueError(f"需求 {self.id}: 登记日晚于需求日")


@dataclass(frozen=True)
class SwitchNotice:
    """渠道切换通知：自 day 起该市场/SKU 只准使用 label 版本。"""

    id: str
    market: str
    sku: str
    day: dt.date
    label: str
    recorded_at: dt.date

    def validate(self) -> None:
        if self.recorded_at > self.day:
            raise ValueError(f"切换 {self.id}: 通知日晚于切换日")


@dataclass(frozen=True)
class Policy:
    """当期用量承诺：窗口内任一生产单缺标概率上限与切换后不可用纸标上限。"""

    from_date: dt.date
    to_date: dt.date
    max_stockout_probability: float
    max_unusable: int
    transfer_days: int

    def validate(self) -> None:
        if not 0.0 <= self.max_stockout_probability <= 1.0:
            raise ValueError("缺标概率阈值必须在 [0,1]")
        if self.max_unusable < 0:
            raise ValueError("报废容限为负")
        if self.transfer_days < 0:
            raise ValueError("调拨在途天数为负")
        if self.from_date > self.to_date:
            raise ValueError("政策窗口倒置")


@dataclass
class Dataset:
    """校验后的完整资料容器。"""

    history: list[HistoryRow]
    eligibility: list[EligibilityRule]
    stock: list[StockLot]
    orders: list[PrintOrder]
    needs: list[FutureNeed]
    switches: list[SwitchNotice]
    policy: Policy
    source_dir: str = ""
    extras: dict = field(default_factory=dict)

    def validate(self) -> None:
        for rows in (self.history, self.eligibility, self.stock, self.orders):
            for row in rows:
                row.validate()
        for need in self.needs:
            need.validate()
        for switch in self.switches:
            switch.validate()
        self.policy.validate()
        # 已承诺印单之间的印版冲突是资料错误；备选/推荐印单与他单的冲突是计划层
        # 需要显式取舍的约束（资料里允许存在），由 planner 判定不可确认。
        for i, left in enumerate(self.orders):
            for right in self.orders[i + 1 :]:
                if left.overlaps(right) and left.committed and right.committed:
                    raise ValueError(
                        f"印版 {left.plate} 被两个已承诺印单 {left.order}/{right.order} 同时占用"
                    )

    def plate_conflicts(self) -> list[tuple[str, str, str]]:
        """返回 (印版, 印单1, 印单2) 形式的全部印版占用冲突。"""
        conflicts: list[tuple[str, str, str]] = []
        for i, left in enumerate(self.orders):
            for right in self.orders[i + 1 :]:
                if left.overlaps(right):
                    conflicts.append((left.plate, left.order, right.order))
        return conflicts
