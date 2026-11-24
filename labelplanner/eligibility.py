"""纸标版本可用性、渠道共用关系与切换通知。

同一款饮料的纸标可能有多个版本（old/new/shared），部分市场允许共用。
切换通知会临时提前切换日：切换日当天起只准使用新版本，旧标只剩渠道例外
（指定批号 + 截止日）这一条合法去处。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .models import Dataset, EligibilityRule, SwitchNotice


@dataclass(frozen=True)
class LabelOption:
    """某需求日某市场/SKU 可以使用的一个纸标版本。"""

    label: str
    exception_lot: str | None  # 非空表示只有该实物批号可走渠道例外

    @property
    def is_exception(self) -> bool:
        return self.exception_lot is not None


class EligibilityIndex:
    def __init__(self, dataset: Dataset):
        self._rules = dataset.eligibility
        self._switches = tuple(dataset.switches)

    @property
    def switches(self) -> tuple[SwitchNotice, ...]:
        return self._switches

    def _switch_day(self, market: str, sku: str) -> dt.date | None:
        """生效中的切换日：取该市场/SKU 最早一条已登记通知。

        资料可能追加更晚的通知；计划时刻 recorded_at 之前的通知才视为已知。
        """
        days = [
            s.day
            for s in self._switches
            if s.market == market and s.sku == sku
        ]
        return min(days) if days else None

    def options(
        self,
        market: str,
        sku: str,
        day: dt.date,
        as_of: dt.date | None = None,
    ) -> list[LabelOption]:
        """返回需求日当天该市场/SKU 的全部合法纸标版本。

        as_of 用于回测：只采用该时点已登记的切换通知，避免把未来信息泄漏进计划。
        """
        switch_day = None
        candidates = [
            s
            for s in self._switches
            if s.market == market and s.sku == sku
            and (as_of is None or s.recorded_at <= as_of)
        ]
        if candidates:
            switch_day = min(s.day for s in candidates)

        options: list[LabelOption] = []
        for rule in self._rules:
            if rule.market != market or rule.sku != sku or not rule.covers(day):
                continue
            if switch_day is not None and day >= switch_day and not rule.exception_lot:
                # 切换日起，普通旧标窗口失效；渠道例外窗口（带 exception_lot）保留。
                if rule.label != self._switch_label(market, sku, as_of):
                    continue
            options.append(LabelOption(rule.label, rule.exception_lot))
        # 去重并保持稳定顺序：普通版优先，例外版随后。
        seen: set[tuple[str, str | None]] = set()
        unique: list[LabelOption] = []
        for option in sorted(options, key=lambda o: (o.is_exception, o.label)):
            key = (option.label, option.exception_lot)
            if key not in seen:
                seen.add(key)
                unique.append(option)
        return unique

    def _switch_label(self, market: str, sku: str, as_of: dt.date | None) -> str:
        labels = [
            s.label
            for s in self._switches
            if s.market == market and s.sku == sku
            and (as_of is None or s.recorded_at <= as_of)
        ]
        if not labels:
            raise RuntimeError("切换标签缺失")
        # 同一市场/SKU 多条通知指向同一新版；若不一致取最早通知所指版本。
        earliest = min(
            (s for s in self._switches
             if s.market == market and s.sku == sku
             and (as_of is None or s.recorded_at <= as_of)),
            key=lambda s: s.day,
        )
        return earliest.label

    def rules_for(self, market: str, sku: str) -> list[EligibilityRule]:
        return [r for r in self._rules if r.market == market and r.sku == sku]

    def shared_markets(self, label: str) -> set[str]:
        """使用某共用版纸标的市场集合（用于联合调拨与联合缺标）。"""
        return {r.market for r in self._rules if r.label == label and not r.exception_lot}

    def is_cross_market_shared(self, label: str) -> bool:
        """该版本是否为显式的跨市场共用纸标。

        数据里每个 (市场, SKU) 有自己的版本换代链（old -> new，部分带渠道
        例外），主版本只在本市场内出现；真正的共用版是额外挂到 *别的市场的
        SKU* 上、且不参与那个 SKU 换代链的版本。判据：

        1. 该版本出现在两个不同市场、两个不同 SKU 上；
        2. 对每个被绑定的 (市场, SKU)，该市场/SKU 都存在一个不含本版本名的
           主版本（即换代链 old/new），说明本版本是"外挂"的共用关系；
        3. 各市场上本版本的窗口一致。

        new 在两个市场各自就是换代链主版本，判据 2 不成立；old 同理。
        """
        rules = [
            r for r in self._rules
            if r.label == label and not r.exception_lot
        ]
        pairs = {(r.market, r.sku) for r in rules}
        # 至少出现在两个不同市场。
        if len({m for m, _ in pairs}) < 2:
            return False
        # 换代链版本（old/new）按定义就是市场专属主版本，直接排除。
        if label in {"old", "new"}:
            return False
        # 共用版在至少一个被绑定市场上是"外挂"在该市场换代链之外的备用关系；
        # 纯共用版的 SKU（如 北/c 只有 shared）则要求另一个市场存在换代链。
        chain_hits = 0
        for market, sku in pairs:
            own_versions = {
                r.label
                for r in self._rules
                if r.market == market and r.sku == sku and not r.exception_lot
            }
            if own_versions & {"old", "new"}:
                chain_hits += 1
        if chain_hits == 0:
            return False
        windows = {(r.from_date, r.to_date) for r in rules}
        return len(windows) == 1
