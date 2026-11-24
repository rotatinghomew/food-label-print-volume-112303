"""合成 CSV 的读取与结构校验。

沿用交接约定：UTF-8、带列名、宽度一致的 CSV。读取严格按列名取值，
缺列/空表/非法枚举直接报错，方便在试算前先核对资料。
"""
from __future__ import annotations

import csv
from pathlib import Path

from .models import (
    Dataset,
    EligibilityRule,
    FutureNeed,
    HistoryRow,
    Policy,
    PrintOrder,
    StockLot,
    SwitchNotice,
    parse_date,
    parse_int,
)

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "history.csv": {
        "shift", "day", "planned_at", "plant", "market", "sku", "planned",
        "produced", "stockout", "plan_cancelled", "promo", "received",
        "issued", "returned", "completed_at",
    },
    "eligibility.csv": {
        "market", "sku", "label", "from_date", "to_date", "exception_lot",
    },
    "stock.csv": {"lot", "plant", "label", "qty", "held", "available"},
    "orders.csv": {
        "order", "plate", "label", "qty", "starts", "window_end",
        "arrives", "plant", "status",
    },
    "future.csv": {"kind", "id", "plant", "market", "sku", "day", "quantity",
                   "label", "recorded_at"},
    "policy.csv": {
        "from_date", "to_date", "max_stockout_probability",
        "max_unusable", "transfer_days",
    },
}


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: 缺少表头")
        missing = REQUIRED_COLUMNS[path.name] - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name}: 缺少列 {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: 空表")
    for line, row in enumerate(rows, start=2):
        if any(value is None for value in row.values()):
            raise ValueError(f"{path.name} 第{line}行: 列数与表头不一致")
    return rows


def load_dataset(directory: str | Path) -> Dataset:
    directory = Path(directory)
    present = {p.name for p in directory.glob("*.csv")}
    missing = set(REQUIRED_COLUMNS) - present
    if missing:
        raise ValueError(f"资料目录缺少: {sorted(missing)}")

    history = [
        HistoryRow(
            shift=r["shift"],
            day=parse_date(r["day"]),
            planned_at=parse_date(r["planned_at"]),
            plant=r["plant"],
            market=r["market"],
            sku=r["sku"],
            planned=parse_int(r["planned"]),
            produced=parse_int(r["produced"]),
            stockout=bool(parse_int(r["stockout"])),
            plan_cancelled=bool(parse_int(r["plan_cancelled"])),
            promo=bool(parse_int(r["promo"])),
            received=parse_int(r["received"]),
            issued=parse_int(r["issued"]),
            returned=parse_int(r["returned"]),
            completed_at=parse_date(r["completed_at"]),
        )
        for r in _read_table(directory / "history.csv")
    ]

    eligibility = [
        EligibilityRule(
            market=r["market"],
            sku=r["sku"],
            label=r["label"],
            from_date=parse_date(r["from_date"]),
            to_date=parse_date(r["to_date"]),
            exception_lot=(r["exception_lot"].strip() or None),
        )
        for r in _read_table(directory / "eligibility.csv")
    ]

    stock = [
        StockLot(
            lot=r["lot"],
            plant=r["plant"],
            label=r["label"],
            qty=parse_int(r["qty"]),
            held=parse_int(r["held"]),
            available=parse_date(r["available"]),
        )
        for r in _read_table(directory / "stock.csv")
    ]

    orders = [
        PrintOrder(
            order=r["order"],
            plate=r["plate"],
            label=r["label"],
            qty=parse_int(r["qty"]),
            starts=parse_date(r["starts"]),
            window_end=parse_date(r["window_end"]),
            arrives=parse_date(r["arrives"]),
            plant=r["plant"],
            status=r["status"].strip(),
        )
        for r in _read_table(directory / "orders.csv")
    ]

    needs: list[FutureNeed] = []
    switches: list[SwitchNotice] = []
    for r in _read_table(directory / "future.csv"):
        kind = r["kind"].strip()
        if kind == "需求":
            needs.append(
                FutureNeed(
                    id=r["id"],
                    plant=r["plant"],
                    market=r["market"],
                    sku=r["sku"],
                    day=parse_date(r["day"]),
                    quantity=parse_int(r["quantity"]),
                    recorded_at=parse_date(r["recorded_at"]),
                )
            )
        elif kind == "切换":
            switches.append(
                SwitchNotice(
                    id=r["id"],
                    market=r["market"],
                    sku=r["sku"],
                    day=parse_date(r["day"]),
                    label=r["label"].strip(),
                    recorded_at=parse_date(r["recorded_at"]),
                )
            )
        else:
            raise ValueError(f"future.csv: 未知记录类型 {kind}")

    policy_rows = _read_table(directory / "policy.csv")
    if len(policy_rows) != 1:
        raise ValueError("policy.csv 当期应只有一条政策")
    r = policy_rows[0]
    policy = Policy(
        from_date=parse_date(r["from_date"]),
        to_date=parse_date(r["to_date"]),
        max_stockout_probability=float(r["max_stockout_probability"]),
        max_unusable=parse_int(r["max_unusable"]),
        transfer_days=parse_int(r["transfer_days"]),
    )

    dataset = Dataset(
        history=history,
        eligibility=eligibility,
        stock=stock,
        orders=orders,
        needs=needs,
        switches=switches,
        policy=policy,
        source_dir=str(directory),
    )
    dataset.validate()
    return dataset
