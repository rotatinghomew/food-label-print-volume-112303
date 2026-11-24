"""资料读取与领域不变量的校验测试。"""
import datetime as dt
import unittest

from labelplanner.io import load_dataset
from labelplanner.models import (
    HistoryRow,
    PrintOrder,
    StockLot,
)

DATA = "data"


class IoContractTest(unittest.TestCase):
    def test_loads_all_tables_and_validates(self):
        dataset = load_dataset(DATA)
        self.assertEqual(len(dataset.history), 378)
        self.assertTrue(dataset.needs)
        self.assertTrue(dataset.switches)

    def test_net_issued_equals_produced(self):
        dataset = load_dataset(DATA)
        for row in dataset.history:
            self.assertEqual(row.net_issued, row.produced)

    def test_stockout_and_cancel_are_distinct(self):
        dataset = load_dataset(DATA)
        for row in dataset.history:
            self.assertFalse(row.stockout and row.plan_cancelled)
        # 资料里两种信号都存在，验证模型确实需要区分处理。
        self.assertTrue(any(r.stockout for r in dataset.history))
        self.assertTrue(any(r.plan_cancelled for r in dataset.history))

    def test_stock_lot_rejects_invalid_held(self):
        lot = StockLot(
            lot="x", plant="A", label="old", qty=10, held=20,
            available=dt.date(2026, 10, 1),
        )
        with self.assertRaises(ValueError):
            lot.validate()

    def test_print_order_date_order_validated(self):
        order = PrintOrder(
            order="x", plate="P", label="old", qty=10,
            starts=dt.date(2026, 10, 9), window_end=dt.date(2026, 10, 8),
            arrives=dt.date(2026, 10, 10), plant="A", status="推荐",
        )
        with self.assertRaises(ValueError):
            order.validate()

    def test_history_row_identity_check(self):
        # 正常行：实领-退库 == 出品，通过校验。
        good = HistoryRow(
            shift="z", day=dt.date(2026, 10, 1), planned_at=dt.date(2026, 9, 30),
            plant="A", market="北", sku="a", planned=100, produced=90,
            stockout=False, plan_cancelled=False, promo=False,
            received=112, issued=94, returned=4,
            completed_at=dt.date(2026, 10, 2),
        )
        good.validate()
        # 矛盾行：实领-退库与出品不一致，必须报错。
        bad = HistoryRow(
            shift="z", day=dt.date(2026, 10, 1), planned_at=dt.date(2026, 9, 30),
            plant="A", market="北", sku="a", planned=100, produced=90,
            stockout=False, plan_cancelled=False, promo=False,
            received=112, issued=90, returned=4,
            completed_at=dt.date(2026, 10, 2),
        )
        with self.assertRaises(ValueError):
            bad.validate()


if __name__ == "__main__":
    unittest.main()
