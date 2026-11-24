"""库存占用、联合风险仿真与印量计划约束的测试。"""
import datetime as dt
import unittest

import numpy as np

from labelplanner.forecast import train
from labelplanner.inventory import build_ledger, committed_orders
from labelplanner.io import load_dataset
from labelplanner.models import PrintOrder
from labelplanner.planner import make_plan
from labelplanner.risk import simulate

DATA = "data"


class InventoryAndRiskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATA)
        cls.bundle = train(cls.dataset.history, "tobit")

    def test_held_stock_is_not_available(self):
        ledger = build_ledger(self.dataset)
        by_lot = {p.lot: p for p in ledger.lots}
        # L1 现存 280、占用 40 -> 可分配 240。
        self.assertEqual(by_lot["L1"].free_qty, 240)
        self.assertEqual(by_lot["L2"].free_qty, 150)

    def test_market_specific_labels_never_cross_markets(self):
        result = simulate(self.dataset, self.bundle, n_paths=600)
        for alloc in result.allocations:
            if alloc["transferred"]:
                # 跨厂调拨只能发生在共用版上。
                self.assertEqual(alloc["label"], "shared")

    def test_joint_shortage_respects_total_relationship(self):
        result = simulate(self.dataset, self.bundle, n_paths=1000)
        # 任一缺标概率介于单厂最大概率与 1 之间，且合计 P95 >= 各单 P95 之和的下界。
        per = [o.shortage_probability for o in result.per_need]
        self.assertGreaterEqual(result.any_shortage_probability, max(per) - 1e-9)
        self.assertLessEqual(result.any_shortage_probability, 1.0 + 1e-9)
        # 合计需求 P95 必须大于均值（保留联合波动，不是简单相加均值）。
        self.assertGreater(result.total_demand_q95, result.total_demand_mean)

    def test_committed_orders_always_counted_in_supply(self):
        committed = committed_orders(self.dataset)
        self.assertTrue(any(o.order == "o1" for o in committed))
        ledger = build_ledger(self.dataset)
        orders = {b.order: b for b in ledger.incoming}
        self.assertIn("o1", orders)
        self.assertTrue(orders["o1"].committed)

    def test_recommended_orders_are_recomputed_not_committed(self):
        # 传入推荐印单时，台账应只保留承诺单 + 新推荐，旧备选被替换。
        extra = [
            PrintOrder(
                order="new-rec", plate="P2", label="shared", qty=100,
                starts=dt.date(2026, 10, 7), window_end=dt.date(2026, 10, 10),
                arrives=dt.date(2026, 10, 12), plant="B", status="推荐",
            )
        ]
        ledger = build_ledger(self.dataset, extra_orders=extra)
        ids = {b.order for b in ledger.incoming}
        self.assertIn("o1", ids)          # 承诺保留
        self.assertIn("new-rec", ids)     # 新推荐进入
        self.assertNotIn("o2", ids)       # 旧备选被整体替换
        self.assertNotIn("o3", ids)

    def test_same_plate_cannot_commit_two_overlapping_windows(self):
        plan = make_plan(self.dataset, n_paths=300, grid_step=50)
        recs = [r.as_print_order(f"r{i}") for i, r in enumerate(plan.recommended_orders)]
        for i, left in enumerate(recs):
            for right in recs[i + 1:]:
                if left.plate == right.plate:
                    self.assertFalse(left.overlaps(right))
        # 资料中 o1/o2 的 P1 冲突必须被显式报告。
        plates = {c[0] for c in plan.plate_conflicts}
        self.assertIn("P1", plates)

    def test_hard_gap_when_no_printer_window_is_negotiable(self):
        plan = make_plan(self.dataset, n_paths=300, grid_step=50)
        # f1(10-08) 在现有印版窗口下无法及时到货，必然进入协商清单。
        negotiable_ids = {item["need"] for item in plan.negotiable}
        self.assertIn("f1", negotiable_ids)
        self.assertFalse(plan.feasible)
        self.assertTrue(plan.failures)

    def test_extra_supply_can_bring_shortage_risk_within_policy(self):
        # 充足且及时的印单能把"缺标风险"压到阈值内；这证明不可行来自印厂窗口/
        # 印版约束，而不是风险模型本身。提前切换导致的旧标报废是另一条独立约束，
        # 由 test_switch_strands_old_label 单独验证。
        extra = [
            PrintOrder(
                order="h1", plate="P2", label="new", qty=150,
                starts=dt.date(2026, 10, 3), window_end=dt.date(2026, 10, 5),
                arrives=dt.date(2026, 10, 7), plant="A", status="推荐",
            ),
            PrintOrder(
                order="h2", plate="P2", label="new", qty=400,
                starts=dt.date(2026, 10, 5), window_end=dt.date(2026, 10, 8),
                arrives=dt.date(2026, 10, 10), plant="B", status="推荐",
            ),
            PrintOrder(
                order="h3", plate="P2", label="shared", qty=250,
                starts=dt.date(2026, 10, 8), window_end=dt.date(2026, 10, 10),
                arrives=dt.date(2026, 10, 12), plant="B", status="推荐",
            ),
        ]
        result = simulate(self.dataset, self.bundle, n_paths=800, extra_orders=extra)
        self.assertLessEqual(
            result.any_shortage_probability,
            self.dataset.policy.max_stockout_probability + 1e-9,
        )

    def test_switch_strands_old_label_above_scrap_tolerance(self):
        # 提前切换(10-13)后，南/b 旧标 L2(150) 在窗口内无合格需求日，
        # 构成超过容限(110)的报废；把切换推回 10-17 后 L2 可用于 f2，报废归零。
        current = simulate(self.dataset, self.bundle, n_paths=600)
        self.assertGreater(current.unusable_q95,
                           self.dataset.policy.max_unusable)
        from labelplanner.scenario import ScenarioBuilder

        moved = ScenarioBuilder(self.dataset).move_switch(
            "南", "b", dt.date(2026, 10, 17)
        ).dataset
        relaxed = simulate(moved, self.bundle, n_paths=600)
        self.assertEqual(relaxed.unusable_q95, 0.0)

    def test_deterministic_simulation_with_fixed_seed(self):
        r1 = simulate(self.dataset, self.bundle, n_paths=400,
                      rng=np.random.default_rng(123))
        r2 = simulate(self.dataset, self.bundle, n_paths=400,
                      rng=np.random.default_rng(123))
        self.assertEqual(r1.any_shortage_probability, r2.any_shortage_probability)


if __name__ == "__main__":
    unittest.main()
