"""合格关系、切换通知与跨厂调拨合法性的测试。"""
import datetime as dt
import unittest

from labelplanner.eligibility import EligibilityIndex
from labelplanner.io import load_dataset
from labelplanner.risk import _is_shared_label

DATA = "data"


class EligibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATA)
        cls.index = EligibilityIndex(cls.dataset)

    def test_shared_label_classification(self):
        self.assertTrue(_is_shared_label(self.dataset, "shared"))
        self.assertFalse(_is_shared_label(self.dataset, "old"))
        self.assertFalse(_is_shared_label(self.dataset, "new"))

    def test_options_before_and_after_switch(self):
        # 南/b 在切换日 10-13 前 old/new 都可用；10-13 当天起只剩 new（普通 old 失效）。
        before = {o.label for o in self.index.options("南", "b", dt.date(2026, 10, 12))}
        after = {o.label for o in self.index.options("南", "b", dt.date(2026, 10, 13))}
        self.assertIn("old", before)
        self.assertIn("new", before)
        self.assertNotIn("old", after)
        self.assertIn("new", after)

    def test_exception_lot_window_survives_switch(self):
        # 渠道例外：L2 在 10-17~10-19 仍是南/b 旧标的合法批号。
        options = self.index.options("南", "b", dt.date(2026, 10, 18))
        exceptions = [o for o in options if o.is_exception]
        self.assertTrue(exceptions)
        self.assertEqual(exceptions[0].exception_lot, "L2")
        # 例外窗口外（10-20）旧标彻底无合格去处。
        self.assertFalse(
            any(o.label == "old"
                for o in self.index.options("南", "b", dt.date(2026, 10, 20)))
        )

    def test_as_of_hides_future_switch_notice(self):
        # 通知 10-05 登记；10-04 的计划时刻不应看到切换，old 仍合格。
        early = self.index.options(
            "南", "b", dt.date(2026, 10, 14), as_of=dt.date(2026, 10, 4)
        )
        self.assertIn("old", {o.label for o in early})

    def test_early_switch_narrows_old_label_usability(self):
        from labelplanner.scenario import ScenarioBuilder

        builder = ScenarioBuilder(self.dataset)
        builder.move_switch("南", "b", dt.date(2026, 10, 11))
        moved = EligibilityIndex(builder.dataset)
        options = moved.options("南", "b", dt.date(2026, 10, 12))
        self.assertNotIn("old", {o.label for o in options if not o.is_exception})


if __name__ == "__main__":
    unittest.main()
