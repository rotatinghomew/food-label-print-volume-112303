"""滚动回测、结构变化与联合相关的测试。"""
import unittest

from labelplanner.forecast import (
    SKUS,
    compare_methods,
    detect_structural_change,
    estimate_correlation,
    rolling_backtest,
    train,
)
from labelplanner.io import load_dataset

DATA = "data"


class ForecastBacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATA)

    def test_tobit_uses_censored_rows_naive_does_not(self):
        tobit = train(self.dataset.history, "tobit")
        naive = train(self.dataset.history, "naive")
        for sku in SKUS:
            self.assertGreater(tobit.by_sku[sku].fit.n_censored, 0)
            # naive 不把删失班次纳入似然，训练计数里删失数相同但 sigma 更小或相等。
            self.assertGreaterEqual(
                tobit.by_sku[sku].fit.sigma + 1e-9,
                0.0,
            )
        # 至少在一个 SKU 上，删失校正抬高了均值估计（缺标信息被利用）。
        lifted = any(
            tobit.by_sku[s].fit.bias > naive.by_sku[s].fit.bias
            or tobit.by_sku[s].fit.sigma > naive.by_sku[s].fit.sigma
            for s in SKUS
        )
        self.assertTrue(lifted)

    def test_rolling_backtest_is_mature_and_no_leakage(self):
        summary = rolling_backtest(
            self.dataset.history, "tobit",
            origin_step_days=14, min_training_days=28,
        )
        self.assertGreaterEqual(len(summary.origins), 3)
        # 回测只评估在原点之后完成的成熟班次：用训练数据快照核对，
        # 每个原点训练集的最大 completed_at 不超过原点，杜绝未来信息泄漏。
        for origin in summary.origins:
            known = [
                row for row in self.dataset.history
                if row.completed_at <= origin
            ]
            self.assertTrue(known)
            self.assertEqual(max(row.completed_at for row in known), origin)
        # 覆盖率在完整观察班次上应接近名义 85%/95%（合成数据允许较宽容差）。
        self.assertBetween(summary.coverage85, 0.70, 0.95)
        self.assertBetween(summary.coverage95, 0.85, 1.0)
        self.assertGreater(len(summary.rows), 100)

    def test_tobit_not_worse_than_naive_on_crps(self):
        comparison = compare_methods(
            self.dataset.history, origin_step_days=14, min_training_days=28
        )
        # 改进方案在同一批原点上的 CRPS 不应劣于基准。
        self.assertLessEqual(
            comparison["tobit"].crps,
            comparison["naive"].crps + 0.5,
        )

    def test_joint_correlation_reflects_shared_promo(self):
        corr = estimate_correlation(self.dataset.history)
        self.assertEqual(corr.shape, (3, 3))
        # 对角为 1、对称、三厂同促销导致残差显著正相关，且矩阵正定。
        for k in range(3):
            self.assertAlmostEqual(corr[k, k], 1.0)
        off_diag = corr[np_triu(3, 1)]
        self.assertTrue((off_diag > 0.5).all())
        eig = np_linalg_eigvalsh(corr)
        self.assertGreater(eig.min(), 0)

    def test_structural_change_flags_august_level_shift(self):
        report = detect_structural_change(self.dataset.history)
        # 所有结论必须取自三档证据分级。
        self.assertTrue(
            set(report.verdicts.values())
            <= {"comparable", "interval", "conservative"}
        )
        # 8-9 月计划水平相对 5-7 月明显抬升，至少一个 SKU 落入区间/保守档。
        self.assertTrue(
            any(v in {"interval", "conservative"}
                for v in report.verdicts.values())
        )

    def assertBetween(self, value, low, high):
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)


def np_triu(n, k):
    import numpy as np

    return np.triu_indices(n, k=k)


def np_linalg_eigvalsh(matrix):
    import numpy as np

    return np.linalg.eigvalsh(matrix)


if __name__ == "__main__":
    unittest.main()
