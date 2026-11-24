"""需求分布与删失似然的单元测试。"""
import math
import unittest

from labelplanner.distributions import (
    CensoredSample,
    empirical_normal,
    fit_normal,
    quantile,
    survival_probability,
)


class CensoredLikelihoodTest(unittest.TestCase):
    def test_recovers_known_normal_without_censoring(self):
        rng = np_random(42)
        true_bias, true_sigma = 3.0, 10.0
        samples = [
            CensoredSample(float(rng.normal(true_bias, true_sigma)), False)
            for _ in range(2000)
        ]
        fit = fit_normal(samples)
        self.assertAlmostEqual(fit.bias, true_bias, delta=0.6)
        self.assertAlmostEqual(fit.sigma, true_sigma, delta=0.6)
        self.assertTrue(fit.converged)

    def test_censored_fit_corrects_downward_bias_of_naive(self):
        # 构造一批被右删失的样本：naive 只看残差较小的完整观察，会低估均值与波动。
        rng = np_random(7)
        true_bias, true_sigma = 0.0, 20.0
        raw = [rng.normal(true_bias, true_sigma) for _ in range(3000)]
        cutoff = 10.0  # 残差超过 10 的班次因缺标被截断
        samples = [
            CensoredSample(min(x, cutoff), x > cutoff) for x in raw
        ]
        # 真实的右删失比例（用于断言构造确实产生了大量删失）。
        self.assertGreater(sum(s.censored for s in samples), 800)
        naive = empirical_normal(samples)
        tobit = fit_normal(samples)
        # 删失校正把均值从"只看小残差"的负向上拉，向真实 0 靠拢。
        self.assertLess(naive.bias, tobit.bias)
        self.assertAlmostEqual(tobit.bias, true_bias, delta=1.5)
        self.assertGreater(tobit.sigma, naive.sigma)
        # 朴素矩估计明确有偏（绝对值显著偏离 0），而 tobit 偏差小一个量级。
        self.assertLess(abs(tobit.bias), abs(naive.bias))

    def test_quantile_and_survival_are_consistent(self):
        samples = [CensoredSample(float(x), False) for x in (-5, 0, 5, 10, -10)]
        fit = fit_normal(samples)
        q85 = quantile(fit, 0.85)
        # P(残差 > q85) 应约为 15%。
        self.assertAlmostEqual(survival_probability(fit, q85), 0.15, delta=0.02)

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            fit_normal([CensoredSample(1.0, True)])


def np_random(seed):
    import numpy as np

    return np.random.default_rng(seed)


if __name__ == "__main__":
    unittest.main()
