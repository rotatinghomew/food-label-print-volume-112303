"""命令行报告：把预测、回测、风险与印量计划串成一份可读的中文报告。

用法：
    python3 run_plan.py data                # 完整计划报告
    python3 run_plan.py data --backtest     # 只看滚动回测对比
    python3 run_plan.py data --paths 2000   # 调整仿真路径数
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forecast import compare_methods, train
from .io import load_dataset
from .planner import make_plan
from .risk import high_quantile_buffer
from .scenario import ScenarioBuilder, explain_window


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def backtest_report(directory: str, **kwargs) -> dict:
    dataset = load_dataset(directory)
    _print_header("一、滚动原点回测：基准(naive) vs 删失校正(tobit)")
    summaries = compare_methods(dataset.history, **kwargs)
    table = {name: summary.as_dict() for name, summary in summaries.items()}
    for name, summary in summaries.items():
        print(f"\n[{name}] {summary.as_dict()}")
    better = min(summaries, key=lambda k: summaries[k].crps)
    print(f"\n结论：CRPS 更低的方案是 {better}；tobit 显式利用缺标班次的删失信息，"
          "在结构变化后仍以近段似然校正偏差，naive 只看完整班次、天然低估。")
    return table


def plan_report(directory: str, *, n_paths: int, grid_step: int) -> dict:
    dataset = load_dataset(directory)
    bundle = train(dataset.history, method="tobit")

    _print_header("二、当前时点需求预测（删失校正）")
    for sku in ("a", "b", "c"):
        fit = bundle.by_sku[sku].fit
        verdict = bundle.structural.verdicts[sku]
        print(
            f"sku={sku}: 需求=计划量+({fit.bias:+.2f})+N(0,{fit.sigma:.2f}^2)，"
            f"训练完整班次 {fit.n_observed}、删失 {fit.n_censored}，"
            f"结构变化证据等级={verdict}"
        )
    print("\n结构变化说明：")
    for note in bundle.structural.notes:
        print(" -", note)
    print(
        "证据口径：comparable=边缘分布仍有可比证据；interval=仅近段支持点估计、"
        "旧段只给区间；conservative=样本/删失不足，只取保守上界。"
    )

    planned = {n.sku: n.quantity for n in dataset.needs}
    buffer = high_quantile_buffer(bundle, planned, 0.95)
    print("\n三厂联合 P95 高分位余量（保留合计关系，非各厂安全量相加）：")
    print(" ", buffer)

    _print_header("三、印量计划与合同风险")
    plan = make_plan(
        dataset, n_paths=n_paths, grid_step=grid_step,
        max_extra_per_label=600,
    )
    print(
        f"底线（仅已承诺印单）联合缺标概率="
        f"{plan.baseline.any_shortage_probability:.1%}，"
        f"切换后不可用 P95={plan.baseline.unusable_q95:.0f} 张"
    )
    for outcome in plan.baseline.per_need:
        print(
            f"  底线 {outcome.need.id} {outcome.need.day}: "
            f"缺标P={outcome.shortage_probability:.1%} "
            f"期望缺口={outcome.expected_shortage:.1f}"
        )

    print("\n推荐印单（未确认，可重算；已开印/已发运承诺保持不变）：")
    if not plan.recommended_orders:
        print("  无——所有候选均超风险或超印厂窗口")
    for rec in plan.recommended_orders:
        print(
            f"  {rec.label} 印版{rec.plate} {rec.qty}张，"
            f"窗口 {rec.starts}~{rec.window_end}，到货 {rec.arrives}，"
            f"原因={rec.reason_codes}，替换备选={rec.replaces or '无'}"
        )

    if plan.planned:
        print(
            f"\n采纳推荐后：联合缺标概率="
            f"{plan.planned.any_shortage_probability:.1%}，"
            f"不可用 P95={plan.planned.unusable_q95:.0f} 张，"
            f"政策阈值 缺标<={dataset.policy.max_stockout_probability:.0%}、"
            f"报废<={dataset.policy.max_unusable}张"
        )
        for outcome in plan.planned.per_need:
            print(
                f"  {outcome.need.id}: 缺标P={outcome.shortage_probability:.1%} "
                f"期望缺口={outcome.expected_shortage:.1f} "
                f"需求均值={outcome.mean_demand:.1f} P95={outcome.q95_demand:.1f}"
            )
    print(f"\n是否可签：{'是' if plan.feasible else '否'}")
    for failure in plan.failures:
        print("  不可兼得/超限：", failure)
    print("印版占用冲突（不能同版同时承诺两个交期）：", plan.plate_conflicts)

    print("\n逐需求单诊断（为什么要多印）：")
    for need_id, info in plan.reason_breakdown.items():
        print(f"  {need_id}: {info['reason_codes']}，"
              f"需求均值={info['demand_mean']}，P95={info['q95_demand']}，"
              f"可用版本={info['usable_labels']}")

    print("\n可协商项（所有候选超风险/超窗口时）：")
    for item in plan.negotiable:
        print(
            f"  需求 {item['need']} ({item['day']} {item['market']}/{item['sku']}): "
            f"缺标P={item['shortage_p']:.1%}，期望缺口={item['expected_gap']}，"
            f"可协商印版={item['negotiable_plates']}；{item['suggestion']}"
        )

    return {
        "feasible": plan.feasible,
        "failures": plan.failures,
        "baseline_any_shortage_p": plan.baseline.any_shortage_probability,
        "recommended": [
            {
                "label": r.label, "plate": r.plate, "qty": r.qty,
                "arrives": r.arrives.isoformat(),
                "reasons": r.reason_codes,
            }
            for r in plan.recommended_orders
        ],
        "negotiable": plan.negotiable,
    }


def scenario_demo(directory: str, *, n_paths: int) -> dict:
    """演示追加排期、提前切换与收单状态更新后的重算。"""
    import datetime as dt

    from .models import FutureNeed

    dataset = load_dataset(directory)
    _print_header("四、试算：提前切换日 + 追加需求 + 印单收单")
    base = ScenarioBuilder(dataset).run(n_paths=n_paths, grid_step=50)

    builder = ScenarioBuilder(dataset)
    # 南/b 切换日从 10-13 再提前到 10-11：旧标合格去处进一步收窄。
    builder.move_switch("南", "b", dt.date(2026, 10, 11))
    # 追加一条北/c 需求。
    builder.add_need(
        FutureNeed(
            id="f5", plant="A", market="北", sku="c",
            day=dt.date(2026, 10, 15), quantity=120,
            recorded_at=dt.date(2026, 10, 6),
        )
    )
    # 备选 o3 被印厂确认开印 -> 转为不可撤回承诺。
    builder.update_order_status("o3", "开印")
    scenario = builder.run(n_paths=n_paths, grid_step=50)
    diff = explain_window(base, scenario)
    print(json.dumps(diff, ensure_ascii=False, indent=2, default=str))
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="包装换标联合预测与印量承诺")
    parser.add_argument("directory", type=Path, help="CSV 资料目录")
    parser.add_argument("--paths", type=int, default=1500, help="仿真路径数")
    parser.add_argument("--grid", type=int, default=25, help="印量网格步长")
    parser.add_argument("--backtest", action="store_true", help="只跑回测对比")
    parser.add_argument("--scenario", action="store_true", help="附带试算演示")
    args = parser.parse_args()

    backtest_report(args.directory, origin_step_days=14, min_training_days=28)
    if not args.backtest:
        plan_report(args.directory, n_paths=args.paths, grid_step=args.grid)
        if args.scenario:
            scenario_demo(args.directory, n_paths=args.paths)


if __name__ == "__main__":
    main()
