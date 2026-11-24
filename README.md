# 包装换标用量的联合预测与印量承诺

同一饮料的纸标可在多个规格/市场共用，但渠道切换、批号例外与各厂各自订量会
同时造成旺季缺标与淡季剩标报废。本包用删失感知的联合需求预测、联合蒙特卡洛
风险仿真和印厂窗口/印版约束下的印量计划，回答"下一轮印单怎么签、能不能签、
签不了时该协商什么"。记录均为合成数据，UTF-8 编码、带列名 CSV。

## 运行

使用 Python 3.11，依赖锁定在 `requirements.txt`（numpy、scipy）。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_inputs.py data            # 核对交接格式
python3 run_plan.py data                  # 完整报告：回测+预测+计划
python3 run_plan.py data --backtest       # 只看滚动回测对比
python3 run_plan.py data --scenario       # 追加排期/提前切换/收单试算
python3 -m unittest discover -s tests -v  # 38 项单元测试
```

## 业务口径如何落到代码

| 业务要求 | 实现位置 |
| --- | --- |
| 区分"用量低"与"缺标没开出来" | `distributions.fit_normal` 对右删失班次用生存函数做 Tobit 似然；取消班次剔除 |
| 按事发时已知信息回测，不泄漏未来 | `forecast.rolling_backtest` 按 `planned_at`/`completed_at` 滚动，原点只用已完成班次 |
| 基准 vs 可解释改进 | naive（完整班次矩估计）对比 tobit（删失 MLE），同原点比较 MAE/RMSE/CRPS/覆盖率 |
| 三厂联合缺标概率、保留合计关系 | `risk.simulate` 用估计相关矩阵生成联合需求路径，解析给出合计 P95 余量 |
| 结构变化后的证据分级 | `forecast.detect_structural_change` 输出 comparable/interval/conservative |
| 逐日判断可用版本与库存占用 | `eligibility.EligibilityIndex` + `inventory.build_ledger`（held 不可再分） |
| 跨厂联合调拨而非各订安全量 | `risk._allocate_path` 的 transfer_filter，只允许显式共用版跨市场，含在途天数 |
| 已承诺印单必计入、未确认推荐重算 | `inventory` 区分 committed/推荐；`planner._simulate_committed_only` 给底线 |
| 同一印版不能承诺两个交期 | `models.PrintOrder.overlaps` 与 `planner._candidate_orders` 占用区间顺移 |
| 合同缺标概率与报废容限 | `risk.SimulationResult.passes_policy`，超限时列出不可兼得项 |
| 候选全超风险/窗口时给协商项 | `planner.make_plan` 输出 negotiable（需求日、印版、阻塞原因） |
| 提前切换改变旧标可用性与调拨优先级 | `scenario.ScenarioBuilder.move_switch`，切换日起普通旧标失效、例外批号保留 |
| 反复试算且已签印单业务账可查 | `scenario` 不可变叠加；承诺印单不可回退、始终在 orders 台账中 |
| 解释为什么多印 | `planner._reason_breakdown`：总需求上升/需求联动/旧标无合格去处 |

## 模块

- `models.py` 领域记录与校验（库存、印单、需求、切换、政策的不变量）
- `io.py` 六张 CSV 的严格读取与结构校验
- `eligibility.py` 版本/共用/渠道例外/切换通知的可用性索引
- `distributions.py` 删失正态似然与分位数/生存概率
- `forecast.py` 边缘分布、联合相关、滚动回测、结构变化分级
- `inventory.py` 批号余量、在途到货、跨厂调拨台账
- `risk.py` 联合蒙特卡洛、缺标概率、高分位余量、报废风险
- `planner.py` 印量网格搜索、印版窗口排产、合同风险判定、协商清单
- `scenario.py` 追加排期/切换/收单状态的试算与窗口诊断
- `report.py` / `run_plan.py` 中文报告与命令行入口

## 关键建模说明

1. **删失而非零需求**：`stockout=1` 只告诉我们真实需求 ≥ 当班出品，直接当普通
   观察会系统性低估；`plan_cancelled=1` 的班次不携带需求信息，不进估计。
2. **联合而不独立**：三厂同促销，残差相关 0.9+，整批安全量由联合协方差推出，
   不能把各厂安全量相加；高分位余量同时给分 SKU 与合计两口径。
3. **共用版才跨市场**：市场专属 old/new 即使同名也不能跨市场调拨，只有同窗口
   绑定到不同市场不同 SKU 的共用版（资料中的 `shared`）可以联合调拨。
4. **报废只认切换**：政策窗口内失去合格去处的旧标余量计入不可用；新版/共用版
   窗口延至窗口之后、仍可周转，剩余属安全库存而非报废。
5. **印版是物理约束**：同一印版占用区间不可重叠；推荐印单可在同版串行排产，
   但与已承诺印单冲突时只能协商释放或改期。
