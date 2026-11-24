"""包装换标用量的联合预测与印量承诺。

模块划分：
- models:   领域记录与校验后的资料容器
- io:       合成 CSV 的读取与结构校验
- eligibility: 纸标版本/共用关系/切换通知的可用性索引
- forecast: 区分低用量与缺标删失的需求模型、滚动回测与结构变化判断
- inventory: 库存占用、在途印单与跨厂联合调拨
- risk:     联合缺标概率与高分位余量的仿真（保留合计关系）
- planner:  逐日印量计划、印厂窗口/印版约束、风险与报废容限
- scenario: 追加排期/切换/收单状态的试算与诊断
"""

from .models import (
    HistoryRow,
    EligibilityRule,
    StockLot,
    PrintOrder,
    FutureNeed,
    SwitchNotice,
    Policy,
    Dataset,
)

__all__ = [
    "HistoryRow",
    "EligibilityRule",
    "StockLot",
    "PrintOrder",
    "FutureNeed",
    "SwitchNotice",
    "Policy",
    "Dataset",
]
