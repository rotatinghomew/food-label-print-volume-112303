# 包装换标用量的联合预测与印量承诺

当前资料覆盖包材计划、产线记录和正式收单资料。记录均为合成数据，使用 UTF-8 编码及带列名的 CSV。

## 运行

使用 Python 3.11。分析依赖在 `requirements.txt` 中锁定，本地环境可运行 `python3 -m pip install -r requirements.txt` 安装。在项目根目录查看文件列和记录数，并核对交接格式。

```sh
python3 inspect_inputs.py data
python3 -m unittest discover -s tests -v
```

## 记录约定

日期用工作日历的当地日期，切换日当天起只准使用新版本，渠道例外按记录所指批号及截止日。生产和纸标以张计，库存按实领和实际退回变动。history 每行是计划时刻已知的需求及后来确认的出品，受限班次的已产量可以小于未满足的真实需求。历史三条产线属于同一日促销窗口。风险约束解读为窗口内任一生产单缺标的概率和切换后不可使用的纸标量上限。

- `history.csv` 滚动预测历史，planned_at 之前不会知道实际结果。
- `eligibility.csv` 可使用的纸标版本及获准共用关系。
- `stock.csv` 库存位置、实物批号及已占用量。
- `orders.csv` 待印与签收状态，window_end 是印版资源的最后占用日。
- `future.csv` 待保障的近期生产和新到切换通知。
- `policy.csv` 当期的用量承诺及风险阈值。
