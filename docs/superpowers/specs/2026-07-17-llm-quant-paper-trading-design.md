# LLM 量化模拟交易平台设计

日期：2026-07-17

状态：已批准

## 1. 背景与目标

本项目参考 `llm_quant_papers_summary.md` 中总结的 TradingAgents、FinMem、QuantaAlpha、FinGPT 等工作，实现一个美股日线级自动交易研究平台。第一版采用“确定性规则生成候选，MiniMax 受控审议，硬风控最终裁决”的结构。

系统目标是建立可回测、可重复、可审计的研究与模拟交易闭环，不承诺盈利，也不把 LLM 的自然语言判断当作订单。第一版仅支持历史回测和 paper trading，不连接券商，不使用杠杆，不做空。

## 2. 范围

### 2.1 第一版包含

- 美股复权日线行情的下载、校验和本地缓存。
- 默认股票池：`SPY`、`QQQ`、`IWM`、`AAPL`、`MSFT`、`NVDA`、`AMZN`、`GOOGL`、`META`。
- 趋势、动量、波动率和流动性规则。
- MiniMax OpenAI 兼容 API 适配器。
- 受约束的 LLM 交易审议与结构化输出。
- 事件驱动回测、paper trading、模拟账户和订单撮合。
- 交易成本、滑点、仓位、回撤和止损模型。
- 决策缓存、审计记录、基准对比和 HTML/JSON 报告。
- 可由 cron 调用的幂等 CLI 命令。

### 2.2 第一版不包含

- 真实券商接入、实盘订单或自动转账。
- 分钟级、高频、期权、期货、加密资产、做空或杠杆。
- 历史新闻回测。免费新闻源通常缺少可靠的 point-in-time 快照，直接使用会产生未来信息污染。
- LLM 自动修改风控参数或执行任意代码。
- 完整多 Agent 辩论和自动因子进化；它们预留为后续版本。

## 3. 方案选择

比较过三条路线：

1. 规则底座 + MiniMax 审议：成本和随机性可控，可与纯规则进行公平消融，作为第一版。
2. 完整 TradingAgents：角色清晰但调用次数、延迟和共同过拟合风险较高，作为后续 `v2_multi_agent`。
3. QuantaAlpha 风格因子挖掘：适合策略发现而非第一版交易闭环，作为后续 `v3_factor_mining`。

第一版不允许 MiniMax 引入规则系统未选中的股票，也不允许它提高规则给出的仓位上限。MiniMax 只能维持、降低或否决候选仓位。

## 4. 总体架构

```text
行情源 -> 数据校验/缓存 -> 特征快照 -> 策略版本 -> SignalIntent
                                                |
                                                v
报告 <- 审计/存储 <- 模拟撮合 <- 组合记账 <- 硬风控裁决
```

建议目录：

```text
src/quant_trader/
  cli.py
  config.py
  core/
    models.py
    clock.py
  data/
    base.py
    yfinance_source.py
    validation.py
    cache.py
  features/
    technical.py
    snapshot.py
  llm/
    base.py
    minimax.py
    parsing.py
    cache.py
  strategies/
    base.py
    v1_rules_llm/
      strategy.py
      rules.py
      prompt.py
    v2_multi_agent/
      README.md
    v3_factor_mining/
      README.md
  risk/
    engine.py
    limits.py
    drawdown.py
  portfolio/
    account.py
    sizing.py
  execution/
    simulator.py
    costs.py
  backtest/
    engine.py
    benchmarks.py
    walk_forward.py
  storage/
    database.py
    repositories.py
  reporting/
    metrics.py
    html.py
configs/
  default.yaml
tests/
  unit/
  integration/
  fixtures/
```

共享模块不得反向依赖具体策略版本。每个策略实现同一 `Strategy` 接口并输出 `SignalIntent`，因此后续版本不复制行情、账户、撮合、风控和报告代码。

## 5. 核心数据契约

### 5.1 MarketSnapshot

包含 `as_of`、ticker、复权 OHLCV、指标、数据质量标记和特征版本。`as_of` 表示所有字段在该时点已经可见。快照不可在创建后修改。

### 5.2 SignalIntent

包含：

- `decision_id`：由策略版本、特征版本、ticker 和 `as_of` 组成的稳定标识。
- `ticker`、`side`、`proposed_weight`。
- `signal_time`、`earliest_execution_time`。
- `stop_price`、`invalidation`、`reason_codes`。
- `strategy_version`、`prompt_version`、`llm_cache_key`。

`SignalIntent` 只是意图，不是订单。风险引擎将它转换为 `ApprovedOrder`，也可以缩仓或拒绝。

### 5.3 LLMReview

MiniMax 必须返回可由 Pydantic 校验的结构：

- `action`：`maintain`、`reduce` 或 `reject`。
- `weight_multiplier`：闭区间 `[0, 1]`。
- `confidence`：闭区间 `[0, 1]`，只用于审计，不直接放大仓位。
- `thesis`、`risks`、`invalidation`。
- `input_anomalies`：模型发现的数据疑点。

模型输出的 ticker、价格、仓位和金额不作为事实使用；系统始终使用本地快照和组合状态重新计算。

## 6. 第一版策略

### 6.1 决策频率

- 每周最后一个交易日收盘后生成调仓信号。
- 每个交易日收盘后检查数据质量、移动止损和组合回撤。
- 所有信号最早在下一交易日开盘成交。

### 6.2 规则候选

单个 ticker 至少需要 252 个有效交易日。候选必须满足：

- 收盘价高于 200 日简单移动均线。
- 20 日和 60 日收益率均为正。
- 最近 20 日平均美元成交额至少为 2,000 万美元。

候选分数为 `(0.2 × 20 日收益率 + 0.5 × 60 日收益率 + 0.3 × 120 日收益率) / 20 日年化波动率`。默认最多选择 4 个 ticker。所有周期、权重和阈值写入带版本配置，不允许根据测试期表现暗改。

### 6.3 基础仓位

- 入选标的按波动率倒数分配。
- 组合目标年化波动率为 10%。
- 单标的目标权重不超过 15%。
- 总多头敞口不超过 80%，至少保留 20% 现金。
- MiniMax 的 `weight_multiplier` 只能进一步降低基础仓位。

### 6.4 退出与止损

- 标的不再满足资格条件或 MiniMax 拒绝时，在下一交易日开盘退出。
- 默认移动止损为持仓以来最高收盘价减去 2.5 倍 ATR(14)，在收盘后判定、下一开盘执行。
- 组合从历史净值高点回撤 10% 时，将新增目标仓位减半；回撤达到 15% 时停止新增仓位并退出风险资产。熔断后必须通过显式 CLI 命令重置，不能由 LLM 恢复。

## 7. MiniMax 集成

使用 MiniMax 当前的 OpenAI 兼容 Chat Completions 接口。配置通过环境变量提供：

- `MINIMAX_API_KEY`
- `MINIMAX_BASE_URL`，默认 `https://api.minimax.io/v1`
- `MINIMAX_MODEL`，默认由配置文件指定，不把易变化的模型名硬编码进业务逻辑。

Token Plan Key 与普通按量 Key 分开管理，系统只把它当 Bearer Token 使用。仓库提供 `.env.example`，但 Key 永不写入日志、数据库、报告或 Git。

提示词只接收结构化特征、组合状态、候选规则及风险限制。第一版不接历史新闻。调用采用低随机性设置；完整输入哈希、模型名、Prompt 版本和原始输出进入审计记录。相同输入优先读取缓存。

## 8. 数据与存储

- yfinance 用于研究阶段的复权日线下载；数据源通过 `MarketDataSource` 接口隔离，以便未来替换为有服务等级协议的数据供应商。
- 行情按 ticker 和日期保存为 Parquet，写入前检查时区、重复日期、价格关系、非正价格、缺失值和异常跳变。
- SQLite 保存 paper 账户、持仓、订单、成交、策略运行、LLM 缓存和审计事件。
- 数据库迁移有显式 schema 版本。
- 下载结果必须带 `retrieved_at` 和最大行情日期。paper run 遇到过期或不完整行情时整轮停止，不沿用未知状态继续交易。

## 9. 回测与模拟执行

- 回测和 paper 账户默认初始现金均为 100,000 美元。
- 回测按交易日推进，特征只读取当前及过去数据。
- T 日收盘生成的信号只能在 T+1 开盘撮合。
- 默认使用 10 bps 单边滑点和 1 bp 单边手续费，并额外报告 25 bps 单边滑点的压力场景。
- 同一 `decision_id` 只能创建一次订单，重复运行命令不会重复成交。
- paper trading 使用与回测相同的组合、风控和撮合模块，避免两套实现产生偏差。
- 不支持真实 broker adapter，防止第一版被误配置为实盘。

## 10. 故障处理

- 行情为空、过期、日期错乱或校验失败：终止整轮运行，不创建订单。
- MiniMax 超时或限流：指数退避后有限重试。
- MiniMax 输出无法解析：允许一次结构修复请求；仍失败则该候选为 `reject`。
- 任一候选失败不影响已有持仓的硬止损和组合熔断检查。
- 数据库写入和撮合使用事务；账户、订单和成交必须保持守恒。
- 未知异常返回非零退出码并写入不含密钥的结构化日志。
- 风控规则与模型意见冲突时，始终采用更保守结果。

## 11. CLI

第一版提供：

```text
quant-trader data sync
quant-trader backtest --strategy v1_rules_llm
quant-trader paper init
quant-trader paper run
quant-trader paper status
quant-trader risk reset-circuit-breaker
quant-trader report --run-id <id>
```

命令支持配置文件和显式日期参数，适合由 cron 调用。第一版不实现常驻调度服务。`paper run` 默认需要 `--confirm`，自动定时运行必须在配置中显式启用，并且仍然只操作模拟账户。

## 12. 报告与审计

每个 run 生成机器可读 JSON 和人类可读 HTML 报告，至少包含：

- 年化收益、年化波动率、Sharpe、Sortino、最大回撤、Calmar。
- 换手率、交易次数、胜率、平均盈亏和成本占比。
- 净值曲线、回撤曲线、月度收益和持仓暴露。
- SPY 买入持有、纯规则、规则 + MiniMax 三组并列结果。
- 每次 LLM 审议、风控修改、订单和成交之间的关联链路。

若规则 + MiniMax 在未参与调参的测试期扣除成本后未优于纯规则，报告必须标记“LLM 层无已证实增益”，不能据此提升风险预算。

## 13. 测试与验证

### 13.1 自动化测试

- 指标、特征时间边界和缺失数据的单元测试。
- 仓位、现金、总敞口、止损和回撤熔断的不变量测试。
- MiniMax 正常、超时、限流、非法 JSON 和恶意字段测试。
- 重复 `decision_id`、事务失败和部分撮合的幂等测试。
- 使用固定行情和模拟 LLM 响应的离线端到端测试。
- 纯规则模式与 LLM 缓存模式的可重复性测试。

### 13.2 历史验证

历史数据按时间顺序划分为开发期、验证期和从未参与参数选择的最终测试期。默认报告使用：

- 开发期：2016-01-01 至 2021-12-31。
- 验证期：2022-01-01 至 2023-12-31。
- 最终测试期：2024-01-01 至回测运行日前最近一个完整交易日。

另外执行滚动窗口验证，以及滑点、调仓日、阈值和缺失行情的敏感性测试。不能用最终测试期反复选择参数；如需调整，必须提升策略版本并重新声明新的未触碰测试期。

### 13.3 验收条件

- 所有成交严格晚于信号时间，无未来函数。
- 所有风险不变量在单元、集成和历史验证中成立。
- LLM 不可用、输出非法或被注入无关指令时不会产生意外订单。
- 同一输入和缓存产生相同决策与报告。
- 能从空目录完成数据同步、回测、paper 初始化、一次运行和报告生成。
- 报告完整展示基准、纯规则和规则 + LLM，不隐藏失败结果。
- 组合回撤达到 15% 后不再新增风险仓位。

## 14. 上线边界与后续版本

V1 完成后仅进入 paper trading。至少稳定运行 60 个交易日、无风险越界和重复订单后，才讨论真实券商接口；届时需要单独设计和审批，不属于本规格。

后续版本通过策略目录扩展：

- `v2_multi_agent`：市场、新闻、基本面、多空辩论和交易角色，但继续复用硬风控。
- `v3_factor_mining`：受限 DSL、AST 校验、复杂度惩罚、训练/验证/测试隔离和因子进化。
- 新闻接入只有在获得可靠时间戳、许可和历史 point-in-time 数据后才能进入回测。

## 15. 参考资料

- 项目输入：`llm_quant_papers_summary.md`
- MiniMax OpenAI 兼容接口：<https://platform.minimax.io/docs/api-reference/text-chat-openai>
- MiniMax API 概览：<https://platform.minimax.io/docs/api-reference/api-overview>
- yfinance 文档与使用限制：<https://ranaroussi.github.io/yfinance/index.html>
