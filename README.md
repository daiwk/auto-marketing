# quant-trader

一个刻意保持小而安全的美股量化研究与**纸面交易**工具。项目内置经过校验的
2023—2025 年日线数据，支持规则策略、MiniMax/Codex/Trae X 辅助决策、TradingAgents 多 Agent
分析、历史回测和单次纸面交易。

> **免责声明：** 本项目仅用于研究和纸面模拟，不构成投资建议。项目没有实盘券商接口，
> 不能发送真实订单。历史模拟结果不代表未来收益。

## 本地实验网站

通过页面选择模式、参数并观察后台过程，可以启动统一实验平台：

```bash
quant-trader web \
  --config configs/default.yaml \
  --data-root data \
  --output-root web-runs
```

命令会自动打开浏览器；如果没有自动打开，复制终端打印的完整地址。地址带有随机访问
令牌，并且服务只监听 `127.0.0.1`，不要转发到公网。需要指定端口或禁止自动打开时，可
添加 `--port 8080 --no-open-browser`。

页面支持规则基线、TradingAgents、FinMem、QuantaAlpha 和 Alpha Arena。前三类 LLM
实验可选择本地 Codex、本地 Trae X 或国内 MiniMax M3；Alpha Arena 可以选择已经完成的实验进行横向
比较。提交后，页面每秒刷新任务状态：“中间过程监控”显示排队、进程、Agent/模型阶段和
CLI 日志；“实验效果”显示收益、回撤、候选因子或排行榜，并保留原始 JSON 供核查。

TradingAgents 运行时会额外显示 12 个角色的实时决策看板。角色完成后即可点击查看其立场、
信心、摘要、依据和风险；交易员建议与最终组合经理结论单独展示。回测完成后，页面会绘制
规则策略、LLM 策略和 SPY 买入持有的净值曲线，便于直接比较最终实验效果。鼠标移到曲线
上可查看具体日期和各策略净值；走势接近时使用不同线型区分。新运行生成的 Agent 结论默认
使用简体中文。配置多次完整审核时，看板顶部会按“第 1 次、第 2 次……”保留每次工作流，
运行中和结束后均可切换查看对应标的、全部角色结论、交易员建议和最终组合决策。

“回测与算法参数”区域可以直接配置本次实验使用的全部基础参数：股票池、初始资金、候选数、
流动性门槛、目标波动率、单股/总仓位、现金比例、两级回撤阈值、ATR 止损、滑点和手续费。
页面顶部还可以选择回测开始和结束日期；结束日期不包含当天，默认 `2023-01-01` 至
`2026-01-01`，即覆盖内置的 2023—2025 年行情。所选日期必须落在本地行情数据覆盖范围内。

TradingAgents 的模型审核次数与 Alpha Arena 的参赛实验也会按模式显示。一次“审核机会”
是某个周度调仓日中，一个通过规则筛选、准备交给 LLM 复核的候选标的。日志中的总机会数
（例如 341）是整个时间窗口内这些候选的合计，不是模型请求次数；一个完整 TradingAgents
审核还会依次调用多个角色。有限的审核次数可以选择“均匀分布”（网页默认）、“最前 N 次”
或填写从 1 开始的具体机会序号。提交后平台在
`web-runs/<任务 ID>/config.yaml` 生成独立配置，不修改 `configs/default.yaml`；本次参数同时
显示在任务接口和运行记录中，方便复现。

关闭网页不会中止后台任务，停止启动命令则会终止仍在运行的子进程。运行记录保存在
`web-runs/<任务 ID>/`。MiniMax Key 仍只从启动网站的终端环境读取，网页没有 Key 输入框，
也不会把 Key 写入日志或实验文件：

```bash
export MINIMAX_API_KEY='你的 MiniMax API Key'
quant-trader web
```

Codex 和 Trae X 模式复用各自 CLI 的本机登录状态。平台只允许固定的研究命令和经过校验的
参数，不能提交任意命令、路径或实盘交易请求。

## 安装与快速开始

需要 Python 3.12 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

先运行完全离线的规则回测：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run.json

quant-trader report --run-json run.json --output report.html
```

纸面交易命令：

```bash
quant-trader paper init --db paper.db
quant-trader paper status --db paper.db
quant-trader paper run --db paper.db --config configs/default.yaml --confirm
```

`paper run` 只更新本地 SQLite 状态，不会连接券商或发送实盘订单。

## 本地行情数据

代码库内的数据快照覆盖全部默认标的，日期为 2023-01-03 至 2025-12-31，因此上述回测
不需要联网。数据来源和校验和见 [`data/SOURCES.md`](data/SOURCES.md)。

需要更新数据时，优先使用新浪财经：

```bash
quant-trader data sync \
  --source sina \
  --config configs/default.yaml \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --data-root data
```

Yahoo 只作为显式备用源，使用 `--source yahoo` 开启。

## 策略与安全限制

默认模式完全由本地规则驱动。启用 LLM 后，模型也只能审核规则已经选出的候选标的，
不能新增标的或提高规则给出的仓位。

默认硬限制包括：

- 只做多，不做空，不使用杠杆；
- 单个标的最大仓位 15%；
- 总持仓最大 80%，至少保留 20% 现金；
- 回撤达到阈值后自动降仓或锁定停止；
- LLM 或任一 Agent 出错时按 `reject / 0` 拒绝仓位；
- 所有执行均为历史模拟或纸面交易。

## 配置 MiniMax M3

默认配置面向国内 MiniMax Token Plan：

- API 地址：`https://api.minimaxi.com/v1`
- 模型：`MiniMax-M3`
- API Key 环境变量：`MINIMAX_API_KEY`

先在当前终端配置 Key。不要把 Key 写入 YAML 或提交到代码库：

```bash
export MINIMAX_API_KEY='你的 MiniMax API Key'
```

如果你的账号使用其他地址或模型，可以覆盖：

```bash
export MINIMAX_BASE_URL='https://api.minimaxi.com/v1'
export MINIMAX_MODEL='MiniMax-M3'
```

普通单审核回测可能产生很多调用，建议先限制为 3 次：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-minimax.json \
  --use-llm \
  --llm-max-reviews 3
```

达到上限后，剩余审核点会自动使用本地规则回复，输出文件会注明这次运行被截断。

## 使用本地 Codex

Codex 模式使用本机 Codex CLI 的登录状态，不需要 `MINIMAX_API_KEY`。先检查：

```bash
codex --version
codex login status
```

如果未登录，执行 `codex login`。然后运行一次审核：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-codex.json \
  --use-llm \
  --llm-provider codex \
  --llm-max-reviews 1
```

Codex 调用是临时、只读的。未指定 `--llm-max-reviews` 时，普通 Codex 回测默认只调用
3 次真实审核，之后回退本地规则。

## 使用本地 Trae X

Trae X 模式使用本机 `traex` CLI 的登录状态，同样不需要 `MINIMAX_API_KEY`。先检查：

```bash
traex --version
traex login status
```

如果未登录，执行 `traex login`。命令行中把 Provider 设置为 `traex`：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-traex.json \
  --use-llm \
  --llm-provider traex \
  --llm-max-reviews 1
```

网页中选择“本地 Trae X”后，模型下拉框会显示本机 `traex models` 返回的全部可用模型；
默认使用 `gpt-5.5`，避免继承全局配置中可能需要排队的模型。所选模型只写入本次任务的
独立配置。每次调用都在临时目录和只读沙箱中非交互执行，只读取最终回复；任务记录不会
保存完整 Prompt、原始模型输出或隐藏推理。

## TradingAgents 多 Agent 工作流

`trading-agents` 是固定单轮流程，不会递归调用工具。它包含 12 个逻辑角色：

1. 市场分析师
2. 情绪分析师
3. 新闻分析师
4. 基本面分析师
5. 多方研究员
6. 空方研究员
7. 研究经理
8. 交易员
9. 激进风险分析师
10. 中性风险分析师
11. 保守风险分析师
12. 投资组合经理

没有额外上下文时，情绪、新闻和基本面角色会显示为 `skipped`，不会浪费 LLM 调用。
此时一个完整工作流需要 9 次模型调用；三类上下文齐全时最多 12 次。

### 单个标的一次性分析

MiniMax：

```bash
export MINIMAX_API_KEY='你的 MiniMax API Key'

quant-trader agents analyze \
  --ticker SPY \
  --as-of 2025-12-31 \
  --config configs/default.yaml \
  --data-root data \
  --output agent-run.json \
  --llm-provider minimax
```

Codex：

```bash
quant-trader agents analyze \
  --ticker SPY \
  --as-of 2025-12-31 \
  --config configs/default.yaml \
  --data-root data \
  --output agent-run.json \
  --llm-provider codex
```

只有确定性规则判定合格的标的才会调用 LLM。不合格时命令仍会正常生成 JSON，但
`eligible` 为 `false`、`provider_calls` 为 `0`。

### 在历史回测中启用多 Agent

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-agents.json \
  --use-llm \
  --llm-provider minimax \
  --llm-workflow trading-agents
```

回测默认只运行 **1 个完整多 Agent 工作流**，之后的审核点回退本地规则。这样可以避免
第一次运行就消耗大量额度。如果确实要运行更多完整工作流，显式设置：

```bash
--llm-max-reviews 2
```

这里的 `2` 表示两个完整 TradingAgents 工作流，不是两个单独的 Agent 调用。

命令行默认审核最前面的机会，也可以把有限额度分散到整个回测区间：

```bash
--llm-max-reviews 4 --llm-review-schedule evenly
```

若总机会数为 341，上述配置会选择第 1、114、228、341 次机会。也可以精确指定：

```bash
--llm-max-reviews 4 --llm-review-schedule custom --llm-review-indices 1,114,228,341
```

自定义序号必须唯一、从 1 开始，且数量等于 `--llm-max-reviews`。开始和结束日期可通过
`--start 2024-01-01 --end 2025-01-01` 设置，其中结束日期不包含当天。

## 实时决策 Dashboard：详细使用方法

Dashboard 用于观察 `agents analyze` 和 TradingAgents 回测运行期间的中间决策过程。
它不是独立命令，只需要在原命令最后增加 `--dashboard`。

目前 Dashboard **不接入** `paper run`，也不提供任何买卖按钮。它只是只读观察页面。

### 方法一：观察单个标的分析（最容易理解）

建议第一次先用这个方式。

#### 第 1 步：激活环境

```bash
source .venv/bin/activate
```

如果使用 MiniMax，再确认 Key 已配置：

```bash
export MINIMAX_API_KEY='你的 MiniMax API Key'
```

#### 第 2 步：运行带 Dashboard 的分析

下面的 `SPY / 2025-12-31` 在仓库内置数据中具有完整历史，适合作为示例：

```bash
quant-trader agents analyze \
  --ticker SPY \
  --as-of 2025-12-31 \
  --config configs/default.yaml \
  --data-root data \
  --output agent-run.json \
  --llm-provider minimax \
  --dashboard
```

使用 Codex 时只需替换 Provider：

```bash
quant-trader agents analyze \
  --ticker SPY \
  --as-of 2025-12-31 \
  --config configs/default.yaml \
  --data-root data \
  --output agent-run.json \
  --llm-provider codex \
  --dashboard
```

#### 第 3 步：观察终端输出

命令启动后会打印类似内容：

```text
Dashboard: http://127.0.0.1:54321/一段随机token/
Agent market_analyst started.
Agent market_analyst completed.
...
```

浏览器通常会自动打开。如果没有自动打开，请复制终端中完整的 `Dashboard:` URL 到浏览器，
包括最后的随机 token 和 `/`，不要只复制端口。

#### 第 4 步：理解页面

页面顶部显示当前标的、分析日期、Provider、命令状态和已完成的工作流数量。

12 个角色节点有五种状态：

- `waiting`：等待上游角色完成；
- `running`：当前正在调用 LLM；
- `completed`：已得到并验证结构化结论；
- `skipped`：没有对应时间点的外部上下文，因此主动跳过且不调用 LLM；
- `failed`：输出无效或 Provider 失败，工作流将按安全规则拒绝仓位。

点击任意已完成节点，可以查看：

- 该角色的看多、看空或中性立场；
- 信心分数；
- 结构化摘要；
- 使用的依据；
- 识别出的风险和输入异常。

页面下方还会依次出现：

- **交易员建议**：`maintain / reduce / reject` 和权重倍数；
- **最终组合决策**：风险评审后的最终动作；
- **安全边界**：不能新增标的、不能提高规则仓位、失败自动拒绝、仅纸面交易。

#### 第 5 步：等待命令结束

不要因为某个角色运行时间较长就关闭终端。MiniMax/Codex 是逐角色同步调用，一个完整流程
需要 9—12 次模型调用，耗时可能是数分钟。页面和终端都会显示当前具体卡在哪个角色。

命令结束后：

- 完整审计结果保存在 `agent-run.json`；
- 已打开的页面会保留最后一次渲染结果；
- 本地 Dashboard 服务会自动关闭；
- 此时不要刷新页面，刷新后无法重新连接是正常现象。

### 方法二：观察回测中的多 Agent 决策

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-dashboard.json \
  --use-llm \
  --llm-provider minimax \
  --llm-workflow trading-agents \
  --dashboard
```

Codex 版本：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-dashboard.json \
  --use-llm \
  --llm-provider codex \
  --llm-workflow trading-agents \
  --dashboard
```

回测页面展示当前正在执行的候选工作流。默认第一个完整工作流结束后，后续审核使用本地
规则，因此页面不会为每个历史日期都调用 12 个 Agent。如果想观察两个完整工作流：

```bash
quant-trader backtest \
  --config configs/default.yaml \
  --data-root data \
  --output run-dashboard.json \
  --use-llm \
  --llm-provider minimax \
  --llm-workflow trading-agents \
  --llm-max-reviews 2 \
  --dashboard
```

不要在第一次测试时设置很大的 `--llm-max-reviews`，否则模型调用量和运行时间会迅速增加。

### Dashboard 的安全与隐私

- 只监听本机 `127.0.0.1` 的随机端口；
- 每次运行使用新的随机 URL token；
- 只提供固定页面和只读状态接口；
- 页面不显示原始模型输出、完整 Prompt、API Key 或隐藏推理；
- 页面关闭、轮询失败或 Dashboard 内部异常不会改变交易决策；
- Dashboard 不会写入或修改订单。

### 常见问题

#### 浏览器没有自动打开

复制终端中 `Dashboard:` 后面的完整 URL，粘贴到本机浏览器。

#### 页面显示断开，但终端命令还在运行

先看终端是否仍有 `Agent ... started/completed` 输出。Dashboard 是旁路观察功能，页面断开
不会中止 LLM 或改变结果；最终审计仍会写入 JSON。

#### 报错 `--dashboard requires --use-llm and --llm-workflow trading-agents`

回测 Dashboard 必须同时带上：

```bash
--use-llm --llm-workflow trading-agents --dashboard
```

规则回测和普通单审核工作流没有 12 个 Agent 事件，因此不能使用此 Dashboard。

#### 页面很快结束，没有调用模型

查看输出 JSON。如果 `eligible` 是 `false` 且 `provider_calls` 是 `0`，说明该标的在指定日期
没有通过确定性趋势、流动性或历史长度规则。这不是 API 故障。可以先使用文档示例中的
`SPY --as-of 2025-12-31`。

#### 某些角色显示 `skipped`

这是没有传入 `--context` 时的正常行为。市场、多空研究、交易和风险角色仍会继续执行。

#### MiniMax 一直停在某个角色

每个角色都是一次独立同步请求。先等待当前配置的超时和重试完成，同时检查网络、额度、
`MINIMAX_API_KEY`、`MINIMAX_BASE_URL` 与 `MINIMAX_MODEL`。终端会明确显示当前角色名称。

#### 命令结束后刷新页面打不开

这是预期行为。Dashboard 是随命令启动的临时服务，命令结束后自动关闭。持久化结果请查看
`agent-run.json` 或回测的 `run-dashboard.json`。

## 论文策略实验（MVP）

这一版加入三个彼此独立的论文启发式实验，全部使用本地缓存行情和纸面回测，不会真实下单。
输出会写入 `--output-dir` 下带时间戳的独立目录。

### FinMem：三层记忆辅助决策

MiniMax 中国区 M3：

```bash
export MINIMAX_API_KEY="你的 API Key"
quant-trader experiment run finmem \
  --config configs/default.yaml \
  --data-root data \
  --output-dir runs \
  --llm-provider minimax \
  --dashboard
```

使用本地 Codex 登录：

```bash
quant-trader experiment run finmem \
  --config configs/default.yaml \
  --data-root data \
  --output-dir runs \
  --llm-provider codex \
  --dashboard
```

为避免历史回测产生大量请求，FinMem 默认只让第一个审核点调用一次 LLM，其余审核点使用
本地规则回复。Dashboard 展示短期、中期、长期三条记忆泳道、最近动作和引用的记忆 ID。
第一版从空记忆开始，并把可复用的记忆和最后一次决策写入 `finmem/`。

### QuantaAlpha：安全因子 DSL 与一代进化

```bash
quant-trader experiment run quanta-alpha \
  --config configs/default.yaml \
  --data-root data \
  --output-dir runs \
  --llm-provider minimax \
  --dashboard
```

它最多调用两次 LLM：一次批量生成种子因子，一次生成变异或交叉后代。表达式只能使用白名单
字段和函数，由手写解析器执行；不会运行模型生成的 Python。Dashboard 展示候选表达式、父子
关系、拒绝原因和冻结冠军，结果写入 `quanta_alpha/result.json`。

### Alpha Arena：零调用策略排行榜

Arena 只读取已经生成的实验目录，自身不会打开 MiniMax 或 Codex：

```bash
quant-trader experiment run alpha-arena \
  --config configs/default.yaml \
  --data-root data \
  --output-dir runs \
  --contestant-run runs/finmem-具体运行目录 \
  --contestant-run runs/quanta-alpha-具体运行目录 \
  --dashboard
```

缺少的参赛者会显示为 `absent`，损坏或配置不一致的产物只会让对应参赛者失败。Dashboard
按风险违规、最大回撤绝对值、收益的顺序展示排行榜，同时显示成本和净值记录。

### 如何判断模型慢还是程序卡住

终端会输出 `review started/completed`，Dashboard 顶部显示当前阶段与调用数。MiniMax 和 Codex
仍使用配置或客户端的硬超时；失败后命令会输出脱敏错误并保留已经写入的 manifest、事件和阶段
产物。任何回测收益都只是历史纸面结果，不代表未来表现或实盘收益。
