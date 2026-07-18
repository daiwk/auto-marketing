# FinMem、QuantaAlpha 与 Alpha Arena MVP 设计

## 目标

在现有 `quant-trader` 纸上交易框架中新增三个可独立运行的论文启发式 MVP：

1. FinMem：用分层记忆、画像和反思辅助交易决策；
2. QuantaAlpha：用受限因子 DSL 完成一代、小规模的 LLM 因子进化；
3. Alpha Arena：在相同数据、成本和风控条件下比较规则策略、TradingAgents、FinMem
   与冻结的 QuantaAlpha 冠军因子。

三个模块共享实验调度、行情缓存、回测、风控、事件记录和浏览器 Dashboard，但不串成
一个不可拆分的大流程。所有功能均为离线回测和纸上交易，不接券商或真实下单。

本设计提取 [FinMem](https://arxiv.org/abs/2311.13743) 的 Profiling、分层 Memory 和
Decision-making，以及 [QuantaAlpha](https://arxiv.org/abs/2602.07085) 的因子生成、
轨迹级 mutation/crossover、语义一致性、复杂度和冗余约束。Alpha Arena 是对公开竞赛
思路的本地、安全适配：使用缓存的美股日线和模拟资金，不复现加密货币永续合约实盘。

## 用户入口

新增一个统一实验命令，并保留三个模块的独立性：

```bash
quant-trader experiment run finmem --config configs/default.yaml --data-root data --output-dir runs/demo --dashboard
quant-trader experiment run quanta-alpha --config configs/default.yaml --data-root data --output-dir runs/demo --dashboard
quant-trader experiment run alpha-arena --config configs/default.yaml --data-root data --output-dir runs/demo --dashboard
```

LLM provider 继续复用现有 MiniMax 中国区 M3 和本地 Codex 配置。没有显式开启 LLM、
缺少 provider 配置或数据不完整时，命令必须快速失败并给出可执行的提示，不进行隐式联网下载。

## 总体架构

新增 `quant_trader.experiments` 边界，包含以下职责单一的组件：

- `ExperimentRunner`：运行生命周期、进度、取消、LLM 调用预算和产物目录；
- `ExperimentEvent`：所有策略共用的有界、可序列化事件；
- `ArtifactStore`：原子写入 manifest、事件流、统一摘要及模块专属产物；
- `ExperimentStrategy`：三个模块遵守的最小接口，接收不可变上下文并返回统一结果；
- Dashboard projection：把事件投影成统一驾驶舱状态，再渲染模块专属区域。

论文模块只能生成候选动作或候选因子。已有的确定性回测、费用模型和风控层拥有最终决定权；
LLM 不能扩大候选仓位、绕过风险限制或执行任意生成代码。

## FinMem MVP

### 数据流

1. 从当前日期以前的行情特征、持仓和已完成交易构造市场快照；
2. 按短期、中期、长期三层检索记忆，每层使用独立的时间衰减和固定 Top-K；
3. 将 agent profile、快照和带稳定 ID 的检索结果交给 LLM；
4. LLM 只返回结构化动作、置信度、引用的记忆 ID 和简短理由；
5. 统一风控验证动作，并由纸上交易回测器模拟成交；
6. 平仓或触发明显回撤事件后，可调用一次反思并将有界摘要写入记忆。

记忆记录包含事件日期、可用日期、层级、类别、摘要、重要度、关联 ticker 和来源 ID。
检索严格使用决策时点之前已可用的记录，避免未来数据泄漏。层级晋升和淘汰使用确定性规则，
LLM 不能直接修改重要度或删除历史记录。

### 调用预算

- 每个被 review cap 选中的决策点最多一次决策调用；
- 仅在平仓或风险事件后最多一次反思调用；
- 跳过的日期和不合格候选不消耗调用；
- 所有尝试，包括重试，均计入预算并显示在 Dashboard。

### 可视化

FinMem 专属区域显示三层记忆泳道、各层容量和衰减、被检索条目、证据到动作的引用关系、
风控是否改写动作，以及反思产生或晋升了哪些记忆。页面只显示有界摘要，不显示隐藏思维链、
原始 provider 响应或密钥。

## QuantaAlpha MVP

### 因子 DSL

因子只能由白名单字段和操作符组成。第一版字段限于开高低收、成交量及由当前数据计算的收益；
操作符限于基础算术、`delay`、`delta`、`rank`、`rolling_mean`、`rolling_std`、`rolling_min`、
`rolling_max` 和 `zscore`。解析器必须验证 AST 节点类型、字段、参数范围、窗口上限、树深度和
总节点数。禁止属性访问、导入、函数定义、循环、文件/网络访问以及任意 Python `eval/exec`。

### 数据流

1. 一次批量 LLM 调用生成最多四个“假设 + DSL 表达式”种子；
2. 依次执行语法、安全、复杂度、有限值和最低覆盖率校验；
3. 在按时间顺序划分的训练集和验证集计算 IC、换手、收益、回撤及与已有候选的相关性；
4. 将有界指标和失败步骤交给第二次批量调用，生成最多四个变异或交叉后代；
5. 后代经过相同校验，只按预先定义的验证评分选出一个冠军；
6. 冻结冠军后在从未参与筛选的测试集计算最终指标，不再调参。

整个实验默认恰好零到两次 LLM 调用：生成阶段一次、进化阶段一次。若所有种子都在安全校验
前失败，则不调用进化阶段并返回部分完成结果。表达式去重使用规范化 AST；冗余过滤使用验证集
相关性阈值，不能读取测试集。

### 可视化

QuantaAlpha 专属区域显示可展开的因子家谱、父子和 mutation/crossover 关系、每道校验门、
拒绝原因、表达式复杂度，以及训练/验证/测试指标的明确分栏。测试指标只能在冠军冻结后出现，
避免界面诱导用户基于测试结果继续筛选。

## Alpha Arena MVP

Arena 的参赛者是策略而不是 provider。默认参赛者为：

- 现有确定性规则策略；
- 现有 TradingAgents；
- FinMem；
- QuantaAlpha 已冻结的冠军因子。

所有参赛者使用同一缓存行情、股票池、日期、初始资金、再平衡日、手续费、滑点和仓位上限。
Arena 自身不调用 LLM；它可消费已存在的 run artifact，也可顺序启动缺失参赛者并遵守各自的
review cap。某个参赛者失败或缺席只影响自身状态，不阻塞其他参赛者。

排行榜同时展示累计收益、最大回撤、Sharpe、换手、费用拖累和风险违规数，不只按收益排序。
默认排名键依次为：风险违规数升序、最大回撤绝对值升序、累计收益降序；完整原始指标仍全部展示。

Arena 专属区域显示动态排行榜、叠加净值曲线、动作和置信度分布、费用拖累及风险事件时间点。
每个参赛者都能下钻到对应策略的既有事件和理由，但 Arena 不重新生成理由。

## 统一驾驶舱

Dashboard 延续现有 loopback-only、随机 capability token 和安全文本渲染机制，扩展为“统一
实验驾驶舱”：

1. 顶部显示实验类型、ticker/股票池、数据区间、provider、运行状态、进度和调用预算；
2. 左侧显示阶段时间线，状态包括等待、运行、完成、跳过、失败和取消；
3. 右侧显示当前动作、置信度、最近理由、风控结果、已用调用和等待时长；
4. 底部按实验类型渲染 FinMem 记忆泳道、QuantaAlpha 因子树或 Arena 排行榜；
5. 页面提供安全取消按钮；取消只设置 runner 的协作式停止标志，不终止或修改已落盘结果。

服务端只新增一个带 capability token 的幂等 `POST` 取消端点，不接受任意命令、路径或配置；
其余端点保持只读。取消请求只影响当前 run，已开始的同步 provider 请求仍由硬超时负责结束。

事件 payload 仅允许 schema 中定义的结构化字段，必须限制字符串、数组和树的大小。前端所有动态
内容使用 `textContent` 或等价安全方式，禁止把 LLM 内容注入 HTML。

## 运行状态、超时和失败处理

实验状态为 `pending`、`running`、`partial`、`completed`、`failed` 或 `cancelled`。单次 LLM
默认硬超时 120 秒。只有限流和 provider 5xx 可自动重试一次；超时、无效结构、安全校验失败及
配置错误不自动重试。每次尝试都计入预算。

Runner 在调用前发出带开始时间和硬截止时间的 started 事件，结束后立即发出 completed、failed
或 timed_out 事件。Dashboard 根据这两个时间戳显示实时等待时长，CLI 使用有界进度状态，因此
无需向持久事件流反复写入心跳，也能区分“模型较慢”和“程序卡死”。

产物按阶段原子写入。单个阶段失败时保留已经验证的结果并返回 `partial`；只有无法加载配置、
行情或创建产物目录等前置条件失败时，整个实验为 `failed`。Arena 对参赛者执行故障隔离。

## 产物格式

每次运行写入用户指定的 `<output-dir>/<run-id>/`，至少包含：

- `manifest.json`：实验配置、代码版本、数据区间、数据指纹、provider/model 和预算；
- `events.jsonl`：按序号排列的有界事件，可用于 Dashboard 重放；
- `summary.json`：统一状态、指标、费用、风险和错误分类；
- `finmem/`：记忆快照、检索记录、决策和反思；
- `quanta_alpha/`：候选、规范化表达式、家谱、校验结果和冻结冠军；
- `alpha_arena/`：参赛者引用、排行榜和逐日净值。

只创建当前实验需要的模块目录。API Key、完整环境变量、原始 prompt、原始 provider 响应、隐藏
思维链和未脱敏异常信息不得写入产物。运行产物默认不加入代码库。

## 测试与验收

单元测试覆盖记忆时间边界、衰减、检索、晋升和容量；DSL AST 白名单、复杂度、窗口、异常值、
规范化和去重；时序切分、验证评分、冠军冻结；Arena 统一成本、排名和故障隔离。

provider 使用模拟响应覆盖成功、限流、5xx、超时、非法 JSON 和预算耗尽。集成测试使用仓库内的
小型固定行情数据，分别跑通三个命令，再验证 Arena 能消费独立产物。Dashboard 测试覆盖事件
顺序、断线重连、失败/取消状态、三种专属视图和不安全文本。CI 不访问真实 MiniMax、Codex 或
行情网络。

验收标准是：

1. 三个实验可独立运行，且同一输入和模拟 provider 输出可复现；
2. 不读取决策时点或训练阶段不可见的数据；
3. 任意非法因子不能执行，LLM 决策不能绕过统一风控；
4. 调用数、等待状态、失败原因和中间决策在 CLI 与 Dashboard 中清晰可见；
5. 部分失败保留已验证产物，Arena 不因单一参赛者失败而整体失败；
6. 现有测试、类型检查和 lint 保持通过。

回测是否盈利不属于验收条件，也不能被描述为实盘收益保证。

## 非目标

- 不接真实券商、交易所、钱包或实盘资金；
- 不完整复现论文数据集、训练规模或全部参数；
- 不实现多代、大种群、分布式因子搜索；
- 不实现新闻抓取、向量数据库或外部长期记忆服务；
- 不比较 MiniMax 与 Codex 的模型排名；
- 不根据测试集结果继续调参；
- 不新增前端框架、远程 Dashboard、账户系统或多人协作。
