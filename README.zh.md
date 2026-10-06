# python-agent -- GOSIM survey26 黑客松参赛智能体（participant-agent-protocol-v4）

参赛队：Esther（Jingmeng）、Xinran Wang。基于主办方示例项目
（`gosimfoundation/hackathon-survey26` 的 `examples/python-agent`，CC BY-NC 4.0）改写；
下文描述的调度器、学习模型和大模型环节是我们自己的工作。快速故障探测和指向偏移估计借鉴了
主办方 `python-pro` 参考智能体（同一许可证）的思路，在这里重新实现。

智能体是一个指向 / 光纤 / 曝光联合规划器：按「单位时间期望得分」给候选视场排序，
从自己的观测结果里学习任务卡没有写明的东西，并在输入是自然语言的地方向大模型提几个
有界的问题。模型的回答在改变任何决策之前都会被规则夹紧；required 目标、曝光和故障报告
从不单独依赖模型。

运行时从自己的命中里学到的东西（从不读文件）：

- 天空质量刻度，以及天空实际所在的程序波段；
- **地形**：每个公告的遮挡方向是一个高度、宽度都未知的扇区；一张假设网格由附近每一次
  零分和有分命中加权，指向按「无遮挡的概率」估值（`state.Ridge`）；
- **仪器故障（快）**：故障是只有仪器的台阶式下跌，所以拿本夜最近三次曝光的干净质量比和之前六次比较；
  掉到 55% 以下、且天空波段没有一起下降（天气会压低波段，故障不会），立即用一次免费报告——每两次正确
  报告之间最多一次，地震公告出现后 12 小时内不探测；
- **仪器故障（慢）**：逐夜的干净质量比删失中位数，用程序波段揭示的天空水平修正，再与典型的平静水平比较，
  含地震恢复曲线；它负责较温和的故障和所有付费报告；
- **指向偏移**（Hard mode 卡）：每个命中或落空的目标都会收窄候选偏移网格；当某个偏移明显比零偏移更能解释
  结果时，就按它修正指令，让视场落在计划的位置；
- **节奏**：每次决策的实测成本和引擎自身的每步耗时，使搜索深度和可选的模型调用同时满足公平时钟和平台报告的
  真实时间上限（`wallclock.wall_remaining_seconds`）。

`observer.project.json` 里的开关：`AGENT_SCORE_GAIN=1` 用「预测分减已拿到的最高分」给目标估值；
`AGENT_CONDITIONS=off` 保留学习到的天空刻度（可替换的拟合天气模型在留一卡验证中为负）。
`LLM_DISABLED=1` 给出确定性的无模型运行，用于对比。

[English README](README.md)

## 目录结构

```
agent.py                 入口:stdin/stdout 主循环,按消息类型分支处理
agent_core/
  protocol.py             传输层:逐行读一个 JSON 对象,逐行写一个 JSON 对象
  state.py                 SurveyState:星表 + 跨决策需要跟踪的一切状态
  geometry.py              公开天球几何公式:恒星时、地平坐标、16 根光纤网格
  scoring.py               只用公开评分配置估算 factor/score
  planner.py               决策逻辑:wait / observe / report / finish
  llm_client.py             OpenAI 兼容 chat 客户端,默认对接 Kimi Coding Plan
  memory.py                 可选的、尽力而为的 JSONL 决策轨迹日志(默认关闭)
  validation.py             协议合法性校验 + 确定性兜底动作
observer.project.json      平台项目清单(镶像、运行命令、环境变量)
requirements.txt           无需任何第三方包——只用标准库
.env.example               复制为 .env,运行前先配好 API key
pack_agent.py              把本目录打包成可提交的 ZIP
```

## 为什么这样拆模块

- **protocol.py** —— 把线上协议格式(逐行 JSON、stdout 只给协议用、日志只走 stderr)
  隔离出来,其余代码完全不用操心这件事。
- **state.py** —— 智能体当前所知一切的唯一来源:`initialize` 给的星表(按并行数组
  存储,每个目标一个下标)、按赤纬分带的空间索引(快速查"这附近有什么")、每个
  目标的 `factor`/`misses`/`attempts`、学到的天空 `scale`(从自己命中反推出的质量
  样本的中位数——隐藏的仪器/透明度/天空/视宁度项绝不从任何文件读取)、天气通知、
  故障诊断历史。只在 `initialize` 建一次,每次 `decision_request` 更新一次。还保留
  一份按曝光动作编号记录的精确 factor 账本,这样 Hard mode 的 `state_resync` 可以
  精确回答——丢弃失效动作编号区间内的账本条目,再用剩下的重新算出每个目标的最佳
  factor——而不只是依赖 resync 消息里的 `best_scores`(细节见 `_resync` 的注释)。
- **geometry.py** —— 来自参赛者指南 Geometry 与 Scoring 小节的公开公式:恒星时、
  地平坐标转换、光纤网格判定、月光因子。任何智能体都需要这些,且与具体场景
  数据无关。
- **scoring.py** —— `initialize.payload.scoring` 中**公开**的部分(`q0`、
  `flux_zero_point`、`exposure_zero_point_seconds`、`airmass_exponent`、
  `lunar_model`、`program.{bands,multipliers,mismatch_multiplier}`、`required.*`、
  `uniformity.*`)转换成 factor/波段/倍数估算。
- **planner.py** —— 锚点搜索策略:给可见、尚未完成的目标排序(必做且尚未过公开
  factor 门槛的目标拿到一个大加成;快要落下或剩余夜数不多的目标排名更高);对排
  名靠前的几个候选,逐一尝试把每根光纤当作指向中心,用能落在玻璃上的最有价值邻居
  填满其余光纤,保留最好的指向;选能带来最佳期望得分/秒的曝光时长,以及最多目标
  会匹配上的项目。白天整段 `wait` 过去;公告里出现全天区 rain/storm 就关闸;只有在
  质量出现持续、无法解释的下降并在多个夜晚得到确认后,才上报可疑的仪器故障。限时
  观测请求(`active_requests`)在曝光时长搜索里只拿到一次只读的"选边"机会:在与
  最佳得分/秒速率相差不大的几个候选时长里,优先选能让某个仍需完成的目标越过其
  `completion_factor_threshold` 的那个——绝不会为此改变指向本身,也不会去追一个本
  来就不会被曝光覆盖的目标(请求过期本身不扣分,所以这里只是顺手,不是去抢)。
- **llm_client.py** —— OpenAI 兼容客户端(纯 `urllib`,不依赖任何 SDK),默认对接
  Kimi Coding Plan 接口,改环境变量即可换成任何其它 OpenAI 兼容 `/chat/completions`
  接口。
- **memory.py** —— 可选的、尽力而为的 JSONL 决策轨迹日志(今晚建议、结束汇总),
  默认关闭。state/planner 真正跨决策携带的记忆(学到的 scale、每个目标的进度、
  今晚建议、report 冷却)放在 `SurveyState`/`Planner` 自己身上,因为那本就是它们
  各自该跟踪的状态。
- **validation.py** —— 发往 stdout 前的最后一道防线:检查协议真正会校验的每一项
  (数值范围、光纤/目标重复使用、未知字段、连续 report 次数上限),并提供
  `fallback_action()`——无论智能体处于什么状态,这个回复永远合法。

## LLM 用在哪里

所有调用都经 `LLMClient.ask_json`，回答是一个 JSON 对象，规划器先清洗再使用。只有两类调用，
都不在逐次决策的路径上：

1. **读预报和值班日志**（`Planner._on_forecast`）。周预报到达时，请模型给出它覆盖的每一夜
   （以及之后最多七夜）的可用程度，依据是预报公告、固定表的估计，以及 `experience`：本次运行里
   各类公告之下的夜实际值多少。期间到达的任何长文本字段——例如观测请求的 `reason`，在某些卡上
   是观测员之间的非正式交接记录，多语言、含暗号、简单密码和更正——作为 `duty_log` 一并交给模型，
   模型还可以从中返回 `terrain`：某个罗盘方向的遮挡有多高。边界：仅凭记录判定差的夜仍按四分之一
   的价值规划；地形高度只是软先验，山脊模型自己的证据会覆盖它；模型漏掉的夜保留规则值。
2. **故障仲裁**（`Planner._arbitrate`）。规则检测器在边界证据上达到报告阈值时，模型看到最近几夜的
   比值、天空水平和公告，可以否决报告、再等一夜（每次告警最多五次）。它不能触发报告；强证据路径
   不会询问模型。

当模型预算、公平时钟或推算的真实墙钟（含引擎耗时）不允许时，调用被跳过，由规则回答。

### 调用行为

不管模型做什么，`LLMClient` 都保证：

- 单次超时 30 s，墙钟吃紧时同步缩短（永不侵占巡天预算的最后 60 s）；
- 每次运行的模型总时间预算 500 s（按卡的时限缩放）；
- 每个问题只尝试一次；所有异常（网络、超时、坏 JSON、缺字段）都被捕获，`ask_json` 返回 `None`，
  规划器视为「没有答案，用规则」；
- 密钥只从环境变量读取、从不写日志；HTTP `User-Agent` 由清单设置。

## 配置(.env)

把 `.env.example` 复制为 `.env`,运行前先配好一个 API key:

```
OPENAI_API_KEY=sk-...
```

这样就够了:客户端默认对接
[Kimi Coding Plan](https://www.kimi.com/code/docs/en/) 接口
`https://api.kimi.com/coding/v1`,模型用 `k3`。非中国大陆账号应改用
`https://api.kimi.ai/coding/v1`:

```
OPENAI_BASE_URL=https://api.kimi.ai/coding/v1
OPENAI_API_KEY=sk-...
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` 会覆盖上面的默认值,所以任何其它 OpenAI 兼容的
`/chat/completions` 接口(OpenAI 本身、本地代理等)改两个变量就能用。key 也可以用
`KIMI_API_KEY` 这个别名来设。在平台上,`OPENAI_BASE_URL` / `OPENAI_API_KEY` 会被
自动注入(平台自己的模型代理和一次性临时凭证),提交时完全不需要自己配置,`.env`
也永远不会被打进提交 ZIP。

没配 API key(`OPENAI_API_KEY` 或 `KIMI_API_KEY`)时,进程会在启动时、读任何 stdin
之前就检查这一点,然后把错误信息打到 stderr,以非零退出码退出。

本示例调用的是 OpenAI 兼容的 `/chat/completions` 接口。如果你的队伍在工作区的
**模型 API** 设置里改选了 Anthropic Messages 协议,平台同样会注入
`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`(Anthropic SDK 自己的环境变量名);如果你
自己的智能体改用 Claude 的 Messages API,配上官方 `anthropic` SDK 使用这两个变量即可。
完整对比见 `docs/model-api-keys.md`。

## 本地运行

本项目只是智能体一侧——不含模拟器/评分器。请用任何支持该协议(stdin/stdout)的
本地运行器,或平台本身,把 `--agent` 指向本目录的 `agent.py`。完整协议与评分公式
见 `docs/`。

`agent.py` 在 stdin 上读取 `initialize` / `decision_request` / `finish`,在 stdout 上
写 `decision_response`;其余内容(日志)全部走 stderr,与协议要求完全一致。

## 打包 / 提交

```bash
python3 pack_agent.py --out ../python-agent.zip
```

会把本项目打包,`observer.project.json` 置于 ZIP 根目录
(`"protocol": "jsonl-v4"`、`run: ["python3", "-u", "agent.py"]`),并跳过 `.env`、
`__pycache__`、`run_output/`。把 ZIP 作为完整项目上传,或把本目录推到 GitHub 仓库。
不需要任何第三方包;如果你加了依赖,请同时写进 `requirements.txt` **并**在
`observer.project.json` 里加一个匹配的 `build` 步骤——否则依赖不会被自动安装。

## 安全属性

- 从不读取任何文件:所有状态都来自 stdin。
- `agent.py` 里每个分支都包在 `try`/`except` 中;planner 出 bug 只会得到一个安全的
  `wait`,绝不会崩溃或变成 `agent_error`。
- `validate_action()` 会在写入 stdout 之前,剥掉一切协议会拒绝的内容。
- 一次规划调用持续失败,只会让那个问题的答案这一晚不计入合并;其余决策流程
  (选目标、算曝光、校验)照常运行。

## 许可 / 引用

任务卡、模拟数据、评测代码与本示例项目按 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
提供（署名、非商业）；使用请引用 GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/)。选手自己编写的代码不受此限制。
详见 `LICENSE.md`。
