# 空当接龙 FreeCell

这是一个用 Python 编写的空当接龙游戏项目。项目目前包含可复用的游戏规则模型、Tkinter 图形界面、命令行入口、离线搜索求解器和求解器基准评估工具，可以作为后续开发自动游玩和 AI 训练环境的基础。

## 功能特性

- 使用标准 52 张扑克牌，随机洗牌后分发到 8 列牌堆。
- 支持 4 个空当位和 4 个按花色归档的目标堆。
- 实现基础空当接龙移动规则：
  - 列到列：按点数递减、红黑交替移动到目标列。
  - 列到空当：将列顶部牌移动到空闲空当位。
  - 空当到列：将空当牌移动回合法列。
  - 列到目标堆：按 A 到 K 的顺序归入对应花色目标堆。
- 图形界面支持鼠标拖拽单张牌和合法连续牌组。
- 支持右键自动归堆、`Ctrl+Z` 撤销、重开一局和胜利检测。
- 命令行入口可以展示当前牌局、列出合法动作，并按编号执行动作。
- 离线求解器基于核心模型生成合法动作、执行搜索并返回可重放路径。
- `autoplay.py` 可以按固定 seed 运行求解器并验证成功路径。
- `benchmark.py` 可以批量评估固定 seed 的求解率、节点数、路径长度和耗时。
- 已解牌局可以导出为 JSON trace，并通过 `replay.py` 复现校验。
- `dataset_builder.py` 可以把已解 trace 转换为 JSONL 策略学习样本。
- `policy_baseline.py` 可以在 JSONL 数据集上评估简单策略基线。
- `policy_player.py` 可以让简单策略直接游玩并统计胜率、步数和归堆进度。
- `report.py` 可以汇总求解器、离线策略命中率和直接游玩表现。
- 使用标准库 `unittest` 覆盖核心规则、求解器、自动游玩入口和基准评估工具。

## 环境要求

- Python 3.9 或更高版本。
- Tkinter。通常 Windows/macOS 的官方 Python 安装包会自带 Tkinter。
- 当前项目没有第三方 Python 依赖。

本项目源码使用 UTF-8 编码保存。

## 运行方式

推荐运行图形界面：

```powershell
cd D:\freecell
python gui.py
```

图形界面操作：

- 鼠标左键拖拽牌到目标列、空当位或目标堆。
- 拖拽连续合法牌组时，程序会根据当前空当位和空列数量限制最大可移动张数。
- 在牌桌上右键点击可以自动将当前可归堆的牌移动到目标堆。
- 按 `Ctrl+Z` 撤销上一步操作。
- 点击“重开一局”开始新牌局。

也可以运行命令行版本：

```powershell
cd D:\freecell
python main.py
```

命令行版本会打印当前牌局状态和可选动作，输入动作编号执行对应移动，输入 `q` 退出。

运行离线求解器：

```powershell
cd D:\freecell
python autoplay.py --seed 1 --max-nodes 5000 --max-depth 200
python autoplay.py --seed 1 --max-nodes 5000 --max-depth 200 --save-trace traces\seed_000001.json
```

回放已保存的求解路径：

```powershell
cd D:\freecell
python replay.py traces\seed_000001.json
```

批量评估求解器：

```powershell
cd D:\freecell
python benchmark.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200
python benchmark.py --seeds 1 2 3 --max-nodes 1000 --max-depth 100 --format csv
python benchmark.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --save-solved-traces traces
```

生成策略学习数据集：

```powershell
cd D:\freecell
python dataset_builder.py --trace traces\seed_000001.json --output datasets\one.jsonl
python dataset_builder.py --trace-dir traces --output datasets\freecell_policy.jsonl --skip-invalid
```

评估策略基线：

```powershell
cd D:\freecell
python policy_baseline.py --dataset datasets\freecell_policy.jsonl --policy first_legal
python policy_baseline.py --dataset datasets\freecell_policy.jsonl --policy heuristic
```

直接评估策略游玩表现：

```powershell
cd D:\freecell
python policy_player.py --seed 1 --policy heuristic --max-steps 500
python policy_player.py --seed-start 1 --seed-count 10 --policy heuristic --max-steps 500
```

生成统一评估报告：

```powershell
cd D:\freecell
python report.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --player-policy heuristic --max-steps 500
python report.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --dataset datasets\freecell_policy.jsonl --baseline-policy heuristic --player-policy heuristic --format json
```

运行测试：

```powershell
cd D:\freecell
python -m unittest discover -s tests
```

## 项目结构

```text
freecell/
├── game_model.py   # 牌、移动类型和 FreeCellGame 核心规则模型
├── solver.py       # 离线搜索求解器
├── autoplay.py     # 按 seed 自动求解并验证路径
├── replay.py       # 回放并验证已保存的求解 trace
├── trace_io.py     # JSON trace 读写与验证
├── dataset_builder.py # 从 solved trace 生成 JSONL 样本
├── policy_baseline.py # 在 JSONL 样本上评估简单策略基线
├── policy_player.py # 让策略直接游玩并输出真实表现指标
├── report.py       # 汇总 solver、policy baseline 和 policy player 指标
├── benchmark.py    # 批量求解器评估工具
├── gui.py          # Tkinter 图形界面和交互逻辑
├── main.py         # 命令行游戏入口
├── benchmarks/     # 固定基准 seed 列表
├── tests/          # unittest 测试
├── test.py         # 当前仅用于打印 Python 解释器路径
├── .gitignore      # 忽略编辑器配置和 Python 缓存目录
└── README.md       # 项目说明文档
```

## 核心模块说明

`game_model.py` 定义了项目的核心数据结构和规则：

- `Suit`：扑克牌花色枚举。
- `Card`：单张牌，包含花色、点数、颜色和字符串显示。
- `MoveType`：移动类型枚举。
- `Move`：一次可执行移动的描述。
- `FreeCellGame`：维护 8 列牌堆、4 个空当位和 4 个目标堆，并提供合法移动判断、统一移动执行、合法动作生成、状态克隆、状态哈希、胜利/终止判断和自动归堆能力。

`gui.py` 基于 `FreeCellGame` 实现图形界面：

- 使用 `tkinter.Canvas` 绘制牌桌、空当位、目标堆和牌列。
- 支持拖拽移动牌。
- 支持连续牌组移动，并复用核心模型判断最大可移动长度。
- 通过历史快照实现撤销。
- 检测 52 张牌全部归入目标堆后的胜利状态。

`main.py` 是命令行入口，主要用于直接观察游戏状态、查看当前合法动作并执行移动。

`solver.py` 提供离线搜索求解器：

- `SolveResult`：保存是否解出、动作路径、探索节点数、生成节点数、最大 frontier 和结束原因。
- `solve(game, max_nodes=50000, max_depth=200)`：在不修改输入游戏状态的前提下搜索可重放解法。
- 求解器只通过 `FreeCellGame` 的模型层接口生成和执行动作。

`autoplay.py` 是求解器命令行入口：

- 支持 `--seed`、`--max-nodes`、`--max-depth`。
- 解出时打印路径并在 clone 上重放验证，验证失败会返回非 0 退出码。
- 支持 `--save-trace` 保存已验证成功的解法 JSON。

`trace_io.py` 和 `replay.py` 负责求解路径持久化：

- trace 使用 UTF-8 JSON，记录 seed、搜索参数、结果统计和动作序列。
- `replay.py` 从 trace 重新创建牌局并逐步应用动作，验证失败会返回非 0 退出码。

`benchmark.py` 用于批量评估：

- 支持显式 seed 列表或 seed 区间。
- 支持 `text` 和 `csv` 输出。
- 汇总求解率、平均耗时、平均探索节点数和已解路径平均长度。
- 支持 `--save-solved-traces` 批量保存已解牌局 trace。

`dataset_builder.py` 用于生成后续策略学习数据：

- 从一个或多个 solved trace 生成 JSONL。
- 每条样本包含 action 前状态、合法动作集合、求解路径选择的动作和 action index。
- 默认遇到非法 trace 返回非 0；`--skip-invalid` 可跳过坏 trace。

`policy_baseline.py` 用于评估无训练策略基线：

- `first_legal` 总是选择第一个合法动作。
- `heuristic` 使用手写动作优先级选择动作。
- 输出样本数、命中数和 accuracy，用于后续模型训练的对照基线。

`policy_player.py` 用于评估策略真实游玩表现：

- 每一步从当前游戏状态生成合法动作，再由策略选择动作并通过模型层执行。
- 检测重复状态循环，并支持 `max_steps` 上限。
- 输出胜利数、胜率、平均步数和平均归堆数量。

`report.py` 用于生成统一评估报告：

- 总是汇总 solver benchmark 和 policy player 表现。
- 提供 `--dataset` 时额外汇总 `policy_baseline.py` 的离线 action accuracy。
- 支持 `text` 和 `json` 输出，便于人工查看和脚本消费。

## 当前限制和后续方向

- 图形界面依赖 Tkinter；如果当前 Python 环境没有安装 Tkinter，`gui.py` 无法启动。
- 求解器目前是启发式离线搜索，不保证在给定节点/深度上解出所有牌局。
- 当前还没有 GUI 自动播放控制。
- AI 训练和策略模型尚未实现；当前 JSONL 数据集和策略基线可以作为后续训练输入与对照指标。
- `test.py` 目前只打印 Python 解释器路径，后续可以删除或改为更有用的开发入口。
