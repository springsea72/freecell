# FreeCell

这是一个使用 Python 标准库实现的 FreeCell 项目。当前仓库包含统一规则模型、命令行入口、Tkinter GUI、离线启发式求解器、trace 回放、benchmark、JSONL 数据集构建、策略基线、直接策略游玩评估和统一报告工具。

## 功能概览

- `game_model.py` 是唯一规则来源，提供发牌、合法动作生成、动作执行、状态哈希、胜利/终止判断和自动归堆。
- `main.py` 提供命令行游戏入口，列出当前合法动作并通过 `FreeCellGame.apply_move()` 执行。
- `gui.py` 提供 Tkinter 图形界面，支持拖拽、右键归堆、撤销、后台求解并播放、暂停/继续/停止、播放速度调整、保存 trace、加载 trace 播放。
- `solver.py` 提供启发式离线搜索，返回可重放的 `SolveResult`，不修改传入的 game。
- `autoplay.py` 可以按 seed 运行 solver、验证解法并可选保存 solved trace。
- `trace_io.py` 和 `replay.py` 负责 solved trace 的 JSON 读写、验证和命令行回放。
- `benchmark.py` 批量评估固定 seed 的求解表现，可输出 text/csv，并可选保存已解 trace。
- `dataset_builder.py` 将 solved trace 转换成 JSONL 策略学习样本。
- `policy_baseline.py` 在 JSONL 样本上评估 `first_legal` 和 `heuristic` 基线。
- `policy_player.py` 让策略直接在 `FreeCellGame(seed=...)` 上游玩，统计胜负、步数和归堆进度。
- `report.py` 汇总 solver benchmark、JSONL policy baseline accuracy 和 policy player 游玩表现。
- 测试使用标准库 `unittest`，不需要 pytest。

## 环境要求

- Python 3.9 或更高版本。
- Tkinter。Windows/macOS 官方 Python 通常自带 Tkinter。
- 当前项目没有第三方 Python 依赖。

## 运行命令

启动 GUI：

```powershell
python gui.py
```

GUI 支持：

- 拖拽单牌或合法连续牌组。
- 右键自动归堆。
- `Ctrl+Z` 撤销。
- “求解播放”在后台求解当前牌局，找到路径后逐步播放。
- “暂停/继续”和“停止播放”控制自动播放。
- 播放中禁用手动拖拽和右键归堆，避免破坏当前 solver path。
- 速度滑块调整自动播放间隔。
- 状态栏显示求解参数和结果统计，包括 explored/generated/frontier/path length。
- “保存 trace”只在当前解法可从 seed 复现并验证通过时可用。
- “加载 trace 播放”会先验证 trace，再用 trace seed 重建牌局并播放。

启动命令行游戏：

```powershell
python main.py
```

运行离线求解：

```powershell
python autoplay.py --seed 1 --max-nodes 5000 --max-depth 200
python autoplay.py --seed 1 --max-nodes 5000 --max-depth 200 --save-trace traces\seed_000001.json
```

回放 trace：

```powershell
python replay.py traces\seed_000001.json
```

批量 benchmark：

```powershell
python benchmark.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200
python benchmark.py --seeds 1 2 3 --max-nodes 1000 --max-depth 100 --format csv
python benchmark.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --save-solved-traces traces
```

构建 JSONL 数据集：

```powershell
python dataset_builder.py --trace traces\seed_000001.json --output datasets\one.jsonl
python dataset_builder.py --trace-dir traces --output datasets\freecell_policy.jsonl --skip-invalid
```

评估 JSONL policy baseline：

```powershell
python policy_baseline.py --dataset datasets\freecell_policy.jsonl --policy first_legal
python policy_baseline.py --dataset datasets\freecell_policy.jsonl --policy heuristic
```

直接评估策略游玩表现：

```powershell
python policy_player.py --seed 1 --policy heuristic --max-steps 500
python policy_player.py --seed-start 1 --seed-count 10 --policy heuristic --max-steps 500
```

生成统一评估报告：

```powershell
python report.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --player-policy heuristic --max-steps 500
python report.py --seed-start 1 --seed-count 10 --max-nodes 5000 --max-depth 200 --dataset datasets\freecell_policy.jsonl --baseline-policy heuristic --player-policy heuristic --format json
```

运行测试：

```powershell
python -m unittest discover -s tests
```

## 项目结构

```text
freecell/
├── game_model.py          # 核心牌、动作和 FreeCellGame 规则模型
├── solver.py              # 启发式离线搜索求解器
├── autoplay.py            # 按 seed 求解、验证和可选保存 trace
├── replay.py              # 回放并验证 solved trace
├── trace_io.py            # JSON trace 读写和验证
├── benchmark.py           # 批量求解器评估
├── dataset_builder.py     # solved trace 转 JSONL 样本
├── policy_baseline.py     # JSONL 上的 first_legal / heuristic accuracy
├── policy_player.py       # 策略直接游玩评估
├── report.py              # 汇总 solver、policy baseline、policy player 指标
├── gui.py                 # Tkinter GUI 和自动求解播放/trace 回放
├── main.py                # 命令行游戏入口
├── benchmarks/
│   └── seeds.txt          # 固定 benchmark seed 列表
├── tests/                 # unittest 测试
├── .gitignore             # 本地缓存、环境和生成产物忽略规则
└── README.md
```

## 当前限制

- solver 是启发式搜索，有节点和深度上限，不保证解出所有牌局。
- GUI 自动播放只接入 solver 路径和 solved trace 回放，未接入 policy player。
- AI 训练、神经网络和策略模型尚未实现。
- JSONL 数据集构建和 policy baseline 只是后续训练的基础设施，不执行训练。
- `traces/`、`datasets/`、`reports/`、临时 `_trace_*.json`、`_dataset_*.jsonl` 和 `_gui_trace_*.json` 属于本地生成产物，默认不会提交。
