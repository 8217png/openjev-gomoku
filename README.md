# OpenJev 五子棋竞技场

基于 [apus-ailab/APUS-OpenJev-v1](https://huggingface.co/apus-ailab/APUS-OpenJev-v1) 的五子棋对战系统：
**JEV（OpenJev 决策模型）** 可以与 **人类 / 大模型（LLM）/ 五子棋算法机器人** 任意组合对战。

## 1. 模型要点（读 HF 模型卡的结论）

- OpenJev-v1 是**决策模型**，不是聊天模型：输入 `state`（上下文）+ `instructions`（任务）+ 2–16 个候选
  （`criteria`），runtime 给候选分配 A–P 标签，只对这些标签打分，返回 `choice` 和候选上的相对概率（未校准）。
- 基于 Qwen3.5，有 4B / 9B / 35B-A3B；本项目部署 **9B-3000**（BF16，官方推荐的质量优先版本）。
- 可选计算深度：`effort=high` 跑满 32 层 + 完整 LM head；`effort=low` 在第 16 层提前退出，只投影候选标签行。
- 官方参考 runtime（`openjet_runtime/`，随权重发布）严格要求 `transformers==5.16.1`、`torch 2.8`。

## 2. 服务架构

| 服务 | 端口 | GPU | 说明 |
|---|---|---|---|
| OpenJev 决策服务 `jev_service/server.py` | 18310 | RTX 5090（约 19 GB） | FastAPI 封装官方 runtime：`POST /decide`、`POST /generate`、`GET /health` |
| 大模型对手（llama.cpp 容器 `gomoku-llm`） | 18300 | RTX 4060 Ti | Qwen3.5-9B Q4_K_M，OpenAI 兼容 `/v1/chat/completions` |
| 五子棋服务 + 前端 `gomoku/app.py` | 38320 | — | 游戏 API + 网页界面 |

一键启动（已在运行的服务会跳过）：`./start_all.sh`，然后打开 **http://localhost:38320/**

首次启动需先 `pip install -r requirements.txt`、下载模型到 `models/APUS-OpenJev-v1/`（或设置 `JEV_MODEL_DIR`），并在 `.env` 中设置 `LLM_GGUF_DIR`（可从 `.env.example` 复制）指向存放 `Qwen3.5-9B-Q4_K_M.gguf` 的目录。

关闭：`./stop_all.sh`；只停某个服务：`./stop_all.sh game|jev|llm`；`--dry-run` 只预览不执行。

### /decide 示例

```bash
curl -s localhost:18310/decide -H 'content-type: application/json' -d '{
  "state": "Order 731 has been delivered. The customer says thank you.",
  "instructions": "Select the appropriate next workflow action.",
  "criteria": [{"id":"close","description":"Close the resolved ticket."},
               {"id":"refund","description":"Refund an undelivered order."}],
  "effort": "high"}'
# -> {"choice":"close","probabilities":{"close":0.999,...},"executed_layers":32,"latency_ms":130,...}
```

官方示例验证：二选一题 low/high 都选 `close`；16 候选浏览器题 high 正确选 `click-12`（99.5%），
low 选错为 `click-14`（17%）——low 明显更弱，与模型卡描述一致。

## 3. 五子棋设计

- 规则：15×15 无禁手，五连或更长获胜，黑先。
- **算法机器人**（`gomoku/engine.py`）：棋型评分 + 增量更新 + alpha-beta 迭代加深。
  难度 easy（带噪声贪心）/ medium（2 层）/ hard（4 层）/ expert（6 层）。
- **JEV 玩家**：OpenJev 只能从 2–16 个候选中选，所以由算法生成前 10 个候选点，
  并为每个候选写出"进攻效果/防守效果"描述，`state` 中给出棋盘 ASCII 图和双方威胁摘要，由 OpenJev 做最终决策。
  候选顺序随机打乱（避免模型直接吃启发式排序），并用 2 次不同顺序的投票取平均概率，抵消模型对某些字母标签的偏好。
- **LLM 玩家**：看到同样的棋盘，自由输出坐标 `MOVE: H8`；非法（被占/无法解析）会带错误提示重试 2 次，
  仍失败则从启发式前 5 中随机兜底并记录（`llm_fallbacks`）。模型地址/名称/API Key 可在界面中改为任意 OpenAI 兼容服务。
- 前端：选择黑白双方类型（人/JEV/LLM/算法）与参数，支持 AI 自动连续对弈、悔棋、单步；
  右侧显示每一步 JEV 的候选概率、执行层数、延迟和选中理由，LLM 的回复与重试情况。

## 4. 测试

```bash
.venv/bin/python -m pytest -q tests                 # 引擎单测（12 项）
.venv/bin/python scripts/ui_check.py                # 无头浏览器走一遍界面（截图在 docs/）
.venv/bin/python scripts/diagnose.py                # 在 60 个局面上检查 JEV 的选择质量
.venv/bin/python scripts/arena.py jev-high algo-medium --games 20
.venv/bin/python scripts/summarize.py results
```

arena 规则：每个随机开局（中心 + 2 子随机）双方各执黑一次；`cand-top1` / `cand-random` 是对照组——
和 JEV 看同一份候选列表，但分别"总选启发式第一"和"随机选"，用来判断模型是否真的在做决策。
"漏堵"= 对手下一步即可成五时没有去堵。

### 结果（当前版本，results/）

| A | B | 局数 | A胜 | B胜 | 平 | A漏堵 | B漏堵 | A 平均ms/步 | B 平均ms/步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cand-random | algo-easy | 20 | 0 | 20 | 0 | 18 | 0 | 0.7 | 0.6 |
| cand-random | algo-hard | 20 | 0 | 20 | 0 | 17 | 0 | 0.6 | 101.6 |
| cand-random | algo-medium | 20 | 0 | 20 | 0 | 16 | 0 | 0.6 | 6.6 |
| cand-top1 | algo-easy | 20 | 8 | 1 | 11 | 0 | 0 | 1.1 | 1.3 |
| cand-top1 | algo-hard | 20 | 3 | 13 | 4 | 0 | 0 | 1.0 | 203.3 |
| cand-top1 | algo-medium | 20 | 7 | 13 | 0 | 0 | 0 | 0.9 | 9.1 |
| cand-top1 | llm | 4 | 4 | 0 | 0 | 0 | 3 | 0.7 | 14358.8 |
| jev-high | algo-easy | 20 | 8 | 8 | 4 | 0 | 0 | 433.0 | 1.6 |
| jev-high | algo-hard | 20 | 2 | 18 | 0 | 0 | 0 | 443.1 | 170.5 |
| jev-high | algo-medium | 20 | 7 | 12 | 1 | 0 | 0 | 436.3 | 12.0 |
| jev-high | llm-think | 2 | 2 | 0 | 0 | 0 | 2 | 456.0 | 191320.5 |
| jev-high | llm | 6 | 6 | 0 | 0 | 0 | 5 | 431.9 | 13516.6 |
| jev-low | algo-easy | 20 | 1 | 19 | 0 | 14 | 0 | 238.4 | 0.9 |

### 结论

- **JEV vs 大模型**：JEV-high 全胜（非思考 6:0，思考模式 2:0；思考模式 LLM 每步约 3 分钟）。Qwen3.5-9B 自由下棋时经常下到已占位置（多次重试、兜底）且频繁漏堵。
- **JEV vs 算法**：JEV-high 能稳定处理必胜/必堵（0 漏堵），与 easy 持平，对 medium 约 35% 胜率，对 hard 明显不敌。
  整体强度与"总选启发式第一"的对照组相当，远强于随机选择——说明模型确实在读候选描述做判断，
  但并没有超越提供候选的启发式本身。典型失误是在对方有活三时选择自己做活三（优先级判断错误）。
- **effort=low 明显更弱**（大量漏堵），五子棋场景应使用 high。
- 提示设计影响巨大：初版（候选只有简单标签、无威胁摘要、单次询问）JEV-high 对 easy 仅 2:18、对 medium 0:20，
  旧结果保存在 `results/v1_initial_prompt/`。
- 延迟：JEV-high 每步约 430 ms（2 次投票 × 约 200 ms，约 900 prompt tokens，参考 runtime 单请求串行）；LLM 每步约 13 s。

## 5. 目录

```
jev_service/server.py     OpenJev 推理服务
gomoku/engine.py          棋盘、棋型、算法机器人、候选描述、威胁摘要
gomoku/players.py         JEV / LLM / 算法 玩家适配
gomoku/app.py, static/    游戏服务与前端
scripts/                  arena 对战、诊断、UI 检查、结果汇总
models/APUS-OpenJev-v1/   9B-3000 权重（含官方 runtime）
```

## 许可证

本项目代码采用 [MIT License](LICENSE)。OpenJev 模型权重与官方 runtime（`openjet_runtime/`）来自
[apus-ailab/APUS-OpenJev-v1](https://huggingface.co/apus-ailab/APUS-OpenJev-v1)，不包含在本仓库中，适用其各自的许可证（MIT / Apache 2.0）。
