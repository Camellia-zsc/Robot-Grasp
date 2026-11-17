# Sequential Batch Grasp Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement and review the independent protocol, queue, and UI tasks.

**Goal:** 一条自然语言指令选择多件物品，由现有机械臂顺序抓放。

**Architecture:** 模型给出经过全量校验的计划，控制器的单工作线程逐项执行。独立 batch.py 只负责实测目标绑定与完整足迹判断，控制器负责状态、恢复和取消，网页消费同一队列快照。

**Tech Stack:** 现有 Python、NumPy、HTTP/SSE、原生 HTML/CSS/JS，无新依赖。

- [ ] 协议：在 assistant/llm.py 新增严格 pick_batch，并在 test_assistant_batch_llm.py 覆盖 2–6 项、区域默认值、重复 ID、无效项、只读提示和两种流式 API；旧响应格式保持兼容。
- [ ] 队列：新增 assistant/batch.py，build_items 固定初始实测目标/目的区，bind_item 使用新场景唯一匹配并返回经 validate_decision 复核的单物品决策；whole_footprint_in_zone 根据检测长宽/yaw 与区域边界保守判定。测试新 ID、漂移、歧义、缺失、完整足迹与区域中心不足。
- [ ] 控制器：server.py 增加 batch 快照与逐项执行方法，在现有 submit 中仅新增批量分支；一项一项执行及恢复，逐项更新当前状态。停止与异常清理剩余状态，保留完成项；聚合结果保持现有 metrics 语义。
- [ ] 网页：static/app.js/index.html/style.css 加队列显示，更新示例及帮助。用模拟 DOM 检查顺序/完成数/失败/取消，不把批量初始计划误当当前选择。
- [ ] 真仿真：新增两个水果完整顺序抓放回归，核验 fresh scene、新编号与两个独立落位成功；全套相关单元、单物品和恢复回归通过后提交。
- [ ] 交付：更新使用说明和验证记录，按功能提交、推送 Robot-Grasp，快进同步 D 盘，保留用户当前摆场/已放置结果和聊天记录重启 8767，核对新版资产及 API 配置。

测试入口：在 ws/src/cube_task 中运行 python -m pytest -q test/test_assistant_batch*.py，然后运行已有 test_assistant_llm.py/test_assistant_server.py/test_assistant_recovery_workflow.py 等受影响回归；JS 使用 node --check 与模拟 DOM。
