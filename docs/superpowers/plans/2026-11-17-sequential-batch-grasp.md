# Sequential Batch Grasp Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement and review the independent protocol, queue, and UI tasks.

**Goal:** 一条自然语言指令选择多件物品，由现有机械臂顺序抓放。

**Architecture:** 模型给出经过全量校验的计划，控制器的单工作线程逐项执行。独立 batch.py 只负责实测目标绑定与完整足迹判断，控制器负责状态、恢复和取消，网页消费同一队列快照。

**Tech Stack:** 现有 Python、NumPy、HTTP/SSE、原生 HTML/CSS/JS，无新依赖。

- [x] 协议：在 assistant/llm.py 新增严格 pick_batch，并在 test_assistant_batch_llm.py 覆盖 2–6 项、区域默认值、重复 ID、无效项、只读提示和两种流式 API；旧响应格式保持兼容。
- [x] 队列：新增 assistant/batch.py，build_items 固定初始实测目标/目的区，bind_item 使用新场景唯一匹配并返回经 validate_decision 复核的单物品决策；whole_footprint_in_zone 根据检测长宽/yaw 与区域边界保守判定。测试新 ID、漂移、歧义、缺失、完整足迹与区域中心不足。
- [x] 控制器：server.py 增加 batch 快照与逐项执行方法，在现有 submit 中仅新增批量分支；一项一项执行及恢复，逐项更新当前状态。停止与异常清理剩余状态，保留完成项；聚合结果保持现有 metrics 语义。
- [x] 网页：static/app.js/index.html/style.css 加队列显示，更新示例及帮助。Node 模拟 DOM 50 项断言覆盖顺序/完成数/失败/取消，不把批量初始计划误当当前选择；入口 test/test_assistant_batch_ui.cjs。真实浏览器交互尚未验收。
- [x] 真仿真：两组实际控制器顺序抓放通过，覆盖苹果→红/梨→绿与两件均放红区，核验 fresh scene、新编号、两个独立落位成功和先完成物品不受后续动作影响。模型计划由本地固定客户端提供。
- [x] 在线清单：真实 ModelClient 使用现场 scene_version=32 返回苹果→绿、梨→红的严格顺序清单，一次 API、6.31 秒、70 个流式片段，未调用仿真动作。证据 local_validation/batch_online_plan_check.json。尚未把在线模型和生产界面机械运动放在同一条任务中完成端到端联调。
- [x] 最终综合回归：八文件协议/控制器/身份/恢复组合 242 passed in 5.51s；两组真实批量仿真 2 passed in 121.16s；原单件与恢复仿真 10 passed in 656.84s。共 254 项 Python 测试，另有 50 项 DOM 断言；JS 语法、Python 编译与差异检查通过。
- [x] 使用说明与验收文档：加入多目标、全部、先后顺序示例，明确一次正常模型计划请求、逐件新观测、每件最多三次恢复、失败停队列、运动学/编辑限制及验证边界。
- [x] 交付：协议、队列控制和网页分功能提交并推送 Robot-Grasp main，源码 f818775 已快进同步 D 盘并重启 8767。HTTP 核对 ready、batch idle、在线模型已配置、6 件物品全部变换/区域一致、12 条聊天记录和原有计数保留、四份页面资源与本地一致、实时相机正常。证据 local_validation/batch_deployment_check.json。

测试入口：在 ws/src/cube_task 中运行 python -m pytest -q test/test_assistant_batch*.py，然后运行已有 test_assistant_llm.py/test_assistant_server.py/test_assistant_recovery_workflow.py 等受影响回归；JS 使用 node --check 与模拟 DOM。
