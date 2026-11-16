# 放置容量与模型恢复实现计划

**Goal:** 修复边界放置失败，交付可核验的三物同区和模型恢复闭环。

**Architecture:** 现有本地规划和 SafeArm 负责动作；新增恢复决策协议仅选择本地选项。网页控制器负责最多三次恢复循环，仿真适配器区分空夹爪扫描重试与持物放置续接。

**Tech Stack:** Python、NumPy、OpenCV、现有 L1 模拟器、stdlib HTTP、原在线模型流式接口、原生 JavaScript。

1. `task/reach.py`、`task/runner.py`、`task/skills.py`：先复现三物同区失败；候选放置点避开工作边界；保存最后抓取上下文并提供 `has_held_recovery`、`retry_held_placement` 和保持夹持的放置续接。用现有 task 和新容量回归测试验证。
2. `assistant/recovery.py`、`assistant/llm.py`：新增严格恢复协议与共享流式传输。对版本、动作、选项、取消、错误脱敏做离线测试；保留普通决策兼容性。
3. `assistant/simulation.py`：保存测量目标和邻居；提供 `prepare_recovery`、`execute_recovery`。持物时静止拍摄，空夹爪时完整扫描；复用最终独立落位核验。测试实际放置失败到续接成功。
4. `assistant/server.py`：执行失败后获取上下文、向在线模型发出恢复请求、验证选项、执行、核验并循环。最多三次，停止优先，API失败保留现场。测试最终指标、无进展、非法选项、夹持、取消与故障。
5. `assistant/static/`：显示恢复横幅、编号及流式回复；检测扫描进度；非浏览器脚本检查和 JS 语法验证。
6. 分组执行必要回归，完成真实在线协议联调和三物同区演示；保存脱敏记录，更新中文使用说明，启动并展示最终服务。

执行顺序：前三个独立模块并行开发；适配层和网页控制器按接口顺序接入；最后联调。既有未提交工作保留在当前工作区，不做提交或分支迁移。
