# LLM Simulation UI Implementation Plan

**Goal:** 让用户双击启动可视化 L1 仿真，用现有中转站模型选择并抓取物体，看到模型流式回复。

**Architecture:** 复用现有仿真会话和 TaskRunner；将 API、仿真适配、会话服务与静态界面分开。单任务线程执行，HTTP 线程只读取快照与提交任务，浏览器定时刷新。

**Tech Stack:** Python 3.11、标准库 HTTP/urllib、现有 NumPy/OpenCV、原生浏览器 Canvas。

## 任务

- [x] `assistant/llm.py`：实现当前配置/环境变量加载、Responses/Chat SSE、固定决策格式及本地校验。用伪网络流测试碎片/错误/超时，不在单元测试调用付费服务。
- [x] `assistant/simulation.py`：创建 L1 sim 会话、观察编号、抓取和真值验证。注入可调速时钟，提供真实 FK 与世界对象快照；验证抓取、旧场景、中止。
- [x] `assistant/server.py`：提供 `/api/state`、`/api/command`、`/api/stop`、`/api/reset`、`/api/speed` 与 `/api/camera.jpg`。拒绝并发抓取和跨来源修改，输入限长，响应不含密钥。
- [x] `assistant/static/index.html`、`style.css`、`app.js`：实现机器人画布、物体、观测、任务状态及流式模型回复，窄屏可用。
- [x] 注册 `python -m cube_task assistant_ui`，添加根目录双击启动文件与中文说明，安装包包含静态资源。
- [x] 92项相关测试通过；真实API红块抓取与未观察到蓝块已验证，界面脚本与接口已检查。浏览器外观检查受工具限制，证据与边界见 `local_validation/方案一验收记录.md`。

## 实施约定

用户已明确要求接入和实现，按上述范围直接完成；不增加重复的批准步骤。保留现有文件与未提交工作，不自动提交。API 验证仅发送本任务所需的指令和仿真对象表，不发送项目源码、聊天历史或凭据。
