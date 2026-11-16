# 多物品摆场与反馈抓取实施计划

**目标：** 在现有网页中摆放并识别抓取苹果、香蕉、梨及三色方块，保留运行反馈和恢复。

**架构：** 独立目录库和合成训练模块，复用 L1 世界、相机、通用 TaskRunner 与反馈技能。后端串行编辑并使观测失效，前端使用独立俯视编辑器，不与 3D 旋转交互争用。

**技术：** Python、NumPy/OpenCV、现有 KNN/点云截面规划、原生 HTML/CSS/Canvas，无新模型下载和前端依赖。

## 1. 物品库与识别（独立模块）

- [x] 新建 assistant/catalog.py：六种物体、中文名、固定尺寸、默认区域；提供 catalog_entries()、make_catalog_object() 与 CatalogEditor.apply()。请求 action/scene_version/xy/yaw 按设计约定，校验后一次更新。
- [x] 新建 assistant/catalog_perception.py：configure_sandbox(cfg)、make_sandbox_perception(cfg)。离线孤立物体渲染训练，运行时使用RGB-D特征分类；处理不同类别的完整性与抓取先验，保留L1.labels方块列表。
- [x] 新增有意义的目录、原子变更、分类与姿态覆盖测试；实测每种水果的渲染检测，记录拒绝原因。

## 2. 执行集成与反馈

- [x] simulation.py 构造 Sandbox runner，保留六视角和图像检测目标；加入 edit_scene()，清空过期检测/视角/占用/恢复数据。快照增加 editor、dimensions 和反馈，不改变真值与感知边界。
- [x] server.py 加入 edit_scene(request) 与 POST /api/scene-edit，统一 _launch worker，拒绝 busy/持物/过期请求。静态文件白名单增加 editor.js。
- [x] llm.py 中文说明从方块推广到物品，仍只选检测编号和已有区域。
- [x] 运行真实RGB-D→形状规划→IK→抓放的三种水果回归，修复有证据的失败，不扩大SafeArm限制。最终三个真实仿真测试文件合计 15 项通过，耗时 483.33 秒，包含三水果独立抓放、混合场景、原有助手与恢复回归。

## 3. 网页交互

- [x] 新建 editor.js：库卡片拖入俯视桌面，已有物体拖动、朝向滑块、删除、清空、不可编辑提示；过期版本与服务器错误可见。
- [x] app.js/index.html/style.css：接入编辑状态、中文水果名、按shape渲染球/椭球/香蕉链/组合物体，并展示测量和夹持反馈。沿用当前色彩和页面布局。
- [ ] 验证真实浏览器拖放、旋转和增删；当前任务期间锁定操作。模拟 DOM 已覆盖这些操作及 busy 锁，但真实浏览器工具因网址验证受限而中止，真实浏览器验收仍未完成。

## 4. 交付

- [x] 更新中文说明和来源，记录当前运动学/遮挡/类别限制。
- [x] 运行助手、恢复、相关旧规划测试和脚本语法检查；独立审查真值泄漏、编辑并发、几何不一致。已记录各组通过结果及 1 项在发布基线同样失败的旧规划测试；最终 `test_assistant_catalog_simulation.py`、`test_assistant_simulation.py`、`test_assistant_recovery_simulation.py` 为 15 passed in 483.33s。
- [x] 按功能提交、推送 Robot-Grasp，并将已验证改动同步回用户指定项目目录，保留当前上游原始分支和运行材料。功能版本 `be139b2` 已发布并同步；本地 `main` 跟踪新仓库，原上游保留为 `upstream`，原始分支为 `codex/agilex-original`，覆盖前文件及索引备份保存在本地 `.git/robot-grasp-sync-backup-*`。8767 已使用 D 盘新版重启，8768 临时测试服务已关闭。
