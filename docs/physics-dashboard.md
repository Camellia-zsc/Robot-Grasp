# 物理抓取与叠放

交互界面现在默认使用 MuJoCo。物品是带质量的自由刚体，机械臂通过关节伺服运动，双指通过碰撞、接触和摩擦抓住物品。界面中的物品位置是物理结果的显示镜像；不会将物品焊在夹爪上，也不会在释放时直接写入一个“成功位置”。

## 安装和启动

在已有 NumPy、OpenCV、SciPy 等项目依赖的 Python 环境中，安装独立的物理依赖：

```powershell
python -m venv --system-site-packages .venv-physics
.\.venv-physics\Scripts\python.exe -m pip install -r requirements-physics.txt
```

MuJoCo 版本固定为 **3.3.7**，因为现有机械臂驱动使用 3.x 的惯量矩阵接口。界面启动前会检查版本；缺少依赖或版本不符会明确报错。

运行根目录的“启动智能抓取界面.cmd”会优先使用本项目 `.venv-physics`。也可设置 `MUJOCO_PYTHON`，指定已有物理环境的 Python。直接启动：

```powershell
cd ws\src\cube_task
& '<物理环境的 Python 完整路径>' -m cube_task assistant_ui --backend mujoco --port 8767 --open-browser
```

如需复现实验原有的运动学模式，显式添加 `--backend sim`；该模式不提供真实重力叠放。`--fast` 可用于离线验证，它取消画面时间节奏，不跳过物理步或接触检查。

## 夹爪模型

本项目的 `robot.gripper.max_width` 已配置为 **100 mm**。若未单独指定 `sim.mujoco.gripper_max_width`，交互物理环境将此行程写入 MJCF 的真实双指滑动关节范围，保留力限制、接触和摩擦。界面会显示实际物理夹爪行程。

这是项目采用的扩展行程模型。原始 PiPER MJCF 的 70 mm 行程仍可通过 `sim.mujoco.gripper_max_width` 单独配置；真实机械臂的可夹宽度应按实际硬件校验。形状完整性、指尖间隙和碰撞限制不会为了抓住某种物品而取消。

## 使用和边界

- 可输入“把苹果放到红色方块上”。红色区域与红色方块顶面是两个不同的目标。
- 首版支持以相机确认的平顶方块为底座。苹果、香蕉、梨等曲面物品可以作为被放置物；曲面物品暂不作为自动规划中的可靠底座。
- 顶面位置和释放高度来自 RGB-D 测量。释放后会等待落稳，并同时检查新的相机证据、真实物理接触及双方运动速度。
- 偏心、滚动、掉落或人为移走底座会使原完成证据失效。系统会报告现场变化，供后续重新观察和恢复；不会将“动作执行完了”等同于“叠放成功”。
- 重心不必在同一条竖线上；能否稳定由接触支撑范围、质量分布和摩擦共同决定。MuJoCo 会实际积分运动，但这些库物品仍是简化的刚体碰撞模型，尚未模拟水果变形。

无需在线 API 的物理验证：

```powershell
cd ws\src\cube_task
& '<物理环境的 Python 完整路径>' -m pytest -q test/test_assistant_physics.py test/test_assistant_physics_session.py
```

验证包含居中叠放、偏心下落、滚动、移走底座、编辑重建、真实夹持和搬运掉落，以及完整的相机识别→抓苹果→叠到红块→移走红块后原目标不再成立。
