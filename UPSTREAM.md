# 源码与模型来源

## 项目基线和提交结构

- 原项目：[Oner-Z/agileX](https://github.com/Oner-Z/agileX)。
- 本地首次下载：2026-11-14，基线提交 [`ed5912fe6abd1313a91268bfec947f94cc456686`](https://github.com/Oner-Z/agileX/commit/ed5912fe6abd1313a91268bfec947f94cc456686)。
- 新项目：[Camellia-zsc/Robot-Grasp](https://github.com/Camellia-zsc/Robot-Grasp)。
- 第一个提交是无父提交的原始源码快照，Git tree 与上述基线完全一致，保留原文件和子模块版本，不导入上游历史。
- 第二个提交记录截至 2026-11-16 的本地开发：交互界面、在线大模型、RGB-D 多视角观测、L1 抓放与恢复、测试、中文文档和验收材料。

原作者信息、代码中的版权和许可声明予以保留；本文件不改变上游代码及依赖的许可条件。

## 官方依赖

以下仍是原项目的 Git 子模块，由 `.gitmodules` 记录地址：

| 路径 | 来源 | 固定提交 |
|---|---|---|
| `third_party/pyAgxArm` | [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) | `d90e234ba8e408cfa964b38aa9120b3131fc9f85` |
| `ws/src/agx_arm_ros` | [agilexrobotics/agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) | `286633e0d2edb038ee3a1047d9feb7f614a1c076` |
| `ws/src/handeye_calibration_ros` | [agilexrobotics/handeye_calibration_ros](https://github.com/agilexrobotics/handeye_calibration_ros) | `3f8e29606f91b2e6c46457961af03154f7ad3809` |

纯 L1 运动学仿真和网页界面不依赖这些子模块。需要 ROS 2 描述与 MoveIt 配置时，可选择性下载：

```bash
git submodule update --init --recursive ws/src/agx_arm_ros
```

## PiPER MuJoCo 模型

`local_assets/mujoco_menagerie/agilex_piper/` 以普通文件形式收录所需模型、网格、说明和 LICENSE，便于后续物理仿真复现。

- 来源：[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/agilex_piper)。
- 下载版本：`370c43965323c0944b433b202aac5058b5f907a9`。
- 模型许可：该目录的 MIT LICENSE，Copyright (c) 2024 RosenYin。
- 只收录 `agilex_piper` 子目录；不收录模型库的 Git 元数据及其他机器人模型。当前默认网页演示仍使用 L1，不因收录模型而表示已完成接触物理验证。

## 可复现材料与本机配置

`local_validation/` 保留历史 L1/L2 运行报告、界面与恢复验收说明、JSON、截图和复现脚本。原始报告中的绝对路径和时间代表验收时的本机环境。

运行缓存、服务进程号、临时服务日志、构建产物和本机凭据不纳入发布。API 地址、模型和密钥由运行者在本机配置，详见 [大模型抓取界面使用说明](大模型抓取界面使用说明.md)。验收脚本可能调用收费 API，应先阅读脚本参数和说明。
