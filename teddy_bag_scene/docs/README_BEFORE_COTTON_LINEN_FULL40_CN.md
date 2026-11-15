# 双 Franka 软熊装入布袋实验

**熊已按上一版放大 30%，当前初始尺寸约 15.94 × 21.28 × 6.69 cm。完整原生仿真 full30 已通过原验收和独立核对，并已设为桌面默认版本。**

项目目录：`/home/chloe/Desktop/genesis-local/teddy_bag_scene`。

## 查看结果与启动

- 最终视频：[simulation.mp4](runs/two_panel_full_30/simulation.mp4)，约 77 秒。
- 独立核对：[independent-gravity-execution-review.json](runs/two_panel_full_30/independent-gravity-execution-review.json)，41/41 项通过，`passed` 和 `independent_task_success` 均为 `true`。
- 原验收：[validation.json](runs/two_panel_full_30/validation.json)，`task_success=true`，21 项通过。
- 执行报告：[report.json](runs/two_panel_full_30/report.json)，`status=completed`、`requested_task_passed=true`。

双击桌面“**双机械臂布袋实验**”可运行当前版本。默认配置 `config.json` 与成功运行的冻结配置逐字一致；上一默认已备份为 `config_before_s663_full30.json`。本机本次计算耗时约 44.5 分钟，视频播放时长不是实时计算耗时。

无窗口运行时显式指定一个尚不存在的输出目录：

```bash
/home/chloe/Desktop/genesis-local/teddy_bag_scene/run.sh \
  --headless --task full \
  --config /home/chloe/Desktop/genesis-local/teddy_bag_scene/config.json \
  --output teddy_bag_s663_new
```

需要窗口时将 `--headless` 换为 `--viewer`。`--task handoff` 仅检查撑袋和单手持袋，不执行完整装袋。

## 当前场景

| 项目 | 当前值 |
| --- | --- |
| 熊的初始尺寸（资产 X × Y × Z） | 15.94 × 21.28 × 6.69 cm |
| 尺寸来源 | 上一版 s51 线性放大 30%，即此前 s85 的 78% |
| 熊的推导质量 | 63.7795 g，密度保持 100 kg/m³ |
| 熊的夹持控制 | 每指力上限 1.8252 N，闭合目标 0；上限不等于实际接触力 |
| 布袋 | 两片 46 × 40 cm 柔性布沿左右和底边直接缝合，上口开放 |
| 袋底与提带 | U 形缝，无底片、侧片或立体底宽；两根柔性提带 |
| 袋的推导质量 | 120.0796 g |
| 桌面／双手撑袋目标高度 | 0.74 m／1.40 m |
| 已验证候选 | `config_bear_s663_offset_left_trial.json` |
| 已验证运行 | `runs/two_panel_full_30/` |

熊初始仰躺在桌面上。双臂撑袋后，左手持续握持前提带，右手松开后提带并撤离，再抓熊、抬升、转向、搬运、下放、松爪和撤手。袋身没有刚体或隐藏支撑；运行中不重置软体顶点，不用固定点、理想粘附或人为缩放辅助装袋。

本轮入袋位置依据前次接触记录向 −X 调整 15 mm，前后位置保持原值。实际袋口观测和路径检查通过后，下放到 TCP 高度 0.915 m 停稳，再主动松爪。低位释放减少了下落距离。运动、形变、抓持和接触保护门限没有放宽。

## 本次验证结果

实际仿真从 0 执行到 76.710 s：55.510 s 开始松爪，56.010 s 完成张爪，72.710 s 完成右臂撤离，之后自然稳定 4 秒。

- 松爪前和最终状态均为全部 1450 个熊节点及体积加权质心在分析袋身内；末尾三个留袋样本均通过。
- 每步最小四面体体积比 0.10894，大于原 0.05 下限；没有体积翻转。
- 布边最大长度比 1.04794，低于原 1.15 上限；记录的桌面穿透为 0。
- 右侧手部最大 IPC 合力 1.78591 N、TCP 接触力矩 0.03689 N·m，低于原 8 N／0.15 N·m 试验门限。
- 左手交接后最大 TCP 位移约 0.184 mm；最终右夹爪已张开。
- 最终布与熊的 12521 个离散接触检查样点未发现超过 1 mm 的穿入。

独立审计重新检查了冻结输入、归档轨迹、原生反力、真实松爪条件、撤离顺序和原最终验收。报告顶层旧字段 `physical_success=false` 是未更新的初始化字段；本次结论以 `passed=true`、`independent_task_success=true` 和重算的原 `task_success=true` 为准，原始报告未修改。

## 物理模型与适用范围

布袋使用 FEM 薄壳，熊使用连续四面体可压缩弹性体；接触、摩擦及机器人反作用由原生求解器处理。关节以有限力 PD 跟踪连续限速参考轨迹。布袋仍是两片布直接缝合，分析时使用的虚拟口盖不参与仿真或承载。

材料参数尚未用真实布料和玩偶标定。熊没有独立表布与内部纤维模型；袋身／提带的有效剪切系数为 50／250 N/m。IPC `thickness=0.0003 m` 为单边接触半径，质量按 `2r=0.0006 m` 换算，并非实测厂家布厚。

成功结论适用于本次离散仿真与现有检查。保存状态和离散碰撞检查不构成连续无穿透证明，仿真成功也不等同于实物机器人已经验证。0.15 N·m 等接触门限是未标定的工程试验保护，不是已证实的硬件或材料失效界限。

## 文件与历史

当前资产为 `assets/bear_uniform_coarse_s663.obj` 和 `assets/two_panel_bag_wide_tall_long_handles_coarse.*`。成功运行目录保留配置、源码、资产哈希、实际状态、逐步指标、关节与接触记录。`dependency_manifest.json` 的当前结果、默认入口和最新审计字段已更新；其中旧候选字段保留作历史索引。

本地环境为 Python 3.12、Genesis 1.4.3 可编辑源码安装，以及场景 `vendor/` 中的 pyuipc 0.0.29。Genesis 使用 CPU 前端，IPC 使用 CUDA 后端；无窗口运行也需离屏 OpenGL/EGL。启动器只设置当前进程环境，不安装依赖，复制目录不能保证其他机器可直接复现。

full28 在较高位置松爪后因脚部压缩触发保护；full29 下放时因手掌加载前提带根部而触发力矩保护。两轮原始失败记录保持不变。full26／full27 因用户改尺寸主动中断，也不计为成功。此前过程见 [full30 成功前说明](docs/DEVELOPMENT_HISTORY_CN_2026-11-15_BEFORE_FULL30_SUCCESS.md)、[full28 前说明](docs/DEVELOPMENT_HISTORY_CN_2026-11-15_BEFORE_FULL28_README.md) 和 [更早开发历史](docs/DEVELOPMENT_HISTORY_CN_2026-11-13.md)。
