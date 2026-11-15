# 双 Franka 软熊装入布袋实验

**当前版本仅调整布袋的物理属性，目标是容易弯折、较难拉长的棉麻布效果。熊的尺寸、两片布直接缝合的袋型、机器人动作和流程保持原样。完整原生仿真 full40 与独立核对均通过，已设为桌面默认。**

## 查看与启动

- [完整视频](runs/two_panel_full_40/simulation.mp4)：约87秒。
- [原验收](runs/two_panel_full_40/validation.json)：20/20项通过。
- [独立核对](runs/two_panel_full_40/independent-gravity-execution-review.json)：41/41项通过，`passed=true`、`independent_task_success=true`。
- [执行报告](runs/two_panel_full_40/report.json)：完整结束，`requested_task_passed=true`。

双击桌面“**双机械臂布袋实验**”可运行当前版本。`config.json`与成功运行的冻结配置逐字一致；调整前默认已保存在`config_before_cotton_linen_material.json`。本机本次计算约57分钟，视频时长不等于计算耗时。

无窗口运行时使用一个尚不存在的输出名称：

```bash
/home/chloe/Desktop/genesis-local/teddy_bag_scene/run.sh \
  --headless --task full \
  --config /home/chloe/Desktop/genesis-local/teddy_bag_scene/config.json \
  --output teddy_bag_cotton_linen_new
```

需要窗口时将`--headless`换为`--viewer`。`--task handoff`仅检查撑袋与交接，不执行完整装袋。

## 本次仅修改的布袋参数

| 参数 | 调整前（full30） | 当前（full40） |
| --- | ---: | ---: |
| 膜内抗拉输入 E | 4 MPa | 10 MPa |
| 独立弯曲刚度 | 50 µN·m | 10 µN·m |
| 密度 | 500 kg/m³ | 333.333 kg/m³ |
| 对应面密度 | 300 g/m² | 200 g/m² |
| 提带有效剪切系数 | 250 N/m | 350 N/m |
| 推导袋重 | 120.0796 g | 80.0531 g |

面密度和袋重由密度及同一网格推导，并非新增几何修改。袋身剪切系数仍为50 N/m，摩擦输入仍为0.65；单边接触半径仍为0.3 mm，质量换算全厚仍为0.6 mm。降低弯曲刚度使布更容易弯折，提高膜内抗拉输入及提带抗剪切能力以减少拉长。

熊仍为s663，初始尺寸约15.94 × 21.28 × 6.69 cm，推导质量63.7795 g。袋身仍是两片46 × 40 cm布沿左右和底边直接缝合，上口开放，没有底片、侧片或立体底宽。网格、材质贴图、熊材料、抓取控制、路径规则、求解设置和原验收门均未更改。

双臂撑袋后，左手持续握持前提带，右手松开后提带，抓熊、抬升、转向、搬运、下放、松爪、撤手。布袋没有刚体或隐藏支撑，不用固定点、理想粘附、重置软体顶点或运行中缩放辅助装袋。控制规则和输入保持一致；材料改变实际袋口观测后，原控制器计算出的运动时长可以不同。

## 已测到的材料效果

对完整保存状态使用与原版相同的分析方法：

| 持袋前提带的局部峰值 | 调整前 | 当前 |
| --- | ---: | ---: |
| 构造长轴伸长 | 4.78% | 2.35% |
| 任意面内方向最大主伸长 | 8.57% | 4.62% |
| 原生膜内轴最大伸长 | 2.13% | 1.24% |

这三种方向定义不同，不能互换，也不是实物纱线测试。提带的改善并不表示整袋每个热点都改善：后片局部主伸长峰由3.49%增至5.95%。布料仍允许折皱和斜向变形；没有新增粘性阻尼或纱线迟滞，不能称为完全不伸长或回弹已消除。

[完整应变对比](runtime_ipc/full_40-cloth-material-baseline.json)、[原生轴诊断](runtime_ipc/two_panel_full_40-native-axis-strain-review-v2.json)与[材料读回](runtime_ipc/full40-native-material-readback-review.json)均保留。保存状态间隔约50 ms，局部峰为已保存帧中的峰值。

## 完整任务验证

实际仿真0–86.760秒；60.950秒开始松爪，61.450秒完成张爪，82.760秒完成右臂撤离，随后自然稳定4秒。

- 松爪前与最终状态的全部1450个熊节点及体积加权质心均在分析袋身内；末尾三个留袋样本通过。
- 每步最小四面体体积比0.10948，超过原0.05下限；没有体积翻转。
- 最大布边长度比1.04138，低于原1.15上限；记录的桌面穿透为0。
- 右侧手部最大软体接触合力0.88154 N、TCP接触力矩0.05768 N·m，低于原8 N／0.15 N·m试验门限。
- 实际关节速度和加速度均在原限制内；左手指定提带材料组保持受夹持，最终右夹爪张开。
- 最终12521个布与熊离散检查样点未发现超过1 mm的穿入；离散检查不构成连续无穿透证明。

独立审计复核冻结输入、实际轨迹、原生反力、松爪条件、撤离顺序和最终留袋结果。独立报告中的旧顶层`physical_success=false`是未更新的初始化字段，结论以`passed=true`、`independent_task_success=true`和原`task_success=true`为准；原报告未改写。

## 材料依据与文件

200 g/m²与约10 µN·m的弯曲刚度参考公开棉和棉麻织物实测量级。当前参数并非来自同一真实样布的完整标定，缝制提带350 N/m是工程近似；详见[材料参考记录](material_reference_notes.json)。熊仍使用同一个均匀可压缩弹性体模型，没有新增表皮或填充纤维模型。

当前资产为`assets/bear_uniform_coarse_s663.obj`及`assets/two_panel_bag_wide_tall_long_handles_coarse.*`。full40目录保存冻结配置、源码、资产哈希和执行记录。此前完整成功的full30、full38与失败/主动中断的材料试验均保留，未把中断或预检拒绝当作成功。full39提带更强的候选未通过实际袋口检查，未采用。

调整前说明见[原默认说明](docs/README_BEFORE_COTTON_LINEN_FULL40_CN.md)。本地环境仍为Python 3.12、Genesis 1.4.3和pyuipc 0.0.29；Genesis使用CPU前端，IPC使用CUDA后端。启动器不安装依赖，本次没有修改依赖环境。
