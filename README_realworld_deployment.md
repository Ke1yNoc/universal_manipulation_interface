# UMI 真实部署与扩展指南

本指南面向真实世界部署与扩展，梳理系统架构、部署流程，并给出新增机械臂（示例：AgileX Piper）的集成方法。

## 系统架构总览
- 主评估入口：`eval_real.py`，读取 `example/eval_robots_config.yaml` 构造多臂环境，组织观测、推理与动作执行。参考 `eval_real.py:108-126`、`eval_real.py:175-197`。
- 环境层：
  - 双臂环境 `umi/real_world/bimanual_umi_env.py` 依据 `robot_type` 实例化控制器并管理相机、抓手、录像与时间对齐。参考 `umi/real_world/bimanual_umi_env.py:26-67`、`umi/real_world/bimanual_umi_env.py:208-246`、`umi/real_world/bimanual_umi_env.py:351-479`、`umi/real_world/bimanual_umi_env.py:523-631`。
  - 单臂环境 `umi/real_world/umi_env.py` 与双臂一致的接口。参考 `umi/real_world/umi_env.py:30-87`、`umi/real_world/umi_env.py:231-261`、`umi/real_world/umi_env.py:350-444`、`umi/real_world/umi_env.py:446-589`。
- 控制器层：
  - UR 系列 RTDE 控制器 `umi/real_world/rtde_interpolation_controller.py`，独立进程，支持 `servoL` 与时间戳对齐的 `schedule_waypoint`。参考 `umi/real_world/rtde_interpolation_controller.py:23-50`、`umi/real_world/rtde_interpolation_controller.py:210-219`、`umi/real_world/rtde_interpolation_controller.py:221-373`。
  - Franka 控制器 `umi/real_world/franka_interpolation_controller.py`，通过 ZeroRPC 与中间层交互（Polymetis 服务端），接口与 RTDE 控制器一致。参考 `umi/real_world/franka_interpolation_controller.py:71-91`、`umi/real_world/franka_interpolation_controller.py:224-233`、`umi/real_world/franka_interpolation_controller.py:235-376`。
- 抓手层：WSG50 `umi/real_world/wsg_controller.py`，与环境同频对齐，接口为 `schedule_waypoint/ get_all_state`。
- 相机与可视化：`umi/real_world/multi_uvc_camera.py`、`umi/real_world/multi_camera_visualizer.py`，统一采集、变换、录像与展示。
- Franka 部署说明：`franka_instruction.md`，包含服务器端与客户端入口。参考 `franka_instruction.md:6-11`、`franka_instruction.md:14-30`。

## 部署流程
- 硬件准备
  - 机器人臂：UR5/UR5e 或 Franka（需中间层 NUC + Polymetis）。UR 教学器设置见主 `README`；Franka 参考 `franka_instruction.md`。
  - 抓手：WSG50，按主 `README` 配置 Web 端与启动脚本。
  - 相机：GoPro + HDMI 采集卡，授予 USB 权限：`sudo chmod -R 777 /dev/bus/usb`。
  - 遥操作：3Dconnexion SpaceMouse 安装 `libspnav-dev spacenavd` 并启动服务。
- 机器人配置
  - 编辑 `example/eval_robots_config.yaml`：每条 `robots`/`grippers` 对应一只手臂与一个抓手，`robot_type` 取 `ur5/ur5e/franka`。
  - 参数含义：
    - `robot_obs_latency / robot_action_latency` 与 `gripper_*_latency` 为软对齐延迟（秒）。
    - `tcp_offset` 为末端到 TCP 的 z 轴偏置（米）。
    - `height_threshold` 为桌面碰撞抬升阈值（米，`-inf` 关闭）。
    - `sphere_{radius, center}` 为双臂间球体碰撞规避参数。
    - `tx_left_right` 为右臂基坐标在左臂基坐标系下的齐次变换。
- Franka 专属步骤
  - 在实时内核主机安装 Polymetis 并启动服务端：`python scripts_real/launch_franka_interface_server.py`。参考 `franka_instruction.md:14-21`。
  - 在评估机将 `robots[*].robot_type` 设为 `franka` 并写入 `robot_ip`（指向中间层）。
- 运行评估
  - 下载或指定策略 checkpoint（`.ckpt`）。
  - 启动评估脚本：
    - 多臂：`python eval_real.py --robot_config=example/eval_robots_config.yaml -i <checkpoint> -o <output_dir>`。
  - 人机切换与键位：窗口聚焦后按 `C` 交给策略、`S` 夺回、`Q` 退出。SpaceMouse 左键开旋转、右键开 Z 轴。
  - 数据与视频：自动保存至 `<output_dir>/replay_buffer.zarr` 与 `<output_dir>/videos/*`。

## 关键时序与对齐
- 控制频率 `frequency`（Hz）决定动作步长 `dt=1/frequency`，相机默认 60Hz，UR RTDE 125/500Hz，Franka 1000Hz 中间层。
- 环境对齐策略：以对齐相机最后时间戳为“当前时刻”，相机按最近帧对齐，机器人与抓手用插值对齐到统一时间轴。参考 `umi/real_world/bimanual_umi_env.py:352-458`。
- 动作调度：策略产生 `[N, 7 * n_robots]` 的目标序列，环境将每步拆分为每臂 6D 末端位姿 + 1D 抓手宽度，并按目标时间投递到控制器与抓手。参考 `umi/real_world/bimanual_umi_env.py:481-521`。

## 新增机械臂集成（示例：AgileX Piper）
- 选择控制方案
  - 直接驱动：如果 Piper 提供官方 Python/SDK，可仿照 RTDE 控制器实现独立进程，周期性发送末端位姿命令并采集状态。
  - 中间层驱动：若 Piper 通过外部实时服务（如 ROS/自研控制器）更易实现，仿照 Franka 采用 ZeroRPC 客户端与服务端交互。
- 控制器接口要求（需与环境一致）
  - 进程生命周期：`start(wait) / stop(wait) / is_ready`。
  - 命令：`servoL(pose, duration)`（插值到位）、`schedule_waypoint(pose, target_time)`（按绝对时间调度）。
  - 状态接收：环形缓冲返回字典，至少包含：
    - `ActualTCPPose`（6D 末端姿态）
    - `ActualQ`（关节位置）
    - `ActualQd`（关节速度）
    - `robot_receive_timestamp`、`robot_timestamp`（秒）
  - 参考实现：
    - UR RTDE 状态与键名对应见 `umi/real_world/rtde_interpolation_controller.py:116-140`。
    - Franka 通过函数映射封装见 `umi/real_world/franka_interpolation_controller.py:134-155`。
- 环境接入
  - 在 `umi/real_world/bimanual_umi_env.py` 与 `umi/real_world/umi_env.py` 增加分支：当 `robot_type.startswith('piper')` 时实例化 `PiperInterpolationController`。
  - 频率与增益：依据 Piper 官方极限速度/加速度选择安全的 `frequency/max_pos_speed/max_rot_speed`。
- 配置文件示例
```json
{
  "robots": [
    {
      "robot_type": "piper",
      "robot_ip": "192.168.0.50",
      "robot_obs_latency": 0.0001,
      "robot_action_latency": 0.05,
      "tcp_offset": 0.200,
      "height_threshold": -0.020,
      "sphere_radius": 0.1,
      "sphere_center": [0, -0.06, -0.185]
    }
  ],
  "grippers": [
    {
      "gripper_ip": "192.168.0.60",
      "gripper_port": 1000,
      "gripper_obs_latency": 0.01,
      "gripper_action_latency": 0.1
    }
  ],
  "tx_left_right": [
    [1,0,0,0],
    [0,1,0,-0.60],
    [0,0,1,0],
    [0,0,0,1]
  ]
}
```
- 最小实现提示
  - 若走中间层，参考 Franka 客户端封装 `FrankaInterface`，将 Piper 的末端位姿统一到 6D 轴角格式，确保 TCP 偏置与工具坐标一致。
  - 若走直驱，参考 UR 的插值器 `PoseTrajectoryInterpolator` 与环形缓冲，保证周期发送与频率稳定。
- 验证步骤
  - 先用遥操作脚本验证控制链路：`python scripts_real/control_robots.py` 或在评估脚本 `eval_real.py` 人控阶段通过 SpaceMouse 观察末端响应与状态刷新。
  - 确认 `is_ready` 为真、状态时间戳单调递增、`get_all_state()` 返回的键齐全。
  - 配置策略 checkpoint 后进入策略控制阶段，观察动作投递条数与时间预算日志。

## 安全与常见问题
- 强烈建议在初期限制 `max_pos_speed/max_rot_speed` 并仅在空载与远离桌面测试，逐步打开速度上限。
- 若相机画面无更新或录像失败，先执行 USB 设备重置（环境会自动重置 Elgato），再检查权限与采集帧率配置。
- SLAM 与数据收集弱光敏感，尽量避免强直射光环境；详见主 `README` 的已知问题说明。

## 参考与入口
- 主 `README.md` 的“🦾 Real-world Deployment”与硬件/抓手/相机配置说明。
- Franka 专属指引：`franka_instruction.md`。
- 评估入口：`eval_real.py`。双臂评估参考 `scripts_real/eval_real_bimanual_umi.py`（命令行参数相同）。

