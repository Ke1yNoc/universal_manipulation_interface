import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import scipy.spatial.transform as st
from piper_sdk import C_PiperInterface_V2
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


def _rotvec_to_piper_euler_thousand_deg(rv):
    r = st.Rotation.from_rotvec(rv)
    euler = r.as_euler('xyz', degrees=True)
    return np.round(euler * 1000).astype(np.int64)


def _piper_euler_thousand_deg_to_rotvec(euler_thousand_deg):
    euler_deg = np.asarray(euler_thousand_deg, dtype=np.float64) / 1000.0
    r = st.Rotation.from_euler('xyz', euler_deg, degrees=True)
    return r.as_rotvec()

def _j6_to_tcp_pos_mm_1000(pos_mm_1000, euler_thousand_deg, tool_offset_m):
    X, Y, Z = pos_mm_1000
    RX, RY, RZ = euler_thousand_deg
    rx_rad = np.deg2rad(RX / 1000.0)
    ry_rad = np.deg2rad(RY / 1000.0)
    rz_rad = np.deg2rad(RZ / 1000.0)
    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy
    tx = tool_offset_m[0] * 1_000_000.0
    ty = tool_offset_m[1] * 1_000_000.0
    tz = tool_offset_m[2] * 1_000_000.0
    off_x = r00 * tx + r01 * ty + r02 * tz
    off_y = r10 * tx + r11 * ty + r12 * tz
    off_z = r20 * tx + r21 * ty + r22 * tz
    return (int(np.round(X + off_x)), int(np.round(Y + off_y)), int(np.round(Z + off_z)))


def _apply_tool_offset_tcp_to_j6_rotvec(pos_mm_1000, rotvec_rad, tool_offset_m):
    X, Y, Z = pos_mm_1000
    R = st.Rotation.from_rotvec(rotvec_rad).as_matrix()
    tool_mm = np.array(tool_offset_m, dtype=np.float64) * 1_000_000.0
    off = R @ tool_mm
    return (
        int(np.round(X - off[0])),
        int(np.round(Y - off[1])),
        int(np.round(Z - off[2]))
    )


class PiperInterpolationController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            piper_can,
            frequency=200,
            lookahead_time=0.1,
            gain_pos=0.25,
            gain_rot=0.16,
            launch_timeout=5,
            tool_offset=(0.0, 0.0, 0.1773),
            receive_latency=0.0,
            verbose=False,
            get_max_k=None,
            speed_percent=100,
            go_zero_on_start=True):
        # piper_can: Piper 臂 CAN 接口标识
        # frequency: 控制循环频率（建议 100/150/200Hz；避免别名化导致零帧）
        # speed_percent: 速度百分比（1-100），用于 ModeCtrl/MotionCtrl_2
        # go_zero_on_start: 启动后是否回零位
        if get_max_k is None:
            get_max_k = int(frequency * 5)
        super().__init__(name="PiperPositionalController")
        self.piper_can = piper_can
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain_pos = gain_pos
        self.gain_rot = gain_rot
        self.launch_timeout = launch_timeout
        self.tool_offset = np.array(tool_offset, dtype=np.float64)
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.speed_percent = int(np.clip(speed_percent, 1, 100))
        self.go_zero_on_start = bool(go_zero_on_start)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        example_rb = {
            'ActualTCPPose': np.zeros(6, dtype=np.float64),
            'ActualQ': np.zeros(6, dtype=np.float64),
            'ActualQd': np.zeros(6, dtype=np.float64),
            'TargetTCPPose': np.zeros(6, dtype=np.float64),
            'TargetQ': np.zeros(6, dtype=np.float64),
            'robot_receive_timestamp': time.time(),
            'robot_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_rb,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert duration >= (1/self.frequency)
        pose = np.asarray(pose, dtype=np.float64)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': float(duration)
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        pose = np.asarray(pose, dtype=np.float64)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': float(target_time)
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        robot = C_PiperInterface_V2(self.piper_can, dh_is_offset=1)
        robot.ConnectPort()
        status = robot.GetArmStatus()
        if status.arm_status.motion_status != 0:
            robot.EmergencyStop(0x02)
        if status.arm_status.ctrl_mode == 2:
            robot.EmergencyStop(0x02)
        ok = False
        start_t = time.time()
        while not ok:
            ok = robot.EnablePiper()
            time.sleep(0.01)
        # 设置关节/末端控制速度百分比
        robot.ModeCtrl(0x01, 0x01, self.speed_percent, 0x00)
        time.sleep(0.5)

        if self.go_zero_on_start:
            # robot.JointCtrl(0, 24000, -45000, 0, 40000, 0)
            robot.JointCtrl(0,0,0,0,0,0)
            time.sleep(0.5)

        robot.MotionCtrl_2(0x01, 0x00, self.speed_percent, 0x00)
        time.sleep(0.5)

        dt = 1.0 / self.frequency
        init_ok = False
        x0 = robot.GetArmEndPoseMsgs().end_pose
        pos_mm0 = np.array([x0.X_axis, x0.Y_axis, x0.Z_axis], dtype=np.float64)
        euler_thousand_deg0 = np.array([x0.RX_axis, x0.RY_axis, x0.RZ_axis], dtype=np.float64)
        tcp_mm0 = _j6_to_tcp_pos_mm_1000(pos_mm0, euler_thousand_deg0, self.tool_offset)
        pos0 = np.array(tcp_mm0, dtype=np.float64) / 1_000_000.0
        rotvec0 = _piper_euler_thousand_deg_to_rotvec(euler_thousand_deg0)
        curr_pose = np.concatenate([pos0, rotvec0])
        curr_t = time.monotonic()
        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])
        t_start = time.monotonic()
        iter_idx = 0
        last_q = np.zeros(6, dtype=np.float64)
        last_tcp = curr_pose.copy()
        try:
            while True:
                t_now = time.monotonic()
                tcp_pose = pose_interp(t_now)
                target_pose = tcp_pose.copy()
                # 写入目标状态（用于观测/录制对齐）
                self.ring_buffer.put({
                    'TargetTCPPose': target_pose,
                    'TargetQ': np.zeros(6, dtype=np.float64)
                })

                tcp_pos_m = target_pose[:3]
                rotvec = target_pose[3:]
                euler_thousand = _rotvec_to_piper_euler_thousand_deg(rotvec)
                pos_mm_1000 = np.round(tcp_pos_m * 1_000_000.0).astype(np.int64)
                j6_xyz = _apply_tool_offset_tcp_to_j6_rotvec(pos_mm_1000, rotvec, self.tool_offset)
                RX, RY, RZ = int(euler_thousand[0]), int(euler_thousand[1]), int(euler_thousand[2])
                robot.EndPoseCtrl(int(j6_xyz[0]), int(j6_xyz[1]), int(j6_xyz[2]), RX, RY, RZ)

                # 读取末端姿态；零帧回退至上一帧，避免间歇性 0 值
                state_end = robot.GetArmEndPoseMsgs().end_pose
                pos_mm = np.array([state_end.X_axis, state_end.Y_axis, state_end.Z_axis], dtype=np.float64)
                euler_thousand = np.array([state_end.RX_axis, state_end.RY_axis, state_end.RZ_axis], dtype=np.float64)
                tcp_mm = _j6_to_tcp_pos_mm_1000(pos_mm, euler_thousand, self.tool_offset)
                pos_m = np.array(tcp_mm, dtype=np.float64) / 1_000_000.0
                if not np.any(np.concatenate([pos_m, euler_thousand])):
                    state_tcp = last_tcp
                else:
                    rotvec = _piper_euler_thousand_deg_to_rotvec(euler_thousand)
                    state_tcp = np.concatenate([pos_m, rotvec])
                    last_tcp = state_tcp

                # 读取关节状态并转弧度
                js = robot.GetArmJointMsgs().joint_state
                q = np.array([js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6], dtype=np.float64)
                q = q / 57295.7795
                qd = (q - last_q) * self.frequency
                last_q = q
                t_recv = time.time()
                # 写入实测状态与时间戳（对齐用）
                self.ring_buffer.put({
                    'ActualTCPPose': state_tcp,
                    'ActualQ': q,
                    'ActualQd': qd,
                    'robot_receive_timestamp': t_recv,
                    'robot_timestamp': t_recv - self.receive_latency
                })

                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    command = {k: v[i] for k, v in commands.items()}
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        print(target_pos)
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.gain_pos,
                            max_rot_speed=self.gain_rot
                        )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time_input = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time_input
                        curr_time = t_now + dt
                        if self.verbose:
                            print(f"[Piper] Schedule: In={target_time_input:.3f}, Mono={target_time:.3f}, Curr={curr_time:.3f}, Diff={target_time-curr_time:.3f}")
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time
                        )

                t_wait = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait, time_func=time.monotonic)
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
        except KeyboardInterrupt:
            pass
        finally:
            try:
                robot.DisablePiper()
            except Exception:
                pass
            self.ready_event.set()
        if self.verbose:
            print(f"[PiperPositionalController] Disconnected from {self.piper_can}")
