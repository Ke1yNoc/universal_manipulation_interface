import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from pika.gripper import Gripper


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class PikaGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            serial_path,
            frequency=30,
            move_max_speed=200.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=True,
            verbose=False
            ):
        super().__init__(name="PikaGripperController")
        self.serial_path = serial_path
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        example_rb = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
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
            print(f"[PikaGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        self.input_queue.put({'cmd': Command.SHUTDOWN.value})
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

    def schedule_waypoint(self, pos: float, target_time: float):
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        })

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        grip = Gripper(self.serial_path)
        connected = grip.connect()
        if not connected:
            raise RuntimeError(f"Failed to connect Pika gripper at {self.serial_path}")
        enabled = grip.enable()
        if not enabled:
            raise RuntimeError("Failed to enable Pika gripper")

        curr_pos_mm = grip.get_gripper_distance()
        curr_pos = curr_pos_mm / self.scale
        curr_t = time.monotonic()
        last_pos = curr_pos
        pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[[curr_pos,0,0,0,0,0]])

        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        try:
            while keep_running:
                t_now = time.monotonic()
                dt = 1.0 / self.frequency
                target_pos = pose_interp(t_now)[0]
                target_vel = (target_pos - pose_interp(t_now - dt)[0]) / dt

                grip.set_gripper_distance(int(target_pos * self.scale))

                curr_pos_mm = grip.get_gripper_distance()
                curr_pos = curr_pos_mm / self.scale
                velocity = (curr_pos - last_pos) / dt
                last_pos = curr_pos

                state = {
                    'gripper_state': 1,
                    'gripper_position': curr_pos,
                    'gripper_velocity': velocity,
                    'gripper_force': 0.0,
                    'gripper_measure_timestamp': time.time(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {k: v[i] for k, v in commands.items()}
                    cmd = command['cmd']
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=t_now,
                            last_waypoint_time=t_now
                        )
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                t_end = t_start + dt * (iter_idx + 1)
                precise_wait(t_end=t_end, time_func=time.monotonic)
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
        finally:
            self.ready_event.set()
            grip.disable()
            grip.disconnect()
            if self.verbose:
                print(f"[PikaGripperController] Disconnected {self.serial_path}")
