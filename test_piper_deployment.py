import os
import time
import click
import yaml
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import cv2
from umi.real_world.piper_interpolation_controller import PiperInterpolationController
from umi.real_world.pika_gripper_controller import PikaGripperController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.multi_uvc_camera import MultiUvcCamera

@click.command()
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--frequency', '-f', default=10, type=float)
@click.option('--test_move', is_flag=True, default=False)
@click.option('--move_axis', default='z', type=click.Choice(['x','y','z']))
@click.option('--amp', default=0.01, type=float)
@click.option('--duration', default=2.0, type=float)
@click.option('--hold', default=5.0, type=float, help='Hold time for idle sampling when not moving')
@click.option('--gripper_targets', default='', type=str)
@click.option('--gripper_delay', default=1.0, type=float)
@click.option('--save_video', is_flag=True, default=False)
@click.option('--video_dir', default='videos', type=str)
def main(robot_config, frequency, test_move, move_axis, amp, duration, hold, gripper_targets, gripper_delay, save_video, video_dir):
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data.get('grippers', [])
    cameras_config = robot_config_data.get('cameras', None)
    with SharedMemoryManager() as shm_manager:
        controllers = []
        grippers = []
        multi_cam = None
        for rc in robots_config:
            assert rc['robot_type'].startswith('piper')
            ctrl = PiperInterpolationController(
                shm_manager=shm_manager,
                piper_can=rc.get('piper_can', 'can0'),
                frequency=150,
                lookahead_time=0.1,
                gain_pos=2.0,
                gain_rot=6.0,
                tool_offset=[0,0,rc.get('tcp_offset', 0.0)],
                receive_latency=rc.get('robot_obs_latency', 0.0001),
                verbose=False
            )
            controllers.append(ctrl)
        for idx, gc in enumerate(grippers_config):
            if gc.get('gripper_type', 'pika') == 'pika':
                se = gc.get('gripper_serial')
                g = PikaGripperController(
                    shm_manager=shm_manager,
                    serial_path=se,
                    receive_latency=gc.get('gripper_obs_latency', 0.01),
                    use_meters=True
                )
            else:
                g = WSGController(
                    shm_manager=shm_manager,
                    hostname=gc.get('gripper_ip', '127.0.0.1'),
                    port=gc.get('gripper_port', 1000),
                    receive_latency=gc.get('gripper_obs_latency', 0.01),
                    use_meters=True
                )
            grippers.append(g)
        for c in controllers:
            c.start(wait=True)
        for g in grippers:
            g.start(wait=True)
        start_time = time.time() + 0.5
        if cameras_config:
            dev_paths = cameras_config.get('dev_video_paths', [])
            resolution = cameras_config.get('resolution', (1280,720))
            capture_fps = cameras_config.get('capture_fps', 30)
            multi_cam = MultiUvcCamera(
                dev_video_paths=dev_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                receive_latency=0.0
            )
            multi_cam.start(wait=True, put_start_time=start_time)
        ready = all(c.is_ready for c in controllers)
        gready = all(g.is_ready for g in grippers) if grippers else True
        time.sleep(0.5)
        print(f"ready={ready and gready}")
        
        states = [c.get_state() for c in controllers]
        for i,s in enumerate(states):
            pose = s.get('ActualTCPPose', np.zeros(6))
            print(f"robot{i}_pose={pose}")
        if grippers:
            gstates = [g.get_state() for g in grippers]
            for i,gs in enumerate(gstates):
                print(f"gripper{i}_pos={gs.get('gripper_position', 0.0)}")
        if multi_cam and save_video:
            multi_cam.start_recording(video_dir, start_time)
        if test_move:
            axis_idx = {'x':0,'y':1,'z':2}[move_axis]
            poses = []
            for c in controllers:
                s = c.get_state()
                p = np.array(s.get('ActualTCPPose', np.zeros(6)), dtype=np.float64)
                poses.append(p)
            t0 = time.time() + 0.5
            for i,c in enumerate(controllers):
                p = poses[i].copy()
                p[axis_idx] = p[axis_idx] + amp
                c.schedule_waypoint(p, t0)
            if grippers:
                targets = [t for t in [v.strip() for v in gripper_targets.split(',')] if t]
                if targets:
                    tcur = t0
                    for tgt in targets:
                        val = float(tgt)
                        for g in grippers:
                            g.schedule_waypoint(val, tcur)
                        tcur += gripper_delay
            end_t = time.time() + duration
            while time.time() < end_t:
                if multi_cam:
                    vis = multi_cam.get_vis()
                    if 'rgb' in vis:
                        for i in range(vis['rgb'].shape[0]):
                            frame = vis['rgb'][i,0]
                            cv2.imshow(f'cam{i}', frame[..., ::-1])
                    cv2.waitKey(1)
                time.sleep(0.05)
        else:
            t_end = time.time() + hold
            while time.time() < t_end:
                for i,c in enumerate(controllers):
                    s = c.get_state()
                    pose = s.get('ActualTCPPose', np.zeros(6))
                    print(f"robot{i}_pose={pose}")
                if grippers:
                    for i,g in enumerate(grippers):
                        gs = g.get_state()
                        print(f"gripper{i}_pos={gs.get('gripper_position', 0.0)}")
                if multi_cam:
                    vis = multi_cam.get_vis()
                    if 'rgb' in vis:
                        for i in range(vis['rgb'].shape[0]):
                            frame = vis['rgb'][i,0]
                            cv2.imshow(f'cam{i}', frame[..., ::-1])
                    cv2.waitKey(1)
                time.sleep(1.0)
        for c in controllers:
            c.stop(wait=True)
        for g in grippers:
            g.stop(wait=True)
        if multi_cam:
            if save_video:
                multi_cam.stop_recording()
            multi_cam.stop(wait=True)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
