#!/usr/bin/env python3
"""
Test script for PiperInterpolationControllerIK

This script tests the IK-based Piper controller with various movements
and monitors IK solver performance, joint angles, and TCP pose accuracy.

Usage:
    python test_piper_ik_controller.py --robot_config config/piper_config.yaml --test_move
    python test_piper_ik_controller.py --robot_config config/piper_config.yaml --test_move --move_axis z --amp 0.05
"""

import os
import time
import click
import yaml
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import cv2
from umi.real_world.piper_interpolation_controller_ik import PiperInterpolationControllerIK
from umi.real_world.pika_gripper_controller import PikaGripperController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder


@click.command()
@click.option('--robot_config', '-rc', required=True, help='Path to robot config yaml')
@click.option('--frequency', '-f', default=150, type=float, help='Control frequency (Hz)')
@click.option('--test_move', is_flag=True, default=False, help='Execute test movements')
@click.option('--move_axis', default='z', type=click.Choice(['x','y','z']), help='Axis to test')
@click.option('--amp', default=0.05, type=float, help='Movement amplitude (meters)')
@click.option('--duration', default=2.0, type=float, help='Duration per movement')
@click.option('--hold', default=5.0, type=float, help='Hold time for idle sampling when not moving')
@click.option('--gripper_targets', default='', type=str, help='Comma-separated gripper positions')
@click.option('--gripper_delay', default=1.0, type=float, help='Delay between gripper commands')
@click.option('--save_video', is_flag=True, default=False, help='Save camera video')
@click.option('--video_dir', default='videos', type=str, help='Video output directory')
@click.option('--monitor_ik', is_flag=True, default=True, help='Monitor IK solver performance')
@click.option('--verbose', is_flag=True, default=False, help='Verbose output')
def main(robot_config, frequency, test_move, move_axis, amp, duration, hold, 
         gripper_targets, gripper_delay, save_video, video_dir, monitor_ik, verbose):
    
    print("=" * 80)
    print("Piper IK Controller Test Script")
    print("=" * 80)
    print(f"Config: {robot_config}")
    print(f"Frequency: {frequency} Hz")
    print(f"Test Move: {test_move}")
    print(f"Save Video: {save_video}")
    if save_video:
        print(f"Video Dir: {os.path.abspath(video_dir)}")
    print("=" * 80)
    
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data.get('grippers', [])
    cameras_config = robot_config_data.get('cameras', None)
    
    print(f"\nSystem Configuration:")
    print(f"  Robots: {len(robots_config)}")
    print(f"  Grippers: {len(grippers_config)}")
    if cameras_config:
        print(f"  Cameras: {len(cameras_config.get('dev_video_paths', []))}")

    with SharedMemoryManager() as shm_manager:
        controllers = []
        grippers = []
        multi_cam = None
        
        # Initialize Robot Controllers with IK
        print("\n" + "=" * 80)
        print("Initializing IK Controllers...")
        print("=" * 80)
        for i, rc in enumerate(robots_config):
            assert rc['robot_type'].startswith('piper'), f"Only Piper robots supported, got {rc['robot_type']}"
            piper_can = rc.get('piper_can', 'can0')
            tcp_offset = rc.get('tcp_offset', 0.1773)
            
            print(f"\nRobot {i}:")
            print(f"  CAN Interface: {piper_can}")
            print(f"  TCP Offset: {tcp_offset}m")
            print(f"  Control Frequency: {frequency} Hz")
            
            ctrl = PiperInterpolationControllerIK(
                shm_manager=shm_manager,
                piper_can=piper_can,
                frequency=int(frequency),
                lookahead_time=0.1,
                gain_pos=0.25,  # IK controller uses different gains
                gain_rot=0.16,
                tool_offset=[0, 0, tcp_offset],
                receive_latency=rc.get('robot_obs_latency', 0.0001),
                verbose=verbose,
                speed_percent=rc.get('speed_percent', 100),
                go_zero_on_start=rc.get('go_zero_on_start', True)
            )
            controllers.append(ctrl)
        
        # Initialize Gripper Controllers
        if grippers_config:
            print("\n" + "=" * 80)
            print("Initializing Grippers...")
            print("=" * 80)
            for idx, gc in enumerate(grippers_config):
                print(f"\nGripper {idx}:")
                if gc.get('gripper_type', 'pika') == 'pika' or 'gripper_serial' in gc:
                    se = gc.get('gripper_serial')
                    print(f"  Type: Pika")
                    print(f"  Serial: {se}")
                    g = PikaGripperController(
                        shm_manager=shm_manager,
                        serial_path=se,
                        receive_latency=gc.get('gripper_obs_latency', 0.01),
                        use_meters=True,
                        verbose=verbose
                    )
                else:
                    ip = gc.get('gripper_ip', '127.0.0.1')
                    port = gc.get('gripper_port', 1000)
                    print(f"  Type: WSG")
                    print(f"  IP: {ip}:{port}")
                    g = WSGController(
                        shm_manager=shm_manager,
                        hostname=ip,
                        port=port,
                        receive_latency=gc.get('gripper_obs_latency', 0.01),
                        use_meters=True
                    )
                grippers.append(g)
        
        # Start Controllers
        print("\n" + "=" * 80)
        print("Starting Controllers...")
        print("=" * 80)
        for i, c in enumerate(controllers):
            print(f"  Starting Robot {i}...")
            c.start(wait=True)
        for i, g in enumerate(grippers):
            print(f"  Starting Gripper {i}...")
            g.start(wait=True)
        
        # Wait for stabilization
        print("\nWaiting for controllers to stabilize...")
        time.sleep(2.0)
        
        start_time = time.time() + 0.5
        
        # Initialize Cameras
        if cameras_config:
            print("\n" + "=" * 80)
            print("Initializing Cameras...")
            print("=" * 80)
            dev_paths = cameras_config.get('dev_video_paths', [])
            resolution = cameras_config.get('resolution', (1280, 720))
            capture_fps = cameras_config.get('capture_fps', 30)
            
            print(f"  Devices: {dev_paths}")
            print(f"  Resolution: {resolution}")
            print(f"  FPS: {capture_fps}")
            
            # Handle per-camera FPS settings
            if isinstance(capture_fps, list):
                recorders = []
                for fps in capture_fps:
                    recorders.append(VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='bgr24',
                        crf=22
                    ))
            else:
                recorders = VideoRecorder.create_h264(
                    fps=capture_fps,
                    codec='h264',
                    input_pix_fmt='bgr24',
                    crf=22
                )
            
            multi_cam = MultiUvcCamera(
                dev_video_paths=dev_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                receive_latency=0.0,
                use_mjpeg=True,
                video_recorder=recorders,
                verbose=verbose
            )
            multi_cam.start(wait=True, put_start_time=start_time)
        
        # Check readiness
        ready = all(c.is_ready for c in controllers)
        gready = all(g.is_ready for g in grippers) if grippers else True
        time.sleep(0.5)
        
        print("\n" + "=" * 80)
        print(f"System Status: Robots={'READY' if ready else 'NOT READY'}, Grippers={'READY' if gready else 'NOT READY'}")
        print("=" * 80)
        
        # Capture Initial State
        print("\nCapturing initial robot states...")
        initial_poses = []
        initial_joints = []
        for i, c in enumerate(controllers):
            state = c.get_state()
            tcp_pose = np.array(state.get('ActualTCPPose', np.zeros(6)), dtype=np.float64)
            joint_angles = np.array(state.get('ActualQ', np.zeros(6)), dtype=np.float64)
            initial_poses.append(tcp_pose)
            initial_joints.append(joint_angles)
            print(f"  Robot {i}:")
            print(f"    TCP Pose: [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}]")
            print(f"    Joint Angles (deg): {np.rad2deg(joint_angles).round(1)}")
        
        # Verify Data Liveness
        print("\nVerifying data reception...")
        for i, c in enumerate(controllers):
            state = c.get_state()
            ts = state.get('robot_receive_timestamp', 0)
            latency = time.time() - ts
            print(f"  Robot {i}: Timestamp={ts:.3f}, Latency={latency:.3f}s", end="")
            if latency > 1.0:
                print(" [WARNING: STALE DATA]")
            else:
                print(" [OK]")
        
        for i, g in enumerate(grippers):
            state = g.get_state()
            ts = state.get('gripper_receive_timestamp', 0)
            latency = time.time() - ts
            print(f"  Gripper {i}: Timestamp={ts:.3f}, Latency={latency:.3f}s", end="")
            if latency > 1.0:
                print(" [WARNING: STALE DATA]")
            else:
                print(" [OK]")
        
        # Start video recording
        if multi_cam and save_video:
            print(f"\n{'=' * 80}")
            print(f"Starting video recording to {video_dir}...")
            print("=" * 80)
            os.makedirs(video_dir, exist_ok=True)
            rec_start_time = time.time()
            multi_cam.start_recording(video_dir, rec_start_time)
        
        # IK Performance Monitoring
        ik_stats = {
            'total_commands': 0,
            'ik_failures': 0,
            'max_joint_error': 0.0,
            'max_tcp_error': 0.0
        }
        
        try:
            if test_move:
                print("\n" + "=" * 80)
                print(f"Starting Movement Tests (Amplitude: {amp}m)")
                print("=" * 80)
                t0 = time.time() + 1.0
                
                # 1. Gripper Test
                if grippers:
                    print("\n[Test 1/4] Gripper Test: Open → Close → Open")
                    print("-" * 80)
                    for i, g in enumerate(grippers):
                        print(f"  Gripper {i}: Opening to 0.08m")
                        g.schedule_waypoint(0.08, t0)
                    time.sleep(1.5)
                    
                    for i, g in enumerate(grippers):
                        print(f"  Gripper {i}: Closing to 0.0m")
                        g.schedule_waypoint(0.0, t0 + 1.5)
                    time.sleep(1.5)
                    
                    for i, g in enumerate(grippers):
                        print(f"  Gripper {i}: Opening to 0.08m")
                        g.schedule_waypoint(0.08, t0 + 3.0)
                    time.sleep(1.5)
                    
                    t0 = time.time() + 0.5
                    print("  ✓ Gripper test complete")
                
                # 2. Multi-Axis Robot Test
                axes = ['x', 'y', 'z']
                for test_num, axis in enumerate(axes, start=2):
                    print(f"\n[Test {test_num}/4] Testing {axis.upper()}-Axis Movement")
                    print("-" * 80)
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    
                    # Move +Amp
                    print(f"  Moving +{amp}m along {axis.upper()}-axis...")
                    for i, c in enumerate(controllers):
                        target_pose = initial_poses[i].copy()
                        target_pose[axis_idx] += amp
                        c.schedule_waypoint(target_pose, t0 + 1.0)
                        print(f"    Robot {i}: Target TCP = [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
                    
                    time.sleep(2.0)
                    
                    # Check achieved position
                    if monitor_ik:
                        print(f"  Checking achieved position...")
                        for i, c in enumerate(controllers):
                            state = c.get_state()
                            actual_tcp = state.get('ActualTCPPose', np.zeros(6))
                            target_tcp = initial_poses[i].copy()
                            target_tcp[axis_idx] += amp
                            tcp_error = np.linalg.norm(actual_tcp[:3] - target_tcp[:3])
                            ik_stats['max_tcp_error'] = max(ik_stats['max_tcp_error'], tcp_error)
                            print(f"    Robot {i}: TCP Error = {tcp_error*1000:.2f}mm")
                    
                    # Move back to initial
                    print(f"  Returning to initial position...")
                    for i, c in enumerate(controllers):
                        c.schedule_waypoint(initial_poses[i], t0 + 3.0)
                    
                    time.sleep(2.0)
                    
                    # Check return accuracy
                    if monitor_ik:
                        print(f"  Checking return accuracy...")
                        for i, c in enumerate(controllers):
                            state = c.get_state()
                            actual_tcp = state.get('ActualTCPPose', np.zeros(6))
                            tcp_error = np.linalg.norm(actual_tcp[:3] - initial_poses[i][:3])
                            print(f"    Robot {i}: Return Error = {tcp_error*1000:.2f}mm")
                    
                    t0 = time.time() + 0.5
                    print(f"  ✓ {axis.upper()}-axis test complete")
                
                print("\n" + "=" * 80)
                print("All movement tests complete!")
                print("=" * 80)
            
            # Monitoring Loop
            print(f"\n{'=' * 80}")
            print(f"Monitoring for {duration if test_move else hold} seconds...")
            print("=" * 80)
            end_t = time.time() + (duration if test_move else hold)
            last_fps_print = time.time()
            last_ik_print = time.time()
            frame_count = 0
            
            while time.time() < end_t:
                # Camera visualization
                if multi_cam:
                    if save_video:
                        for i, cam in enumerate(multi_cam.cameras.values()):
                            if not cam.video_recorder.is_alive():
                                print(f"WARNING: VideoRecorder {i} is DEAD!")
                            elif not cam.video_recorder.is_ready():
                                print(f"WARNING: VideoRecorder {i} is NOT READY!")
                    
                    vis = multi_cam.get_vis()
                    if 'rgb' in vis:
                        imgs = [vis['rgb'][i, 0] for i in range(vis['rgb'].shape[0])]
                        combined_img = np.concatenate(imgs, axis=1)
                        
                        # Add overlay text
                        cv2.putText(combined_img, f"Time: {time.time():.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(combined_img, f"IK Mode: Active", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        cv2.imshow('Camera Stream', combined_img[..., ::-1])
                        
                        # Save periodic snapshots
                        if int(time.time()) % 5 == 0:
                            snapshot_path = f"snapshot_ik_{int(time.time())}.jpg"
                            cv2.imwrite(snapshot_path, combined_img[..., ::-1])
                        
                        frame_count += 1
                        if time.time() - last_fps_print >= 2.0:
                            fps = frame_count / (time.time() - last_fps_print)
                            print(f"Display FPS: {fps:.1f}")
                            frame_count = 0
                            last_fps_print = time.time()
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                # IK Performance Monitoring
                if monitor_ik and (time.time() - last_ik_print >= 2.0):
                    print("\nIK Controller Status:")
                    for i, c in enumerate(controllers):
                        state = c.get_state()
                        actual_tcp = state.get('ActualTCPPose', np.zeros(6))
                        target_tcp = state.get('TargetTCPPose', np.zeros(6))
                        actual_q = state.get('ActualQ', np.zeros(6))
                        actual_qd = state.get('ActualQd', np.zeros(6))
                        
                        tcp_error = np.linalg.norm(actual_tcp[:3] - target_tcp[:3])
                        
                        print(f"  Robot {i}:")
                        print(f"    TCP Error: {tcp_error*1000:.2f}mm")
                        print(f"    Joint Vel (deg/s): {np.rad2deg(actual_qd).round(1)}")
                        print(f"    Joint Pos (deg): {np.rad2deg(actual_q).round(1)}")
                    
                    last_ik_print = time.time()
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            print(f"\n\nError during test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "=" * 80)
            print("Shutting Down...")
            print("=" * 80)
            
            # Return to initial poses
            print("\nReturning robots to initial poses...")
            try:
                for i, c in enumerate(controllers):
                    if c.is_alive():
                        c.schedule_waypoint(initial_poses[i], time.time() + 2.0)
                        print(f"  Robot {i}: Returning to initial pose")
                time.sleep(3.0)
            except Exception as e:
                print(f"Error during return: {e}")
            
            # Stop controllers
            print("\nStopping controllers...")
            for i, c in enumerate(controllers):
                print(f"  Stopping Robot {i}...")
                c.stop(wait=False)
            for i, g in enumerate(grippers):
                print(f"  Stopping Gripper {i}...")
                g.stop(wait=False)
            
            # Stop cameras
            if multi_cam:
                if save_video:
                    print("Stopping video recording...")
                    multi_cam.stop_recording()
                print("Stopping cameras...")
                multi_cam.stop(wait=True)
            
            cv2.destroyAllWindows()
            
            # Print IK Statistics
            if monitor_ik:
                print("\n" + "=" * 80)
                print("IK Performance Summary")
                print("=" * 80)
                print(f"  Max TCP Error: {ik_stats['max_tcp_error']*1000:.2f}mm")
                print(f"  Max Joint Error: {ik_stats['max_joint_error']:.4f}rad")
                print("=" * 80)
            
            print("\n✓ Test complete!")


if __name__ == '__main__':
    main()
