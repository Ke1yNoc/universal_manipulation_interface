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
from umi.real_world.video_recorder import VideoRecorder

@click.command()
@click.option('--robot_config', '-rc', required=True, help='Path to robot config yaml')
@click.option('--frequency', '-f', default=10, type=float)
@click.option('--test_move', is_flag=True, default=False)
@click.option('--move_axis', default='z', type=click.Choice(['x','y','z']))
@click.option('--amp', default=0.1, type=float)
@click.option('--duration', default=2.0, type=float)
@click.option('--hold', default=5.0, type=float, help='Hold time for idle sampling when not moving')
@click.option('--gripper_targets', default='', type=str)
@click.option('--gripper_delay', default=1.0, type=float)
@click.option('--save_video', is_flag=True, default=False)
@click.option('--video_dir', default='videos', type=str)
def main(robot_config, frequency, test_move, move_axis, amp, duration, hold, gripper_targets, gripper_delay, save_video, video_dir):
    
    print(f"Loaded config from {robot_config}")
    print(f"Save Video: {save_video}")
    print(f"Video Dir: {os.path.abspath(video_dir)}")
    
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data.get('grippers', [])
    cameras_config = robot_config_data.get('cameras', None)
    
    print(f"Robots: {len(robots_config)}")
    print(f"Grippers: {len(grippers_config)}")
    if cameras_config:
        print(f"Cameras: {len(cameras_config.get('dev_video_paths', []))}")

    with SharedMemoryManager() as shm_manager:
        controllers = []
        grippers = []
        multi_cam = None
        
        # Initialize Robot Controllers
        for i, rc in enumerate(robots_config):
            assert rc['robot_type'].startswith('piper')
            print(f"Connecting to Piper {i} at {rc.get('piper_can', 'can0')}...")
            ctrl = PiperInterpolationController(
                shm_manager=shm_manager,
                piper_can=rc.get('piper_can', 'can0'),
                frequency=rc.get('frequency', 200), # Match BimanualUmiEnv default
                lookahead_time=0.1,
                gain_pos=2.0,
                gain_rot=6.0,
                tool_offset=[0,0,rc.get('tcp_offset', 0.0)],
                receive_latency=rc.get('robot_obs_latency', 0.0001),
                verbose=True
            )
            controllers.append(ctrl)
        
        # Initialize Gripper Controllers
        for idx, gc in enumerate(grippers_config):
            print(f"Connecting to Gripper {idx}...")
            if gc.get('gripper_type', 'pika') == 'pika' or 'gripper_serial' in gc:
                se = gc.get('gripper_serial')
                print(f"  Type: Pika, Serial: {se}")
                g = PikaGripperController(
                    shm_manager=shm_manager,
                    serial_path=se,
                    receive_latency=gc.get('gripper_obs_latency', 0.01),
                    use_meters=True,
                    verbose=True
                )
            else:
                ip = gc.get('gripper_ip', '127.0.0.1')
                print(f"  Type: WSG, IP: {ip}")
                g = WSGController(
                    shm_manager=shm_manager,
                    hostname=ip,
                    port=gc.get('gripper_port', 1000),
                    receive_latency=gc.get('gripper_obs_latency', 0.01),
                    use_meters=True
                )
            grippers.append(g)
            
        # Start Controllers
        print("Starting controllers...")
        for c in controllers:
            c.start(wait=True)
        for g in grippers:
            g.start(wait=True)
            
        # Wait for controllers to stabilize
        print("Waiting for controllers to stabilize...")
        time.sleep(2.0)
            
        start_time = time.time() + 0.5
        
        # Initialize Cameras
        if cameras_config:
            print("Connecting to cameras...")
            dev_paths = cameras_config.get('dev_video_paths', [])
            resolution = cameras_config.get('resolution', (1280,720))
            capture_fps = cameras_config.get('capture_fps', 30)
            
            # Handle per-camera FPS settings for video recorders
            # If capture_fps is a list, create recorders with matching FPS
            # If it's a single value, MultiUvcCamera will handle it
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
                # Single FPS value - create one recorder and let MultiUvcCamera duplicate it
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
                use_mjpeg=True,  # Enable MJPEG for Pika cameras (better FPS)
                video_recorder=recorders,
                verbose=True
            )
            multi_cam.start(wait=True, put_start_time=start_time)
            
        ready = all(c.is_ready for c in controllers)
        gready = all(g.is_ready for g in grippers) if grippers else True
        time.sleep(0.5)
        print(f"System Ready: Robots={ready}, Grippers={gready}")
        
        # Capture Initial State for Safe Termination
        initial_poses = []
        for c in controllers:
            s = c.get_state()
            p = np.array(s.get('ActualTCPPose', np.zeros(6)), dtype=np.float64)
            initial_poses.append(p)
            print(f"Initial Pose R{controllers.index(c)}: {p[:3]}")
        print("Initial poses captured. Robots will return here upon termination.")

        # Verify Data Liveness
        print("Verifying data reception...")
        for i, c in enumerate(controllers):
            state = c.get_state()
            ts = state.get('robot_receive_timestamp', 0)
            print(f"Robot {i} Timestamp: {ts:.3f} (Latency: {time.time()-ts:.3f}s)")
            if time.time() - ts > 1.0:
                print(f"WARNING: Robot {i} data is stale!")
                
        for i, g in enumerate(grippers):
            state = g.get_state()
            ts = state.get('gripper_receive_timestamp', 0)
            print(f"Gripper {i} Timestamp: {ts:.3f} (Latency: {time.time()-ts:.3f}s)")
             
        if multi_cam and save_video:
            print(f"Recording video to {video_dir}...")
            # Ensure directory exists
            os.makedirs(video_dir, exist_ok=True)
            # Use current time as start time for recording to ensure we capture incoming frames
            rec_start_time = time.time()
            multi_cam.start_recording(video_dir, rec_start_time)
            
        try:
            if test_move:
                print(f"Starting Multi-Axis Test Move (Amp: {amp}m)...")
                t0 = time.time() + 1.0 # Give a bit more buffer
                
                # 1. Gripper Test
                if grippers:
                    print("Testing grippers: Open -> Close -> Open")
                    for g in grippers: g.schedule_waypoint(0.08, t0)
                    time.sleep(1.0)
                    for g in grippers: g.schedule_waypoint(0.0, t0 + 1.0)
                    time.sleep(1.0)
                    for g in grippers: g.schedule_waypoint(0.08, t0 + 2.0)
                    time.sleep(1.0)
                    t0 = time.time() + 0.5
                
                # 2. Robot Multi-Axis Test
                axes = ['x', 'y', 'z']
                for axis in axes:
                    print(f"Testing Axis: {axis.upper()}")
                    axis_idx = {'x':0,'y':1,'z':2}[axis]
                    
                    # Move +Amp
                    print(f"  Moving +{amp}m...")
                    for i,c in enumerate(controllers):
                        p = initial_poses[i].copy()
                        p[axis_idx] += amp
                        # Controller handles time.monotonic() conversion internally
                        c.schedule_waypoint(p, t0 + 1.0)
                    time.sleep(2.0)
                    
                    # Move -Amp (Back to start)
                    print(f"  Moving back...")
                    for i,c in enumerate(controllers):
                        p = initial_poses[i].copy()
                        # Controller handles time.monotonic() conversion internally
                        c.schedule_waypoint(p, t0 + 3.0)
                    time.sleep(2.0)
                    
                    t0 = time.time() + 0.5 # reset base time for next axis
                
                print("Movement test complete.")

            # Monitoring Loop
            print(f"Monitoring state for {duration if test_move else hold} seconds...")
            end_t = time.time() + (duration if test_move else hold)
            last_fps_print = time.time()
            frame_count = 0
            
            while time.time() < end_t:
                if multi_cam:
                    # Check recorder liveness
                    if save_video:
                        for i, cam in enumerate(multi_cam.cameras.values()):
                            if not cam.video_recorder.is_alive():
                                print(f"WARNING: VideoRecorder {i} is DEAD!")
                            elif not cam.video_recorder.is_ready():
                                print(f"WARNING: VideoRecorder {i} is NOT READY!")
                            
                    vis = multi_cam.get_vis()
                    if 'rgb' in vis:
                        imgs = [vis['rgb'][i,0] for i in range(vis['rgb'].shape[0])]
                        combined_img = np.concatenate(imgs, axis=1)
                        cv2.putText(combined_img, f"Time: {time.time():.1f}", (10,30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.imshow('Camera Stream', combined_img[..., ::-1])
                        if int(time.time()) % 5 == 0:
                            cv2.imwrite(f"snapshot_{int(time.time())}.jpg", combined_img[..., ::-1])
                        
                        # Count frames for FPS calculation
                        frame_count += 1
                        if time.time() - last_fps_print >= 2.0:
                            fps = frame_count / (time.time() - last_fps_print)
                            print(f"Display FPS: {fps:.1f}")
                            frame_count = 0
                            last_fps_print = time.time()
                            
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if int(time.time() * 10) % 20 == 0:
                    for i,c in enumerate(controllers):
                        s = c.get_state()
                        pose = s.get('ActualTCPPose', np.zeros(6))
                        # print(f"R{i}: {pose[:3]}")
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\nTerminating... Returning robots to initial poses.")
            try:
                # Return robots to initial poses
                for i, c in enumerate(controllers):
                    if c.is_alive():
                        # Controller handles time.monotonic() conversion internally
                        c.schedule_waypoint(initial_poses[i], time.time() + 2.0)
                
                time.sleep(3.0)
            except Exception as e:
                print(f"Error during termination: {e}")
                
            for c in controllers:
                c.stop(wait=False)
            for g in grippers:
                g.stop(wait=False)
            if multi_cam:
                if save_video:
                    multi_cam.stop_recording()
                multi_cam.stop(wait=True)
            cv2.destroyAllWindows()
            print("Done.")

if __name__ == '__main__':
    main()
