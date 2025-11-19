#!/usr/bin/env python3
"""
Test BimanualUmiEnv with Piper robots and Pika grippers
This verifies the environment setup without needing a trained model or SpaceMouse
"""

import time
import click
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.bimanual_umi_env import BimanualUmiEnv

@click.command()
@click.option('--output', '-o', default='data_local/test_bimanual_env', help='Output directory for test data')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
@click.option('--duration', '-d', default=5.0, type=float, help='Test duration in seconds')
@click.option('--test_move', is_flag=True, default=False, help='Test robot movement')
def main(output, frequency, duration, test_move):
    """
    Test BimanualUmiEnv with Piper robots
    """
    
    import yaml
    import os
    
    # Load robot config from yaml
    config_path = os.path.join(os.path.dirname(__file__), 'example/eval_robots_config_piper.yaml')
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    robots_config = full_config['robots']
    grippers_config = full_config['grippers']
    camera_config = full_config.get('cameras', None)
    
    print("="*60)
    print("Testing BimanualUmiEnv with Piper Setup")
    print("="*60)
    print(f"Output: {output}")
    print(f"Frequency: {frequency} Hz")
    print(f"Duration: {duration}s")
    print(f"Test Move: {test_move}")
    print("="*60)
    
    with SharedMemoryManager() as shm_manager:
        with BimanualUmiEnv(
            output_dir=output,
            robots_config=robots_config,
            grippers_config=grippers_config,
            frequency=frequency,
            obs_image_resolution=(224, 224),
            obs_float32=True,
            camera_config=camera_config,
            camera_reorder=None,
            init_joints=False,
            enable_multi_cam_vis=True,
            # Observation horizons
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # Action limits
            max_pos_speed=0.5,
            max_rot_speed=1.5,
            shm_manager=shm_manager
        ) as env:
            
            print("\n✓ BimanualUmiEnv created successfully!")
            print("✓ Waiting for environment to be ready...")
            
            # Wait for ready
            time.sleep(2.0)
            
            if not env.is_ready:
                print("✗ Environment not ready!")
                return
            
            print("✓ Environment is ready!")
            
            # Get initial observation
            print("\n✓ Getting initial observation...")
            obs = env.get_obs()
            
            print(f"  Observation keys: {list(obs.keys())}")
            print(f"  Camera 0 shape: {obs['camera0_rgb'].shape}")
            print(f"  Camera 1 shape: {obs['camera1_rgb'].shape}")
            print(f"  Robot 0 EEF pos: {obs['robot0_eef_pos'][-1]}")
            print(f"  Robot 1 EEF pos: {obs['robot1_eef_pos'][-1]}")
            print(f"  Robot 0 gripper: {obs['robot0_gripper_width'][-1]}")
            print(f"  Robot 1 gripper: {obs['robot1_gripper_width'][-1]}")
            
            # Get robot states
            print("\n✓ Getting robot states...")
            robot_states = env.get_robot_state()
            for i, state in enumerate(robot_states):
                pose = state.get('ActualTCPPose', np.zeros(6))
                print(f"  Robot {i} TCP Pose: {pose[:3]}")
            
            # Get gripper states
            print("\n✓ Getting gripper states...")
            gripper_states = env.get_gripper_state()
            for i, state in enumerate(gripper_states):
                pos = state.get('gripper_position', 0)
                print(f"  Gripper {i} Position: {pos:.4f}m")
            
            if test_move:
                print("\n✓ Testing robot movement...")
                
                # Get initial poses
                initial_poses = []
                for state in robot_states:
                    initial_poses.append(state['TargetTCPPose'])
                
                # Test gripper movement
                print("  Testing grippers: Open -> Close -> Open")
                t0 = time.time() + 1.0
                
                # Create actions: [robot0_pose(6), robot0_gripper(1), robot1_pose(6), robot1_gripper(1)]
                action_open = np.concatenate([
                    initial_poses[0], [0.08],  # Robot 0
                    initial_poses[1], [0.08]   # Robot 1
                ])
                action_close = np.concatenate([
                    initial_poses[0], [0.0],
                    initial_poses[1], [0.0]
                ])
                
                env.exec_actions(
                    actions=[action_open],
                    timestamps=[t0],
                    compensate_latency=False
                )
                time.sleep(1.5)
                
                env.exec_actions(
                    actions=[action_close],
                    timestamps=[time.time() + 0.5],
                    compensate_latency=False
                )
                time.sleep(1.5)
                
                env.exec_actions(
                    actions=[action_open],
                    timestamps=[time.time() + 0.5],
                    compensate_latency=False
                )
                time.sleep(1.5)
                
                print("  ✓ Gripper test complete")
                
                # Test small robot movement
                print("  Testing robot movement: Small Z-axis motion")
                for i, initial_pose in enumerate(initial_poses):
                    # Move up 5cm
                    target_pose_up = initial_pose.copy()
                    target_pose_up[2] += 0.05
                    
                    action_up = np.zeros(14)
                    action_up[i*7:i*7+6] = target_pose_up
                    action_up[i*7+6] = 0.08  # Open gripper
                    # Keep other robot at initial pose
                    other_i = 1 - i
                    action_up[other_i*7:other_i*7+6] = initial_poses[other_i]
                    action_up[other_i*7+6] = 0.08
                    
                    env.exec_actions(
                        actions=[action_up],
                        timestamps=[time.time() + 0.5],
                        compensate_latency=False
                    )
                    time.sleep(2.0)
                    
                    # Move back down
                    action_down = np.zeros(14)
                    action_down[i*7:i*7+6] = initial_pose
                    action_down[i*7+6] = 0.08
                    action_down[other_i*7:other_i*7+6] = initial_poses[other_i]
                    action_down[other_i*7+6] = 0.08
                    
                    env.exec_actions(
                        actions=[action_down],
                        timestamps=[time.time() + 0.5],
                        compensate_latency=False
                    )
                    time.sleep(2.0)
                    
                    print(f"  ✓ Robot {i} movement test complete")
            
            # Monitor for duration
            print(f"\n✓ Monitoring for {duration} seconds...")
            print("  (Press Ctrl+C to stop early)")
            
            start_time = time.time()
            try:
                while time.time() - start_time < duration:
                    obs = env.get_obs()
                    
                    # Print status every 2 seconds
                    if int(time.time() - start_time) % 2 == 0:
                        print(f"  Time: {time.time() - start_time:.1f}s | "
                              f"R0: {obs['robot0_eef_pos'][-1][:3]} | "
                              f"R1: {obs['robot1_eef_pos'][-1][:3]}")
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n  Interrupted by user")
            
            print("\n" + "="*60)
            print("✓ BimanualUmiEnv test completed successfully!")
            print("="*60)
            print("\nYour Piper setup is ready for:")
            print("  • Data collection (demo scripts)")
            print("  • Policy evaluation (eval scripts)")
            print("  • Training (after collecting demos)")

if __name__ == '__main__':
    main()
