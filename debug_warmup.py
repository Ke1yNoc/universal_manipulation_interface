import sys
import os
import time
import pathlib
import yaml
import click
import dill
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.common.precise_sleep import precise_wait
from umi.real_world.real_inference_util import (
    get_real_umi_obs_dict, get_real_umi_action, get_real_obs_resolution)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
def main(input, robot_config):
    # Load config
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    # robots_config = robot_config_data['robots']
    # grippers_config = robot_config_data['grippers']
    robots_config = [] # Disable robots
    grippers_config = [] # Disable grippers
    # camera_config = robot_config_data.get('cameras', None)
    camera_config = None # Disable cameras
    
    # Load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    
    # Monkeypatch optimal_row_cols to handle 0 cameras
    import umi.real_world.bimanual_umi_env
    original_optimal_row_cols = umi.real_world.bimanual_umi_env.optimal_row_cols
    def patched_optimal_row_cols(n_cameras, **kwargs):
        if n_cameras == 0:
            return 0, 0, 0, 0
        return original_optimal_row_cols(n_cameras, **kwargs)
    umi.real_world.bimanual_umi_env.optimal_row_cols = patched_optimal_row_cols

    print("Initializing environment (NO HARDWARE)...")
    with SharedMemoryManager() as shm_manager:
        with BimanualUmiEnv(
            output_dir='debug_output',
            robots_config=robots_config,
            grippers_config=grippers_config,
            frequency=30,
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_config=camera_config,
            camera_paths=[], # Explicitly disable auto-detection
            camera_reorder=[0, 1],
            init_joints=False,
            enable_multi_cam_vis=False, # Disable vis for minimal test
            shm_manager=shm_manager
        ) as env:
            
            print("Waiting for environment...")
            time.sleep(2.0)
            while not env.is_ready:
                time.sleep(0.1)
            print("Environment ready!")
            
            print("Creating model...")
            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            
            device = torch.device('cuda')
            policy.eval().to(device)
            
            print("Getting observation (SKIPPED - USING DUMMY)...")
            # obs = env.get_obs()
            obs = dict() # Start with empty dict
            
            # Inject dummy robot data since robots are disabled
            print("Injecting dummy robot data...")
            for robot_id in range(2): # Assuming 2 robots
                # Shape must be (T, D) where T=2 (robot_obs_horizon)
                obs[f'robot{robot_id}_eef_pos'] = np.zeros((2, 3), dtype=np.float64)
                obs[f'robot{robot_id}_eef_rot_axis_angle'] = np.zeros((2, 3), dtype=np.float64)
                obs[f'robot{robot_id}_gripper_width'] = np.zeros((2, 1), dtype=np.float64)
            
            # Inject dummy camera data since cameras are disabled
            print("Injecting dummy camera data...")
            for cam_id in range(2):
                # Shape must be (T, H, W, C) where T=2 (camera_obs_horizon)
                obs[f'camera{cam_id}_rgb'] = np.zeros((2, 224, 224, 3), dtype=np.float32)
            
            # Debug: Print observation shapes and memory usage
            print("\n=== Observation Debug Info ===")
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, size={value.nbytes / 1024 / 1024:.2f} MB")
                else:
                    print(f"  {key}: {type(value)}")
            
            total_obs_size = sum(v.nbytes for v in obs.values() if isinstance(v, np.ndarray))
            print(f"  Total observation size: {total_obs_size / 1024 / 1024:.2f} MB")
            print("=" * 30 + "\n")
            
            episode_start_pose = list()
            for robot_id in range(2): # Hardcoded for 2 robots
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            
            print("Running inference...")
            with torch.no_grad():
                policy.reset()
                
                print("Creating observation dict...")
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
                    tx_robot1_robot0=np.array(robot_config_data['tx_left_right']),
                    episode_start_pose=episode_start_pose)
                
                # Debug: Print processed observation shapes
                print("\n=== Processed Observation Debug Info ===")
                for key, value in obs_dict_np.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, size={value.nbytes / 1024 / 1024:.2f} MB")
                total_processed_size = sum(v.nbytes for v in obs_dict_np.values() if isinstance(v, np.ndarray))
                print(f"  Total processed size: {total_processed_size / 1024 / 1024:.2f} MB")
                print("=" * 40 + "\n")
                
                # Filter keys to match model expectations
                expected_keys = list(cfg.task.shape_meta.obs.keys())
                obs_dict_np = {k: v for k, v in obs_dict_np.items() if k in expected_keys}
                print(f"Filtered observation keys: {list(obs_dict_np.keys())}")

                print("Moving to GPU...")
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                print("Calling predict_action...")
                import sys
                sys.stdout.flush()
                
                result = policy.predict_action(obs_dict)
                print("✓ Inference success!")

if __name__ == '__main__':
    main()
