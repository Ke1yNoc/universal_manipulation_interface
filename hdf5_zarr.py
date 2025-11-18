#!/usr/bin/env python3
"""
Convert Agilex HDF5 episodes to Zarr dataset format using ReplayBuffer (UMI-style).

This matches the UMI dataset format using diffusion_policy's ReplayBuffer class.

Usage:
    python hdf5_to_zarr_umi.py \
        --dataset_dir /path/to/agilex \
        --output_path /path/to/output/dataset.zarr.zip \
        --compression_level 99 \
        --image_size 224 224 \
        --overwrite
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import h5py
import numpy as np
import zarr
import cv2
import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


def load_episode_data(episode_path: Path, target_size: Tuple[int, int] = None) -> Dict:
    """
    Load data from a single HDF5 episode.
    
    Returns:
        Dictionary with keys matching UMI format:
        - 'robot0_eef_pos': (T, 3) float32
        - 'robot0_eef_rot_axis_angle': (T, 3) float32
        - 'robot0_gripper_width': (T, 1) float32
        - 'robot0_demo_start_pose': (T, 6) float64
        - 'robot0_demo_end_pose': (T, 6) float64
        - 'robot1_eef_pos': (T, 3) float32
        - 'robot1_eef_rot_axis_angle': (T, 3) float32
        - 'robot1_gripper_width': (T, 1) float32
        - 'robot1_demo_start_pose': (T, 6) float64
        - 'robot1_demo_end_pose': (T, 6) float64
        - 'camera0_rgb': (T, H, W, 3) uint8
        - 'camera1_rgb': (T, H, W, 3) uint8
    """
    with h5py.File(episode_path, "r") as f:
        # Load poses and gripper data
        pika_r_pose = f['localization/pose/pika_r'][:]  # (T, 6) [x,y,z, rx,ry,rz]
        pika_r_dist = f['gripper/encoderDistance/pika_r'][:].reshape(-1, 1)
        
        pika_l_pose = f['localization/pose/pika_l'][:]  # (T, 6)
        pika_l_dist = f['gripper/encoderDistance/pika_l'][:].reshape(-1, 1)
        
        # Ensure consistent length
        T = min(len(pika_r_pose), len(pika_l_pose), len(pika_r_dist), len(pika_l_dist))
        pika_r_pose = pika_r_pose[:T]
        pika_l_pose = pika_l_pose[:T]
        pika_r_dist = pika_r_dist[:T]
        pika_l_dist = pika_l_dist[:T]
        
        # Extract position and rotation for robot0 (right arm)
        robot0_eef_pos = pika_r_pose[:, :3].astype(np.float32)
        robot0_eef_rot = pika_r_pose[:, 3:6].astype(np.float32)  # axis-angle (radians)
        robot0_gripper = pika_r_dist.astype(np.float32)
        
        # Extract for robot1 (left arm)
        robot1_eef_pos = pika_l_pose[:, :3].astype(np.float32)
        robot1_eef_rot = pika_l_pose[:, 3:6].astype(np.float32)
        robot1_gripper = pika_l_dist.astype(np.float32)
        
        # Demo start and end poses (broadcast to all frames like UMI does)
        # UMI keeps these constant across the episode
        demo_start_pose_r = np.empty((T, 6), dtype=np.float64)
        demo_start_pose_r[:] = pika_r_pose[0].astype(np.float64)
        demo_end_pose_r = np.empty((T, 6), dtype=np.float64)
        demo_end_pose_r[:] = pika_r_pose[-1].astype(np.float64)
        
        demo_start_pose_l = np.empty((T, 6), dtype=np.float64)
        demo_start_pose_l[:] = pika_l_pose[0].astype(np.float64)
        demo_end_pose_l = np.empty((T, 6), dtype=np.float64)
        demo_end_pose_l[:] = pika_l_pose[-1].astype(np.float64)
        
        # Load camera images
        cameras = {}
        camera_names = ['pikaFisheyeCamera_r', 'pikaFisheyeCamera_l']
        
        for idx, cam in enumerate(camera_names):
            ds_path = f'camera/color/{cam}'
            if ds_path in f:
                paths = f[ds_path][:]
                str_paths = [p.decode('utf-8') if isinstance(p, (bytes, np.bytes_)) else str(p) 
                            for p in paths]
                str_paths = str_paths[:T]
                
                images = []
                for rel in str_paths:
                    img_path = (episode_path.parent / rel).resolve()
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            if target_size:
                                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                            images.append(img)
                        else:
                            h, w = target_size[1], target_size[0] if target_size else (480, 640)
                            images.append(np.zeros((h, w, 3), dtype=np.uint8))
                    else:
                        h, w = target_size[1], target_size[0] if target_size else (480, 640)
                        images.append(np.zeros((h, w, 3), dtype=np.uint8))
                
                cameras[f'camera{idx}_rgb'] = np.stack(images, axis=0)
    
    # Return in UMI format - all data for one episode
    episode_data = {
        'robot0_eef_pos': robot0_eef_pos,
        'robot0_eef_rot_axis_angle': robot0_eef_rot,
        'robot0_gripper_width': robot0_gripper,
        'robot0_demo_start_pose': demo_start_pose_r,
        'robot0_demo_end_pose': demo_end_pose_r,
        'robot1_eef_pos': robot1_eef_pos,
        'robot1_eef_rot_axis_angle': robot1_eef_rot,
        'robot1_gripper_width': robot1_gripper,
        'robot1_demo_start_pose': demo_start_pose_l,
        'robot1_demo_end_pose': demo_end_pose_l,
    }
    
    # Add cameras if available
    if 'camera0_rgb' in cameras:
        episode_data['camera0_rgb'] = cameras['camera0_rgb']
    if 'camera1_rgb' in cameras:
        episode_data['camera1_rgb'] = cameras['camera1_rgb']
    
    return episode_data


def create_replay_buffer(
    episodes: List[Path],
    output_path: Path,
    compression_level: int = 99,
    image_size: Tuple[int, int] = None,
    overwrite: bool = False,
):
    """
    Create ReplayBuffer from HDF5 episodes (UMI-style).
    
    Args:
        episodes: List of paths to HDF5 episode files
        output_path: Path to output Zarr file
        compression_level: JpegXL compression level (0-100)
        image_size: Target (width, height) for images, or None to keep original
        overwrite: Whether to overwrite existing output
    """
    if output_path.exists() and overwrite:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    elif output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path exists: {output_path}")
    
    # Create empty replay buffer in memory first
    print("Creating ReplayBuffer...")
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    # Set cv2 threads to 1 for multiprocessing stability
    cv2.setNumThreads(1)
    
    # Process episodes and add low-dimensional data first
    print(f"\nProcessing {len(episodes)} episodes...")
    episode_data_list = []
    
    for ep_path in tqdm.tqdm(episodes, desc="Loading episodes"):
        try:
            ep_data = load_episode_data(ep_path, target_size=image_size)
            
            # Separate lowdim and image data
            lowdim_data = {
                k: v for k, v in ep_data.items() 
                if not k.startswith('camera')
            }
            
            # Add episode with only low-dim data (no compressors for these)
            out_replay_buffer.add_episode(data=lowdim_data, compressors=None)
            
            # Store full episode data for later image processing
            episode_data_list.append(ep_data)
            
        except Exception as e:
            print(f"\nWarning: Failed to load {ep_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not episode_data_list:
        raise ValueError("No episodes loaded successfully")
    
    print(f"\nSuccessfully loaded {len(episode_data_list)} episodes")
    
    # Now add image data with compression
    # Get image dimensions from first episode
    sample = episode_data_list[0]
    has_cam0 = 'camera0_rgb' in sample
    has_cam1 = 'camera1_rgb' in sample
    
    if has_cam0:
        _, img_h, img_w, _ = sample['camera0_rgb'].shape
        print(f"Image size: {img_w}x{img_h}")
    
    # Create image datasets with JpegXL compression
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    total_frames = out_replay_buffer['robot0_eef_pos'].shape[0]
    
    print(f"\nCreating image datasets (total frames: {total_frames})...")
    
    if has_cam0:
        _ = out_replay_buffer.data.require_dataset(
            name='camera0_rgb',
            shape=(total_frames, img_h, img_w, 3),
            chunks=(1, img_h, img_w, 3),
            compressor=img_compressor,
            dtype=np.uint8
        )
    
    if has_cam1:
        _ = out_replay_buffer.data.require_dataset(
            name='camera1_rgb',
            shape=(total_frames, img_h, img_w, 3),
            chunks=(1, img_h, img_w, 3),
            compressor=img_compressor,
            dtype=np.uint8
        )
    
    # Write images episode by episode
    print("\nWriting images to ReplayBuffer...")
    frame_idx = 0
    for ep_data in tqdm.tqdm(episode_data_list, desc="Writing images"):
        ep_len = ep_data['robot0_eef_pos'].shape[0]
        
        if has_cam0 and 'camera0_rgb' in ep_data:
            out_replay_buffer.data['camera0_rgb'][frame_idx:frame_idx+ep_len] = ep_data['camera0_rgb']
        
        if has_cam1 and 'camera1_rgb' in ep_data:
            out_replay_buffer.data['camera1_rgb'][frame_idx:frame_idx+ep_len] = ep_data['camera1_rgb']
        
        frame_idx += ep_len
    
    # Save to disk
    print(f"\nSaving ReplayBuffer to {output_path}")
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    
    print(f"\nDataset created successfully!")
    print(f"Total episodes: {len(episode_data_list)}")
    print(f"Total frames: {total_frames}")
    
    # Print structure
    print("\nDataset structure:")
    with zarr.ZipStore(str(output_path), mode='r') as zip_store:
        root = zarr.open(store=zip_store, mode='r')
        print(root.tree())


def main():
    parser = argparse.ArgumentParser(
        description="Convert Agilex HDF5 episodes to Zarr ReplayBuffer (UMI-style)"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing Agilex HDF5 data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output Zarr path (e.g., dataset.zarr.zip)'
    )
    parser.add_argument(
        '--compression_level',
        type=int,
        default=99,
        help='JpegXL compression level (0-100, default: 99)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=('WIDTH', 'HEIGHT'),
        help='Target image size (width height), e.g., 224 224'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output if it exists'
    )
    
    args = parser.parse_args()
    
    # Find all HDF5 episodes
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    hdf5_files = sorted(dataset_dir.glob("**/data.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 episodes")
    
    if not hdf5_files:
        raise ValueError("No HDF5 files found in the dataset directory")
    
    # Convert using ReplayBuffer
    output_path = Path(args.output_path)
    image_size = tuple(args.image_size) if args.image_size else None
    
    create_replay_buffer(
        episodes=hdf5_files,
        output_path=output_path,
        compression_level=args.compression_level,
        image_size=image_size,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()