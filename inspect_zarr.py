#!/usr/bin/env python3
"""
Inspect and print sample frames from a Zarr dataset (ReplayBuffer format).

Usage:
    python inspect_zarr.py dataset.zarr.zip
    python inspect_zarr.py dataset.zarr.zip --episode 0 --frame 10
    python inspect_zarr.py dataset.zarr.zip --save_image sample.png
"""

import argparse
import zarr
import numpy as np
from pathlib import Path

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("Warning: diffusion_policy codecs not available")


def inspect_zarr(zarr_path: str, episode_idx: int = 0, frame_idx: int = 0, save_image: str = None):
    """Inspect a Zarr dataset and print sample frame information."""
    
    # Open the Zarr store
    if zarr_path.endswith('.zip'):
        store = zarr.ZipStore(zarr_path, mode='r')
    else:
        store = zarr.DirectoryStore(zarr_path)
    
    root = zarr.open(store=store, mode='r')
    
    print("=" * 80)
    print(f"ZARR DATASET STRUCTURE: {zarr_path}")
    print("=" * 80)
    print(root.tree())
    print()
    
    # Get data group
    if 'data' in root:
        data = root['data']
    else:
        data = root
    
    # Get meta group if available
    meta = root.get('meta', None)
    
    # Print dataset info
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    
    # Get all keys
    all_keys = sorted(list(data.keys()))
    
    for key in all_keys:
        ds = data[key]
        print(f"{key}:")
        print(f"  Shape: {ds.shape}")
        print(f"  Dtype: {ds.dtype}")
        print(f"  Chunks: {ds.chunks}")
        if hasattr(ds, 'compressor') and ds.compressor:
            print(f"  Compressor: {ds.compressor}")
        print()
    
    # Print metadata if available
    if meta is not None:
        print("=" * 80)
        print("METADATA")
        print("=" * 80)
        for key in meta.keys():
            print(f"{key}: {meta[key][:]}")
        print()
        
        # Get episode boundaries
        if 'episode_ends' in meta:
            episode_ends = meta['episode_ends'][:]
            num_episodes = len(episode_ends)
            print(f"Number of episodes: {num_episodes}")
            print(f"Episode ends: {episode_ends}")
            
            # Calculate episode lengths
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            episode_lengths = episode_ends - episode_starts
            print(f"Episode lengths: {episode_lengths}")
            print(f"Total frames: {episode_ends[-1]}")
            print()
            
            # Determine frame index in dataset
            if episode_idx >= num_episodes:
                print(f"Error: Episode {episode_idx} does not exist (only {num_episodes} episodes)")
                return
            
            start_idx = episode_starts[episode_idx]
            end_idx = episode_ends[episode_idx]
            ep_len = end_idx - start_idx
            
            if frame_idx >= ep_len:
                print(f"Error: Frame {frame_idx} does not exist in episode {episode_idx} (length: {ep_len})")
                return
            
            abs_frame_idx = start_idx + frame_idx
            
            print(f"Selected: Episode {episode_idx}, Frame {frame_idx}")
            print(f"  Episode range: [{start_idx}, {end_idx})")
            print(f"  Episode length: {ep_len}")
            print(f"  Absolute frame index: {abs_frame_idx}")
            print()
        else:
            abs_frame_idx = frame_idx
            print(f"No episode metadata found, using absolute frame index: {abs_frame_idx}")
            print()
    else:
        abs_frame_idx = frame_idx
        print("No metadata found, using absolute frame index")
        print()
    
    # Print sample frame
    print("=" * 80)
    print(f"SAMPLE FRAME DATA (Absolute Index: {abs_frame_idx})")
    print("=" * 80)
    
    for key in all_keys:
        ds = data[key]
        
        # Skip images for now, print later
        if 'camera' in key and 'rgb' in key:
            continue
        
        try:
            value = ds[abs_frame_idx]
            print(f"{key}:")
            if value.size <= 20:  # Print full array if small
                print(f"  {value}")
            else:  # Print summary if large
                print(f"  Shape: {value.shape}")
                print(f"  Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}")
            print()
        except Exception as e:
            print(f"{key}: Error reading - {e}")
            print()
    
    # Print image information
    print("=" * 80)
    print("IMAGE DATA")
    print("=" * 80)
    
    for key in all_keys:
        if 'camera' in key and 'rgb' in key:
            ds = data[key]
            try:
                img = ds[abs_frame_idx]
                print(f"{key}:")
                print(f"  Shape: {img.shape} (H, W, C)")
                print(f"  Dtype: {img.dtype}")
                print(f"  Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.2f}")
                
                # Check if image is valid (not all zeros)
                if img.sum() == 0:
                    print(f"  WARNING: Image is all zeros!")
                else:
                    print(f"  ✓ Image contains data")
                
                # Save image if requested
                if save_image and key == 'camera0_rgb':
                    try:
                        import cv2
                        # Convert RGB to BGR for OpenCV
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_image, img_bgr)
                        print(f"  Saved to: {save_image}")
                    except Exception as e:
                        print(f"  Failed to save image: {e}")
                
                print()
            except Exception as e:
                print(f"{key}: Error reading - {e}")
                print()
    
    # Print correlation check
    print("=" * 80)
    print("DATA CONSISTENCY CHECKS")
    print("=" * 80)
    
    # Check if all robot data has the same length
    robot_keys = [k for k in all_keys if k.startswith('robot')]
    if robot_keys:
        lengths = [data[k].shape[0] for k in robot_keys]
        if len(set(lengths)) == 1:
            print(f"✓ All robot data has consistent length: {lengths[0]}")
        else:
            print(f"✗ Inconsistent robot data lengths: {dict(zip(robot_keys, lengths))}")
    
    # Check if camera data matches robot data
    camera_keys = [k for k in all_keys if 'camera' in k]
    if camera_keys and robot_keys:
        cam_len = data[camera_keys[0]].shape[0]
        robot_len = data[robot_keys[0]].shape[0]
        if cam_len == robot_len:
            print(f"✓ Camera and robot data lengths match: {cam_len}")
        else:
            print(f"✗ Length mismatch - Camera: {cam_len}, Robot: {robot_len}")
    
    print()
    
    # Close store
    if isinstance(store, zarr.ZipStore):
        store.close()
    
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Zarr dataset and print sample frames"
    )
    parser.add_argument(
        '--zarr_path',
        type=str,
        help='Path to Zarr dataset (.zarr or .zarr.zip)'
    )
    parser.add_argument(
        '--episode',
        type=int,
        default=0,
        help='Episode index to inspect (default: 0)'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='Frame index within episode (default: 0)'
    )
    parser.add_argument(
        '--save_image',
        type=str,
        default=None,
        help='Save camera0_rgb image to file (e.g., sample.png)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.zarr_path).exists():
        print(f"Error: Path does not exist: {args.zarr_path}")
        return
    
    inspect_zarr(args.zarr_path, args.episode, args.frame, args.save_image)


if __name__ == "__main__":
    main()