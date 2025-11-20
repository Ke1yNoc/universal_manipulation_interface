#!/usr/bin/env python3
"""
Test if policy inference works at all
"""

import torch
import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Load checkpoint
ckpt_path = '/home/neepok1/VLA/umi_dp_rog/epoch=0050-train_loss=0.020.ckpt'
print(f"Loading checkpoint from {ckpt_path}...")
payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
cfg = payload['cfg']

print("Creating workspace...")
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

print("Getting policy...")
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model
policy.num_inference_steps = 16

print("Moving to GPU...")
device = torch.device('cuda')
policy.eval().to(device)

print("Creating dummy observation matching model's expected shapes...")
# Get expected shapes from the model's shape_meta
obs_shape_meta = cfg.task.shape_meta.obs

# Create observations with correct shapes
obs_dict = {}

# Camera observations
for key in ['camera0_rgb', 'camera1_rgb']:
    horizon = obs_shape_meta[key].horizon
    shape = obs_shape_meta[key].shape  # Should be (3, 224, 224)
    obs_dict[key] = torch.randn(1, horizon, *shape).to(device)
    print(f"  {key}: {obs_dict[key].shape}")

# Low-dim observations
for key in obs_shape_meta.keys():
    if key.startswith('robot'):
        horizon = obs_shape_meta[key].horizon
        shape = obs_shape_meta[key].shape
        obs_dict[key] = torch.randn(1, horizon, *shape).to(device)
        print(f"  {key}: {obs_dict[key].shape}")

print("Running inference...")
print("(This may take 30-60 seconds on first run for CUDA compilation)")

import time
start = time.time()

with torch.no_grad():
    policy.reset()
    print("  Policy reset complete")
    
    print("  Calling predict_action...")
    result = policy.predict_action(obs_dict)
    
elapsed = time.time() - start
print(f"✓ Inference completed in {elapsed:.2f}s")
print(f"  Action shape: {result['action_pred'].shape}")
print("\n✓ Policy inference test PASSED!")
