"""Local OpenVLA-OFT sample inference (CPU/MPS).

Run from libero-sim env. First call will download ~15GB of weights from HF
to ~/.cache/huggingface/. Set HF_HOME to override.

Usage:
    conda activate libero-sim
    python local/openvla_inference.py
    python local/openvla_inference.py --device mps          # try MPS
    python local/openvla_inference.py --device cpu          # safer, slow
    HF_HOME=/Volumes/MyExternal/hf python local/openvla_inference.py
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

# Make sure openvla-oft is on the path (in case run from project root)
EXT = Path(__file__).resolve().parent.parent / 'external' / 'openvla-oft'
if EXT.exists():
    sys.path.insert(0, str(EXT))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    ap.add_argument('--checkpoint', default='moojink/openvla-7b-oft-finetuned-libero-spatial')
    args = ap.parse_args()

    if args.device == 'mps' and not torch.backends.mps.is_available():
        print('MPS not available, falling back to CPU')
        args.device = 'cpu'

    # Avoid CUDA-only flash-attn import path
    import os
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_vla, get_processor, get_action_head, get_proprio_projector, get_vla_action,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key='libero_spatial_no_noops',
    )

    print(f'Device: {args.device}')
    print(f'Checkpoint: {args.checkpoint}')
    print('Loading VLA (first run downloads ~15GB)...')
    t0 = time.time()
    vla = get_vla(cfg)
    print(f'  VLA loaded in {time.time()-t0:.1f}s')

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    # Move to chosen device (default util loads on cuda; we override)
    if args.device != 'cuda':
        vla = vla.to(args.device).to(torch.float32)  # bf16 unstable on mps
        action_head = action_head.to(args.device).to(torch.float32)
        proprio_projector = proprio_projector.to(args.device).to(torch.float32)

    sample = EXT / 'experiments/robot/libero/sample_libero_spatial_observation.pkl'
    with open(sample, 'rb') as f:
        observation = pickle.load(f)

    print(f'\nRunning inference on sample observation...')
    print(f'Task: {observation["task_description"]}')
    t0 = time.time()
    actions = get_vla_action(
        cfg, vla, processor, observation, observation['task_description'],
        action_head, proprio_projector,
    )
    dt = time.time() - t0
    print(f'\nInference time: {dt:.1f}s for {len(actions)} actions ({dt/len(actions):.2f}s per action)')
    print(f'Generated action chunk:')
    for i, a in enumerate(actions):
        print(f'  step {i}: {a}')


if __name__ == '__main__':
    main()
