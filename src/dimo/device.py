"""Bind the same visible CUDA namespace as the SFT launcher, before NCCL."""
import os
import torch


def initialize_local_cuda_device():
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank < 0:
        return
    visible = torch.cuda.device_count()
    workers = int(os.environ.get('LOCAL_WORLD_SIZE', local_rank + 1))
    if local_rank >= visible or workers > visible:
        raise RuntimeError(f'Invalid DDP GPU mapping: LOCAL_RANK={local_rank}, '
                           f'LOCAL_WORLD_SIZE={workers}, visible GPUs={visible}, '
                           f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    torch.cuda.set_device(local_rank)
