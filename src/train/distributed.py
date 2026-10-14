"""Distributed setup and run bookkeeping for training."""

from __future__ import annotations

import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict


def init_distributed(seed: int = 42) -> edict:
    """Initialize distributed state and return rank, device, and seed metadata."""
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    torch.backends.cudnn.benchmark = True

    return edict(
        {
            "local_rank": local_rank,
            "global_rank": global_rank,
            "world_size": world_size,
            "device": device,
            "is_main_process": global_rank == 0,
            "seed": process_seed,
        }
    )


def init_run(config) -> None:
    """Save the run config and initialize W&B logging."""
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.training.checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(config), f)

    import wandb

    wandb.init(
        project=config.training.get("wandb_project", "panopticlvsm"),
        name=config.training.get("wandb_exp_name", None),
        dir=config.training.checkpoint_dir,
        config=dict(config),
    )
