"""Training helpers: LR schedule and rank-aware printing."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from einops import rearrange
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def print_rank0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine', eta_min=0.0):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine_annealing':
        # Linear warmup followed by cosine decay.
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if warm_up_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warm_up_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=param_update_steps - warm_up_steps, eta_min=eta_min,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_steps],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=param_update_steps, eta_min=eta_min,
            )
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler


def visualize_intermediate_results(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    if result.render is not None:
        target_image = target.image
        rendered_image = result.render
        b, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        
        if hasattr(target, "index") and target.index is not None:
            uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]
            uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        else:
            uid_based_filename = "vis"

        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
        )
        if hasattr(target, "index") and target.index is not None:
            with open(os.path.join(out_dir, "uids.txt"), "w") as f:
                uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]
                uids = "_".join([f"{uid:08}" for uid in uids])
                f.write(uids)

    if hasattr(input, "index") and input.index is not None:
        input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
        input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    else:
        input_uid_based_filename = "vis"
    
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg")
    )
