"""Inspection example: FP16 gradients are clipped before unscaling.
Try inserting scaler.unscale_(optimizer) immediately before clipping.
"""
import torch


def train_step(model, batch, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()
