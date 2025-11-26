import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, scheduler_cfg, epochs):
    if not scheduler_cfg:
        return None
    name = scheduler_cfg.get("name")
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.get("step_size", 10), gamma=scheduler_cfg.get("gamma", 0.1))
    return None
