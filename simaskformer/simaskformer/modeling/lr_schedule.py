from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

def build_custom_lr_scheduler(cfg, optimizer):
    """
    Build a CosineAnnealingWarmRestarts LR scheduler based on Detectron2 cfg.
    """
    schedule= cfg.SOLVER.LR_SCHEDULER_NAME
    if schedule == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.SOLVER.T_0,         # Iteration count for first cycle
            T_mult=1,                   # Keep cycles equal length
            eta_min=cfg.SOLVER.ETA_MIN  # Minimum LR
        )
    elif schedule == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.T_MAX,         
            eta_min=cfg.SOLVER.ETA_MIN, 
            last_epoch = cfg.SOLVER.LAST_EPOCH
        )
    else:
        raise ValueError(f"The schedule {schedule} custom is not defined.")
    
    return scheduler