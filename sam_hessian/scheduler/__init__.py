import torch


def get_scheduler(
    optimizer, 
    sch_name='cosine_annealing_lr',
    sch_hyperparameter={}
):
    if sch_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **sch_hyperparameter
    )
    elif sch_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        **sch_hyperparameter
    )
    elif sch_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        **sch_hyperparameter
    )
    else:
        raise ValueError("Invalid scheduler!!!")