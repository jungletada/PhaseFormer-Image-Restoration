import torch


def create_poly_lr_scheduler(optimizer, num_step, epochs, warmup=False, warmup_epochs=1,
                             warmup_factor=1e-3, power=0.9, last_epoch=0):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** power
    if last_epoch == -1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=-1)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=last_epoch*num_step)


def create_multistep_lr_schedule(optimizer, gamma, milestones, last_epoch=0):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch)
