import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-8):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, X):
        loss = torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) + \
               torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :]))
        return loss


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.real_label = random.uniform(0.9, 1.0)
        self.fake_label = random.uniform(0.0, 0.1)
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
        self.device = device

    def get_target_tensor(self, x, target_is_real):
        if target_is_real: # real image
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label).to(self.device)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(x)

        else: # fake image
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label).to(self.device)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(x)

    def get_zero_tensor(self, x):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0).to(self.device)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(x)

    def loss(self, x, target_is_real):
        # cross entropy loss
        target_tensor = self.get_target_tensor(x, target_is_real)
        loss = F.binary_cross_entropy_with_logits(x, target_tensor)
        return loss

    def __call__(self, x, target_is_real):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(x, list):
            loss = 0
            for pred_i in x:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(x)
        else:
            return self.loss(x, target_is_real)