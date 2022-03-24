from torch import autograd
import torch
import torch.distributed as dist
from torch.nn import functional as F


def compute_grad_gp(d_out, x_in, is_patch=False):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum() if not is_patch else d_out.mean(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.sum() / batch_size
    return reg

def calc_adv_loss(logit, mode):
    assert mode in ['d_real', 'd_fake', 'g']
    if mode == 'd_real':
        loss = F.relu(1.0 - logit).mean()
    elif mode == 'd_fake':
        loss = F.relu(1.0 + logit).mean()
    else:
        loss = -logit.mean()

    return loss
  
def calc_recon_loss(predict, target):
    return torch.mean(torch.abs(predict - target))
