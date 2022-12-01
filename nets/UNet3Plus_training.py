import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor



CUDA_LAUNCH_BLOCKING = 1
USE_JIT = False
relu = nn.ReLU(inplace=True)

if USE_JIT:
    _jit = torch.jit.script
else:
    _jit = lambda f: f


@_jit
def gaussian_kernel(kernel_size: int, sigma: float):
    gauss = torch.arange(0, kernel_size) - kernel_size // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


@_jit
def gaussian_kernel2d(kernel_size: int, channel: int = 1) -> Tensor:
    '''
    2d gauss kernel, out put shape: [channel, 1, window_size, window_size]
    '''
    k = gaussian_kernel(kernel_size, 1.5)
    k = torch.einsum('i,j->ij', [k, k])
    return k.expand(channel, 1, kernel_size, kernel_size).contiguous()


@_jit
def ssim_index(img1: Tensor,
               img2: Tensor,
               kernel: Tensor,
               nonnegative: bool = True,
               channel_avg: bool = False,
               val_range: float = 1.):
    assert img1.shape == img2.shape
    if len(img1.shape) > 3:
        channel = img1.shape[1]
    else:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        channel = 1
    _, channel, height, width = img1.shape
    if img1.dtype == torch.long:
        img1 = img1.float()
    if img2.dtype == torch.long:
        img2 = img2.float()
    L = val_range

    s = 1
    p = 0
    mean1 = F.conv2d(img1, kernel, padding=p, groups=channel, stride=s)
    mean2 = F.conv2d(img2, kernel, padding=p, groups=channel, stride=s)
    mean12 = mean1 * mean2
    mean1 = mean1.pow(2)
    mean2 = mean2.pow(2)

    # https://en.wikipedia.org/wiki/Variance#Definition
    var1 = F.conv2d(img1 ** 2, kernel, padding=p, groups=channel, stride=s) - mean1
    var2 = F.conv2d(img2 ** 2, kernel, padding=p, groups=channel, stride=s) - mean2

    # https://en.wikipedia.org/wiki/Covariance#Definition
    covar = F.conv2d(img1 * img2, kernel, padding=p, groups=channel, stride=s) - mean12

    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    cs = (2. * covar + c2) / (var1 + var2 + c2)
    # print(covar.mean(), var1.mean(), var2.mean(), cs.mean())  # sparse input could result in large cs
    ss = (2. * mean12 + c1) / (mean1 + mean2 + c1) * cs

    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    ss, cs = ss.mean(dim=-1), cs.mean(dim=-1)
    if nonnegative:
        ss, cs = relu(ss), relu(cs)
    return ss, cs


@_jit
def ms_ssim(
        x: Tensor,
        y: Tensor,
        kernel: Tensor,
        weights: Tensor,
        val_range: float = 1.,
        nonnegative: bool = True
) -> Tensor:
    r"""Returns the MS-SSIM between :math:`x` and :math:`y`.

    modified from https://github.com/francois-rozet/piqa/blob/master/piqa/ssim.py
    """

    css = []
    kernel_size = kernel.shape[-1]
    m = weights.numel()

    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
            h, w = x.shape[-2:]
            if h < kernel_size or w < kernel_size:
                weights = weights[:i] / torch.sum(weights[:i])
                break

        ss, cs = ssim_index(
            x, y, kernel,
            channel_avg=False,
            val_range=val_range,
            nonnegative=nonnegative
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1) ** weights
    msss = msss.prod(dim=-1).mean(dim=-1)

    return msss


class SSIMLoss(nn.Module):
    r""" Multi label SIMM Loss for segmentation

    Args:
        win_size: (int, optional): the size of gauss kernel
        nonnegative (bool, optional): force the ssim response to be nonnegative using relu.

    Shape:
        - Input (Tensor): :math:`(B, num_classes, H, W)`, predicted probablity maps
        - Target (Tensor): :math:`(B, H, W)`, range from 0 to num_classes - 1
    """

    def __init__(self, win_size: int = 11, nonnegative: bool = True, process_input: bool = True):

        super(SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input

    def forward(self, pred: Tensor, target: Tensor):
        _, num_classes, h, w = pred.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        kernel = kernel.to(pred.dtype).to(pred.device)

        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.
        for i in range(num_classes):
            ss, _ = ssim_index(pred[:, [i]], target[:, [i]], kernel, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes


class MS_SSIMLoss(nn.Module):
    r""" Multi label SIMM Loss for segmentation
     """

    def __init__(self,
                 win_size: int = 11,
                 weights: Tensor = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]),
                 nonnegative: bool = True,
                 process_input: bool = True):

        super(MS_SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.weights = weights
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input

    def forward(self, pred: Tensor, target: Tensor):
        _, num_classes, h, w = pred.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)

        kernel = kernel.to(pred.dtype).to(pred.device)
        weights = self.weights.to(pred.dtype).to(pred.device)
        # if kernel.device != pred.device:
        #     kernel.to(pred.device)

        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.
        for i in range(num_classes):
            ss = ms_ssim(pred[:, [i]], target[:, [i]], kernel, weights, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes

def BCE_loss(pred,label):
    bce_loss = nn.BCELoss(size_average=True)
    bce_out = bce_loss(pred, label)
    print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out

def binary_iou_loss(pred, target):
    Iand = torch.sum(pred * target, dim=1)
    Ior = torch.sum(pred, dim=1) + torch.sum(target, dim=1) - Iand
    IoU = 1 - Iand.sum() / Ior.sum()
    return IoU.sum()


class IoULoss(nn.Module):
    '''
    multi-classes iou loss
    '''

    def __init__(self, process_input=True) -> None:
        super().__init__()
        self.process_input = process_input

    def forward(self, pred, target):
        num_classes = pred.shape[1]

        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0
        for i in range(num_classes):
            loss = binary_iou_loss(pred[:, i], target[:, i])
            total_loss += loss
        return total_loss / num_classes


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
