'''
Author: John Yang from AHU
Contact: 2268866312@qq.com
Date: 04.03.2020
'''

import torch
import numpy as np

def dice(vol1, vol2, labels=None):
    '''
    A PyTorch implementation of multi-class version of dice_loss

    paramters:
    vol1: torch Tensor. The first volume, which shape: [N, C, :, :, :]; N:batch size, C:channel
    vol2: torch Tensor. The second volume, which sames to vol1
    labels: default none.
    '''

    if labels is None:
        labels = torch.unique(torch.cat((vol1, vol2)))
    label_no_bg = list()
    for i in labels:
        if i != 0:
            label_no_bg.append(i)  # remove background

    dices = torch.Tensor(np.zeros(len(label_no_bg)))

    for idx, lab in enumerate(label_no_bg):
        vol1_d = vol1 == lab
        vol2_d = vol2 == lab

        top = 2 * torch.sum(torch.mul(vol1_d, vol2_d), dtype=float)

        bottom = torch.sum(vol1_d, dtype=float) + torch.sum(vol2_d, dtype=float)
        bottom = torch.max(bottom, (torch.ones_like(bottom, dtype=float) * 1e-5))  # add epsilon.

        dices[idx] = -1 * (top / bottom)

    return torch.mean(dices)
