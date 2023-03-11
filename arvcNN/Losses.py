import torch
import torch.nn as nn
import numpy as np
from scipy import stats

class ArvcLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ArvcLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = inputs.detach().cpu().numpy()
        # targets = targets.detach().cpu().numpy()

        loss = np.zeros_like(targets)

        for batch in range(len(targets)):
            labels = np.unique(targets)
            for label in labels:
                tagt_indx = np.array(np.where(targets[batch] == label)).reshape(-1)

                if len(tagt_indx) > 0:
                    values = inputs[batch][tagt_indx]

                    moda, _ = stats.mode(values)
                    diff = values - moda
                    diff = np.sum(diff)
                    loss_tmp = diff / len(values)
                    loss[batch][tagt_indx] = loss_tmp

        loss = np.mean(loss)
        return loss