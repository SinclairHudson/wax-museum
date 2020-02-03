import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    # implementing focal loss. This module takes SOFTMAXED output in, as well as the labels, and computes the loss.
    # gamma is the gamma described in the focal loss paper, while alpha is the weight for each class.
    # https://arxiv.org/pdf/1708.02002.pdf
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6  # need a small value so we don't do log(0)
        self.alpha = alpha

    def forward(self, predicted, target):
        one_hot = F.one_hot(target)
        sparse_pt = predicted * one_hot  # element-wise - we only care about the probability of the true class
        pt = torch.sum(sparse_pt, 1)  # now we have a batchsize long tensor, containing all pt values
        if self.alpha:
            weighted_one_hot = one_hot * self.alpha  # element-wise again
            weights = torch.sum(weighted_one_hot, 1)
        else:
            weights = 1
        # at the time of writing, .cuda() is required if on gpu. It's a bug in PyTorch, and the devs know about it.
        focal_loss = - weights * torch.pow((1 - pt), self.gamma.cuda()) * torch.log(pt + self.eps)  # element-wise

        return focal_loss.sum()





