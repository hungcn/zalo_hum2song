import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairsLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairsLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss


class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class LosslessTripletLoss(nn.Module):
    def __init__(self, beta=None, eps=1e-8, reduction='mean'):
        #beta - The scaling factor, number of dimensions by default.
        #eps - The Epsilon value to prevent ln(0)
        super(LosslessTripletLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        assert self.reduction in ('none', 'mean', 'sum')

    def forward(self, a, p, n):
        '''
        Arguments:
            a - anchor vectors with values in range (0, 1)
            p - positive vectors with values in range (0, 1)
            n - negative vectors with values in range (0, 1)
        '''
        assert a.shape == p.shape, 'Shapes dont match.'
        assert a.shape == n.shape, 'Shapes dont match.'

        N = a.shape[1]
        beta = N if self.beta is None else self.beta
        dist_p = (a - p).pow(2).sum(dim=1)
        dist_n = (a - n).pow(2).sum(dim=1)
        dist_p = -torch.log(-(    dist_p) / beta + 1 + self.eps)
        dist_n = -torch.log(-(N - dist_n) / beta + 1 + self.eps)
        out = dist_n + dist_p
        if self.reduction == 'none':
            return out
        elif self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            raise ValueError('Unknown reduction type')