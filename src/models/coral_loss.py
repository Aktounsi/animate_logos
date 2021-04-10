import torch.nn.functional as F
import torch


def coral_loss(logits, levels, importance_weights=None, reduction='mean'):

    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as logits (%s). "
                         % (logits.shape, levels.shape))

    term1 = (F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss