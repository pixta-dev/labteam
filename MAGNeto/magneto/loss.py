import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    input: torch.tensor,
    target: torch.tensor,
    beta: float = 1.,
    reduction: str = 'mean',
    smooth: float = 1.
) -> torch.tensor:
    intersection = input * target
    score = ((1. + beta**2) * torch.sum(intersection, dim=-1) + smooth) \
        / (torch.sum(input, dim=-1) + (beta**2) * torch.sum(target, dim=-1) + smooth)

    loss = 1. - score

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

    return loss


class DiceLoss(nn.Module):
    """
    Dice loss's implementation.
    """

    def __init__(self, reduction: str = 'mean', beta: float = 1., smooth: float = 1.):
        """
        input:
            + reduction: specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed.
            + beta: β is chosen such that recall is considered β times as important as precision.
            + smooth: smooth value.
        """
        super(DiceLoss, self).__init__()

        assert beta >= 0, 'β must be a positive real value!'
        assert reduction in ['none', 'mean', 'sum']

        self.beta = beta
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        input:
            + input: prediction.
            + target: ground-truth.
        output:
            + loss value.
        """
        return dice_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
            smooth=self.smooth
        )


class BCEDiceLoss(nn.Module):
    """
    The combination of Binary Cross-Entropy & Dice losses.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        weight: torch.tensor = None,
        beta: float = 1.,
        smooth: float = 1.0
    ):
        """
        input:
            + reduction: specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed.
            + weight: a manual rescaling weight given to the loss
                of each batch element. If given, has to be a Tensor of size `nbatch`,
                used for BCE part.
            + beta: β is chosen such that recall is considered β times as important as precision.
            + smooth: smooth value, used for Dice part.
        """
        super(BCEDiceLoss, self).__init__()

        assert beta >= 0, 'β must be a positive real value!'
        assert reduction in ['none', 'mean', 'sum']

        self.reduction = reduction

        # BCE's params
        self.weight = weight

        # Dice's params
        self.beta = beta
        self.smooth = smooth

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        input:
            + input: prediction.
            + target: ground-truth.
        output:
            + loss value.
        """
        bce = F.binary_cross_entropy(
            input,
            target,
            weight=self.weight,
            reduction=self.reduction
        )
        dice = dice_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
            smooth=self.smooth
        )

        return bce + dice
