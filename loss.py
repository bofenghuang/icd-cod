# coding=utf-8
# Copyright 2024  Bofeng Huang

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


def multi_label_focal_loss_with_logits(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    weight: Optional[Union[float, List[float], np.ndarray, torch.Tensor]] = None,
    alpha: float = 0.5,
    gamma: float = 2.0,
    reduction: str = "none",
):
    """Compute focal loss for multi-label classification."""

    # valdiate input
    if input_tensor.shape != target_tensor.shape:
        raise ValueError(f"Expected input shape {input_tensor.shape} to match target shape {target_tensor.shape}")

    if weight is not None:
        if isinstance(weight, (float, list, np.ndarray)):
            # 1 or num_classes
            weight = torch.tensor(weight, dtype=input_tensor.dtype, device=input_tensor.device)
        elif not isinstance(weight, torch.Tensor):
            raise ValueError(f"Unsupported weight type: {type(weight)}")

    # compute binary cross-entropy loss
    # batch_size x n_classes
    bce_loss = F.binary_cross_entropy_with_logits(input_tensor, target_tensor, reduction="none")

    # compute probabilities
    pt = torch.exp(-bce_loss)

    # focal loss
    alpha_t = torch.where(target_tensor == 1, alpha, 1 - alpha)
    focal_loss = alpha_t * (1 - pt).pow(gamma) * bce_loss

    # weighted focal loss
    # did't use weight in F.binary_cross_entropy_with_logits
    # because need an unweighted bce loss to calculated the pt
    if weight is not None:
        focal_loss *= weight

    # apply reduction
    if reduction == "none":
        return focal_loss
    elif reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


class MultiLabelFocalLossWithLogits(nn.Module):
    def __init__(
        self,
        weight: Optional[Union[float, List[float], np.ndarray, torch.Tensor]] = None,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        return multi_label_focal_loss_with_logits(
            input_tensor, target_tensor, weight=self.weight, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )


class ClassBalancedWrapper(nn.Module):
    """
    Class-balanced loss as described in https://arxiv.org/abs/1901.05555
    Formula: CB(p, y) = \frac{1 - \beta}{1 - \beta^{n_y}} \mxathcal {L}(p, y)
    """

    def __init__(self, criterion: nn.Module, num_samples: Union[List[float], np.ndarray, torch.Tensor], beta: float = 0.99):
        super().__init__()
        self.criterion = criterion
        self.beta = beta

        # convert num_samples to torch.Tensor if it's not already
        num_samples = torch.as_tensor(num_samples, dtype=torch.float32)
        self.n_classes = num_samples.shape[0]

        # Calculate class-balanced weights
        # effective number
        cb_weights = (1 - beta) / (1 - beta**num_samples)
        # handle classes with no samples, when some class exist only in val/test
        cb_weights[cb_weights.isinf()] = 0
        # normalize
        cb_weights = cb_weights / cb_weights.sum() * self.n_classes
        self.register_buffer("cb_weights", cb_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # original version in the paper
        # n_examples, n_classes = logits.shape[:2]
        # weight = torch.as_tensor(self.cb_weights, device=logits.device).float()
        # weight = weight.unsqueeze(0)
        # weight = weight.repeat(n_examples, 1) * targets
        # weight = weight.sum(1)
        # weight = weight.unsqueeze(1)
        # weight = weight.repeat(1, n_classes)

        weight = self.cb_weights.to(dtype=torch.float32, device=logits.device)
        # batch_size x 1
        # weight = torch.sum(weight * targets, dim=1, keepdim=True)
        # # batch_size x num_classes
        # weight = weight.expand_as(logits)

        self.criterion.weight = weight
        return self.criterion(logits, targets)


class NegativeTolerantRegularizationWrapper(nn.Module):
    """
    Negative-Tolerant Regularization as described in https://arxiv.org/abs/2007.09654
    """

    def __init__(
        self,
        criterion: nn.Module,
        num_samples: Union[List[float], np.ndarray, torch.Tensor],
        neg_scale: float = 1.0,
        init_bias: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.neg_scale = neg_scale  # \lambda
        self.eps = eps

        # convert num_samples to torch.Tensor if it's not already
        num_samples = torch.as_tensor(num_samples, dtype=torch.float32)

        # calculate initial bias
        # init_bias: k -> -v_i
        init_bias = -torch.log(num_samples.sum() / (num_samples + self.eps) - 1) * init_bias
        self.register_buffer("init_bias", init_bias)

    def regularize_logits(self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None):
        logits = logits + self.init_bias.to(device=logits.device)

        logits = logits * (1 - targets) * self.neg_scale + logits * targets
        if weight is not None:
            weight = weight / self.neg_scale * (1 - targets) + weight * targets

        return logits, weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, weight = self.regularize_logits(logits, targets.float(), self.criterion.weight)

        self.criterion.weight = weight
        return self.criterion(logits, targets)


class ClassBalancedNegativeTolerantRegularizationWrapper(nn.Module):
    """
    Class-Balanced Negative-Tolerant Regularization.
    """

    def __init__(
        self,
        criterion: nn.Module,
        num_samples: Union[List[float], np.ndarray, torch.Tensor],
        beta: float = 0.99,
        neg_scale: float = 1.0,
        init_bias: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.beta = beta
        self.neg_scale = neg_scale  # \lambda
        self.eps = eps

        # convert num_samples to torch.Tensor if it's not already
        num_samples = torch.as_tensor(num_samples, dtype=torch.float32)
        self.n_classes = num_samples.shape[0]

        # Calculate class-balanced weights
        # effective number
        cb_weights = (1 - beta) / (1 - beta**num_samples)
        # handle classes with no samples, when some class exist only in val/test
        cb_weights[cb_weights.isinf()] = 0
        # normalize
        cb_weights = cb_weights / cb_weights.sum() * self.n_classes
        self.register_buffer("cb_weights", cb_weights)

        # calculate initial bias
        # init_bias: k -> -v_i
        init_bias = -torch.log(num_samples.sum() / (num_samples + self.eps) - 1) * init_bias
        self.register_buffer("init_bias", init_bias)

    def regularize_logits(self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None):
        logits = logits + self.init_bias.to(device=logits.device)

        logits = logits * (1 - targets) * self.neg_scale + logits * targets
        if weight is not None:
            weight = weight / self.neg_scale * (1 - targets) + weight * targets

        return logits, weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        weight = self.cb_weights.to(device=logits.device)

        logits, weight = self.regularize_logits(logits, targets.float(), weight)

        self.criterion.weight = weight
        return self.criterion(logits, targets)


def configure_loss_function(
    criterion_name: str,
    num_examples_per_cls: np.ndarray,
    num_training_examples: int,
):
    if criterion_name == "bce":
        loss_fct = BCEWithLogitsLoss()

    elif criterion_name == "pos-weighted-bce":

        def _get_pos_weight(num_examples_per_cls: np.ndarray, num_examples: int, smooth_factor: int = 1):
            # num_neg / num_pos
            with np.errstate(divide="ignore"):
                pos_weight = (num_examples - num_examples_per_cls) / num_examples_per_cls
            # fill inf
            # actually never used
            # pos_weight[pos_weight == np.inf] = 1.0
            pos_weight = np.where(np.isinf(pos_weight), 1.0, pos_weight)
            # 1 for no smoothing, 2 for square
            pos_weight = np.power(pos_weight, 1 / smooth_factor)
            # tensor
            # todo: might be other dtypes than fp32
            pos_weight = torch.from_numpy(pos_weight).float()
            return pos_weight

        # pos_weight by constant
        # pos_weight = torch.tensor(100.0)

        # pos_weight by num_neg / num_pos
        smooth_factor = 1
        pos_weight = _get_pos_weight(num_examples_per_cls, num_training_examples, smooth_factor=smooth_factor)
        # todo: better handle
        # pos_weight = pos_weight.to(model.device)
        pos_weight = pos_weight.cuda()

        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

    elif criterion_name == "weighted-bce":

        def _get_weight(num_examples_per_cls: np.ndarray, smooth_factor: int = 1):
            # compute weight
            with np.errstate(divide="ignore"):
                # 1 for no smoothing, 2 for square
                weight = 1.0 / np.power(num_examples_per_cls, 1 / smooth_factor)
            # fill inf
            # actually never used, but set to 0 for normalization
            weight = np.where(np.isinf(weight), 0.0, weight)
            # normalize to make np.sum(weight) == num_classes
            # to make the total loss roughly in the same scale
            weight = weight / np.sum(weight) * weight.shape[0]
            # convert to tensor
            # todo: might be other dtypes than fp32
            weight = torch.from_numpy(weight).float()
            return weight

        # np.save("tmp_num_positive_per_cls.npy", num_examples_per_cls)
        smooth_factor = 2
        weight = _get_weight(num_examples_per_cls, smooth_factor=smooth_factor)
        # todo: better handle
        # weight = weight.to(model.device)
        weight = weight.cuda()

        loss_fct = BCEWithLogitsLoss(weight=weight)

    elif criterion_name == "fl":

        # loss_fct = MultiLabelFocalLossWithLogits(weight=None, alpha=0.5, gamma=2.0, reduction="mean")
        loss_fct = MultiLabelFocalLossWithLogits(
            weight=5.0,
            alpha=0.5,
            gamma=2.0,
            reduction="mean",
        )

    elif criterion_name == "cb":

        loss_fct = ClassBalancedWrapper(
            BCEWithLogitsLoss(),
            num_samples=num_examples_per_cls,
            beta=0.99,
        )

    elif criterion_name == "cb-fl":

        loss_fct = ClassBalancedWrapper(
            MultiLabelFocalLossWithLogits(weight=None, alpha=0.5, gamma=2.0, reduction="mean"),
            num_samples=num_examples_per_cls,
            beta=0.99,
        )

    elif criterion_name == "ntr":

        loss_fct = NegativeTolerantRegularizationWrapper(
            BCEWithLogitsLoss(),
            num_samples=num_examples_per_cls,
            neg_scale=2.0,
            init_bias=0.05,
        )

    elif criterion_name == "ntr-fl":

        loss_fct = NegativeTolerantRegularizationWrapper(
            MultiLabelFocalLossWithLogits(weight=None, alpha=0.5, gamma=2.0, reduction="mean"),
            num_samples=num_examples_per_cls,
            neg_scale=2.0,
            init_bias=0.05,
        )

    elif criterion_name == "cb-ntr":

        loss_fct = ClassBalancedNegativeTolerantRegularizationWrapper(
            BCEWithLogitsLoss(),
            num_samples=num_examples_per_cls,
            beta=0.99,
            neg_scale=2.0,
            init_bias=0.05,
        )

    elif criterion_name == "cb-ntr-fl":

        loss_fct = ClassBalancedNegativeTolerantRegularizationWrapper(
            MultiLabelFocalLossWithLogits(weight=None, alpha=0.5, gamma=2.0, reduction="mean"),
            num_samples=num_examples_per_cls,
            beta=0.99,
            neg_scale=2.0,
            init_bias=0.05,
        )

    # elif criterion_name == "cb":

    #     def _get_pos_weight(num_examples_per_cls: np.ndarray, num_examples: int, beta: float = 0.99, smooth_factor: int = 1):
    #         def effective_number(num_samples):
    #             eff_num = (1 - beta) / (1 - beta**num_samples)
    #             eff_num[eff_num == np.inf] = 0
    #             return eff_num

    #         num_negative_per_cls = num_examples - num_examples_per_cls
    #         effective_num_positve_per_cls = effective_number(num_examples_per_cls)
    #         effective_num_negative_per_cls = effective_number(num_negative_per_cls)
    #         pos_weight = effective_num_negative_per_cls / (effective_num_positve_per_cls + 1e-6)
    #         pos_weight[pos_weight == np.inf] = num_examples
    #         # 1 for no smoothing, 2 for square
    #         pos_weight = np.power(pos_weight, 1 / smooth_factor)
    #         return pos_weight

    #     # N x n_classes -> n_classes
    #     num_examples_per_cls = np.array(train_dataset["label"]).sum(0)
    #     pos_weight = _get_pos_weight(num_examples_per_cls, train_dataset.num_rows, 0.99, 1)
    #     # todo: might be other dtypes than fp32
    #     pos_weight = torch.from_numpy(pos_weight).float().to(model.device)

    #     # loss_fct = BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))
    #     loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

    else:
        raise NotImplementedError(f"criterion_name {criterion_name} is not supported")

    return loss_fct
