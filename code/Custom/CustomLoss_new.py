"""
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-07-16 15:42:35
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-23 15:16:35
FilePath: /ESETC/code/Custom/CustomLoss.py
Description: 新的损失函数

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
# Soft Commonsense(type)-Aware Negative Sampling

from typing import Optional

import torch
from pykeen.losses import (
    NSSALoss,
    UnsupportedLabelSmoothingError,
    prepare_negative_scores_for_softmax,
)
from torch.nn import functional as F


class NewSoftTypeawareNegativeSmapling(NSSALoss):
    """
    Calculate the weight by injective confidence and type relatedness.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        injective_confidence: torch.FloatTensor,
        type_relatedness: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        # injective_confidence and type_relatedness should belong [0, 1]
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        type_relatedness = (
            2 * F.sigmoid(type_relatedness.data) - 1
        )  # scaled to [0,1], no grad
        STNS_weights = injective_confidence * type_relatedness + (
            1 - injective_confidence
        ) * (1 - type_relatedness)

        negative_scores = prepare_negative_scores_for_softmax(
            batch_filter=batch_filter,
            negative_scores=negative_scores,
            # we do not allow full -inf rows, since we compute the softmax over this tensor
            no_inf_rows=True,
        )

        STNS_weights = STNS_weights.view(*negative_scores.shape)
        negative_scores = negative_scores * STNS_weights.detach()

        # compute weights (without gradient tracking)
        assert negative_scores.ndimension() == 2
        weights = (
            negative_scores.detach()
            .mul(self.inverse_softmax_temperature)
            .softmax(dim=-1)
        )

        # weights = weights * STNS_weights.detach() # 不计算STNS的梯度

        # fill negative scores with some finite value, e.g., 0 (they will get masked out anyway)
        negative_scores = torch.masked_fill(
            negative_scores, mask=~torch.isfinite(negative_scores), value=0.0
        )

        return self(
            pos_scores=positive_scores,
            neg_scores=negative_scores,
            neg_weights=weights,
            label_smoothing=label_smoothing,
            num_entities=num_entities,
        )