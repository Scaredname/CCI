import math
from typing import Optional

import numpy as np
import torch
from pykeen.constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX
from pykeen.losses import Loss
from pykeen.models.base import Model
from pykeen.training.slcwa import SLCWATrainingLoop
from pykeen.triples.instances import SLCWABatch
from pykeen.typing import InductiveMode


def get_negatives_direction(num_negs_per_pos, shape):
    """
    :param num_negs_per_pos:
    :return: list of direction of negative sampling. 0 represents tail to head. 1 represents head to tail.
    """
    negs_dire = np.zeros((shape, 2))
    # corruption_indices = [0, 1] * (
    #     shape // num_negs_per_pos
    # )  # 0: head2tail, 1: tail2head
    # split_idx = int(math.ceil(shape / len(corruption_indices)))
    # for index, start in zip(corruption_indices, range(0, shape, split_idx)):
    #     stop = min(start + split_idx, shape)
    #     negs_dire[start:stop] = index
    corruption_indices = [(1, 0), (0, 1)] * (
        shape // num_negs_per_pos
    )  # 0: head2tail, 1: tail2head
    split_idx = int(math.ceil(shape / len(corruption_indices)))
    for index, start in zip(corruption_indices, range(0, shape, split_idx)):
        stop = min(start + split_idx, shape)
        negs_dire[start:stop] = index
    return negs_dire


class TypeSLCWATrainingLoop(SLCWATrainingLoop):
    """
    SLCWA training loop with injective confidence and type relatedness.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.index_num = 0
        self.negs_dire = None

    @staticmethod
    def _process_batch_static(
        model: Model,
        loss: Loss,
        mode: Optional[InductiveMode],
        batch: SLCWABatch,
        start: Optional[int],
        stop: Optional[int],
        negs_dire: torch.Tensor,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=model.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
            positive_filter = positive_filter.to(model.device)

        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(model.device)

        # Compute negative and positive scores
        positive_scores, _, _ = model.score_hrt(positive_batch, mode=mode)

        negative_scores, injective_confidence, type_relatedness = model.score_hrt(
            negative_batch, mode=mode
        )

        negative_scores = negative_scores.view(negative_score_shape)
        # 负例方向，头到尾还是尾到头。

        # type_r = type_relatedness[indices, negs_dire]
        type_r = (type_relatedness * negs_dire).sum(-1)
        # negs_direction = negs_dire.detach()
        # injective_conf = injective_confidence[indices, negs_dire]
        injective_conf = (injective_confidence * negs_dire).sum(-1)
        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                injective_confidence=injective_conf,
                type_relatedness=type_r,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=model._get_entity_len(mode=mode),
            )
            + model.collect_regularization_term()
        )

    def _process_batch(
        self,
        batch: SLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        positive_batch, negative_batch, positive_filter = batch
        index_num = (stop - start) * negative_batch.shape[1]
        # num_negs_per_pos = negative_batch.shape[1] injective_confidence.shape[0] = (stop - start) * negative_batch.shape[1]
        if index_num != self.index_num:
            self.negs_dire = torch.tensor(
                get_negatives_direction(
                    negative_batch.shape[1],
                    (stop - start) * negative_batch.shape[1],
                )
            ).to(self.model.device)
            self.index_num = index_num
        return self._process_batch_static(
            model=self.model,
            loss=self.loss,
            mode=self.mode,
            batch=batch,
            start=start,
            stop=stop,
            label_smoothing=label_smoothing,
            slice_size=slice_size,
            negs_dire=self.negs_dire,
        )
