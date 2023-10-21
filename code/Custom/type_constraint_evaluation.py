import logging
import math
from collections import defaultdict
from typing import (
    Callable,
    DefaultDict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from pykeen.constants import COLUMN_LABELS, TARGET_TO_KEY_LABELS, TARGET_TO_KEYS
from pykeen.evaluation import MacroRankBasedEvaluator, RankBasedEvaluator
from pykeen.evaluation.rank_based_evaluator import RankBasedMetricResults
from pykeen.typing import (
    LABEL_HEAD,
    LABEL_TAIL,
    RANK_OPTIMISTIC,
    RANK_PESSIMISTIC,
    RANK_REALISTIC,
    RANK_TYPES,
    SIDE_BOTH,
    ExtendedTarget,
    MappedTriples,
    RankType,
    Target,
)

K = TypeVar("K")
logger = logging.getLogger(__name__)


def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}


class TypeConstraintEvaluator(RankBasedEvaluator):
    """Information retrieval rank-based evaluation."""

    weights: MutableMapping[Target, List[np.ndarray]]

    def __init__(self, ents_types: torch.Tensor, rels_types: torch.Tensor, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`RankBasedEvaluator.__init__`.
        """
        super().__init__(**kwargs)
        self.ents_types = torch.tensor(ents_types)
        self.rels_types = torch.tensor(rels_types)

    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        hrt_batch = hrt_batch.detach().cpu()
        h = hrt_batch[:, 0]
        r = hrt_batch[:, 1]
        t = hrt_batch[:, 2]

        if target == "head":
            constraint_score = 100 * (
                self.ents_types[None] * self.rels_types[0][r].unsqueeze(dim=1)
            ).sum(-1).clamp(0, 1)
        elif target == "tail":
            constraint_score = 100 * (
                self.ents_types[None].squeeze() * self.rels_types[1][r].unsqueeze(dim=1)
            ).sum(-1).clamp(0, 1)
        constraint_score = constraint_score.to(scores.device)

        scores += constraint_score
        true_scores += 100
        super().process_scores_(
            hrt_batch=hrt_batch,
            target=target,
            scores=scores,
            true_scores=true_scores,
            dense_positive_mask=dense_positive_mask,
        )
