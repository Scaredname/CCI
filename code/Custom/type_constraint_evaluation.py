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
from pykeen.evaluation.ranks import Ranks
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

    def __init__(
        self,
        ents_types: torch.Tensor,
        rels_types: torch.Tensor,
        entity_match: bool = False,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`RankBasedEvaluator.__init__`.
        """
        super().__init__(**kwargs)
        self.ents_types = torch.tensor(ents_types)
        self.rels_types = torch.tensor(rels_types)
        self.batch_ranks = defaultdict(list)
        self.entity_match = entity_match

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

        if self.entity_match:
            if target == "head":
                constraint_score = 100 * (
                    self.ents_types[None] * self.ents_types[t].unsqueeze(dim=1)
                ).sum(-1).clamp(0, 1)
            elif target == "tail":
                constraint_score = 100 * (
                    self.ents_types[None] * self.ents_types[h].unsqueeze(dim=1)
                ).sum(-1).clamp(0, 1)

            true_constraint_score = 100 * (self.ents_types[h] * self.ents_types[t]).sum(
                -1
            ).clamp(0, 1)

        else:
            if target == "head":
                constraint_score = 100 * (
                    self.ents_types[None] * self.rels_types[0][r].unsqueeze(dim=1)
                ).sum(-1).clamp(0, 1)
                true_constraint_score = 100 * (
                    self.ents_types[h] * self.rels_types[0][r]
                ).sum(-1).clamp(0, 1)
            elif target == "tail":
                constraint_score = 100 * (
                    self.ents_types[None] * self.rels_types[1][r].unsqueeze(dim=1)
                ).sum(-1).clamp(0, 1)
                true_constraint_score = 100 * (
                    self.ents_types[t] * self.rels_types[1][r]
                ).sum(-1).clamp(0, 1)
        constraint_score = constraint_score.to(scores.device)
        true_constraint_score = true_constraint_score.to(scores.device)

        scores += constraint_score
        true_scores += true_constraint_score.view(*true_scores.shape)
        if true_scores is None:
            raise ValueError(f"{self.__class__.__name__} needs the true scores!")
        batch_ranks = Ranks.from_scores(
            true_score=true_scores,
            all_scores=scores,
        )
        self.num_entities = scores.shape[1]
        for rank_type, v in batch_ranks.items():
            ranks_np = v.detach().cpu().numpy()
            self.ranks[target, rank_type].append(ranks_np)
            for i in range(len(ranks_np)):
                self.batch_ranks[target, rank_type].append(
                    (hrt_batch.numpy()[i], ranks_np[i])
                )
        self.num_candidates[target].append(
            batch_ranks.number_of_options.detach().cpu().numpy()
        )
