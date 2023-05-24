import logging
import math
from collections import defaultdict
from typing import (Callable, DefaultDict, Iterable, List, Mapping,
                    MutableMapping, NamedTuple, Optional, Sequence, Tuple,
                    Type, TypeVar, Union, cast)

import numpy as np
import torch
from pykeen.constants import (COLUMN_LABELS, TARGET_TO_KEY_LABELS,
                              TARGET_TO_KEYS)
from pykeen.evaluation import MacroRankBasedEvaluator, RankBasedEvaluator
from pykeen.evaluation.rank_based_evaluator import RankBasedMetricResults
from pykeen.typing import (LABEL_HEAD, LABEL_TAIL, RANK_OPTIMISTIC,
                           RANK_PESSIMISTIC, RANK_REALISTIC, RANK_TYPES,
                           SIDE_BOTH, ExtendedTarget, MappedTriples, RankType,
                           Target)

K = TypeVar("K")
logger = logging.getLogger(__name__)

def _flatten(nested: Mapping[K, Sequence[np.ndarray]]) -> Mapping[K, np.ndarray]:
    return {key: np.concatenate(value) for key, value in nested.items()}

class RankPack(NamedTuple):
    """A pack of ranks for aggregation."""

    target: ExtendedTarget
    rank_type: RankType
    ranks: np.ndarray
    num_candidates: np.ndarray
    weights: Optional[np.ndarray]

    def resample(self, seed: Optional[int]) -> "RankPack":
        """Resample rank pack."""
        generator = np.random.default_rng(seed=seed)
        n = len(self.ranks)
        ids = generator.integers(n, size=(n,))
        weights = None if self.weights is None else self.weights[ids]
        return RankPack(
            target=self.target,
            rank_type=self.rank_type,
            ranks=self.ranks[ids],
            num_candidates=self.num_candidates[ids],
            weights=weights,
        )

def _iter_ranks(
    ranks: Mapping[Tuple[Target, RankType], Sequence[np.ndarray]],
    num_candidates: Mapping[Target, Sequence[np.ndarray]],
    weights: Optional[Mapping[Target, Sequence[np.ndarray]]] = None,
    changed_ranks_flat: Mapping[K, np.ndarray] = None,
    changerank: bool = False,
) -> Iterable[RankPack]:
    # terminate early if there are no ranks
    if not ranks:
        logger.debug("Empty ranks. This should only happen during size probing.")
        return

    sides = sorted(num_candidates.keys())
    # flatten dictionaries
    if not changerank:
        ranks_flat = _flatten(ranks)
    else:
        ranks_flat =  changed_ranks_flat
    num_candidates_flat = _flatten(num_candidates)
    weights_flat: Mapping[Target, np.ndarray]
    if weights is None:
        weights_flat = dict()
    else:
        weights_flat = _flatten(weights)
        
    for rank_type in RANK_TYPES:
        # individual side
        for side in sides:
            yield RankPack(
                side, rank_type, ranks_flat[side, rank_type], num_candidates_flat[side], weights_flat.get(side)
            )

        # combined
        c_ranks = np.concatenate([ranks_flat[side, rank_type] for side in sides])

        c_num_candidates = np.concatenate([num_candidates_flat[side] for side in sides])

        c_weights = None if weights is None else np.concatenate([weights_flat[side] for side in sides])

        yield RankPack(SIDE_BOTH, rank_type, c_ranks, c_num_candidates, c_weights)


class IRRankBasedEvaluator(RankBasedEvaluator):
    """Information retrieval rank-based evaluation."""

    weights: MutableMapping[Target, List[np.ndarray]]

    def __init__(self, **kwargs):
        """
        Initialize the evaluator.

        :param kwargs:
            additional keyword-based parameters passed to :meth:`RankBasedEvaluator.__init__`.
        """
        super().__init__(**kwargs)
        self.keys = defaultdict(list)
    
    @staticmethod
    def _change_rank(ks: defaultdict[list], ranks: Mapping[Tuple[Target, RankType], Sequence[np.ndarray]]):
        """Change the rank.

        :param keys:
            the keys, in batches

        :return: Mapping[K, np.ndarray]
            the flatten ranks
        """
        ranks_flat = _flatten(ranks)
        for target, keys in ks.items():
            keys = np.concatenate(list(keys), axis=0)
            
            keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            for i in range(len(keys)):
                p = np.where(inverse==i)
                if len(p[0]) > 1:
                    for each in ranks_flat:
                        if target == each[0]:
                            min_rank = np.min(ranks_flat[each][p[0]])
                            ranks_flat[each][p[0]] = min_rank
        return ranks_flat

    # docstr-coverage: inherited
    def process_scores_(
        self,
        hrt_batch: MappedTriples,
        target: Target,
        scores: torch.FloatTensor,
        true_scores: Optional[torch.FloatTensor] = None,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        super().process_scores_(
            hrt_batch=hrt_batch,
            target=target,
            scores=scores,
            true_scores=true_scores,
            dense_positive_mask=dense_positive_mask,
        )
        # store keys for calculating macro weights
        self.keys[target].append(hrt_batch[:, TARGET_TO_KEYS[target]].detach().cpu().numpy())

    # docstr-coverage: inherited
    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        if self.num_entities is None:
            raise ValueError
        # compute macro weights
        # note: we wrap the array into a list to be able to re-use _iter_ranks
        
        changed_ranks = self._change_rank(ks=self.keys, ranks=self.ranks)
        # calculate weighted metrics
        result = RankBasedMetricResults.from_ranks(
        metrics=self.metrics,
        rank_and_candidates=_iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates, changed_ranks_flat=changed_ranks, changerank=True),
    )
        # Clear buffers
        self.keys.clear()
        self.ranks.clear()
        self.num_candidates.clear()

        return result
    