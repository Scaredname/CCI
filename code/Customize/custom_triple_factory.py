"""
Author: Ni Runyu & MonkeyDC
Date: 2024-03-12 15:59:39
LastEditors: Ni Runyu & MonkeyDC
LastEditTime: 2024-03-12 16:02:53
Description: 

Copyright (c) 2024 by Ni Runyu, All Rights Reserved. 
"""

import logging
from typing import ClassVar, Dict, Mapping, Tuple

import numpy as np
import torch
from numpy import linalg as LA
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.typing import EntityMapping, LabeledTriples

logger = logging.getLogger(__name__)


def L1_normalize_each_rows_of_matrix(matrix: np.array) -> np.array:
    """
    description:
    param matrix: np.float32
    return {np.float32}
    """

    for i in range(matrix.shape[0]):
        if np.sum(abs(matrix[i]), dtype=np.float32) > 0:
            matrix[i] = matrix[i] / np.sum(abs(matrix[i]), dtype=np.float32)
    return matrix


def category_to_id(cate_triples: np.array):

    categories = np.unique(np.ndarray.flatten(cate_triples[:, 2]))
    category_to_id: Dict[str, int] = {
        value: key for key, value in enumerate(categories)
    }
    return category_to_id


def create_adjacency_matrix_of_entities_categories(
    cate_triples: np.array,
    entity_to_id: EntityMapping,
    category_to_id,
):
    """
    Create matrix where each row corresponds to an entity and each column to a cate.
    """
    # Prepare literal matrix, set every cate to zero, and afterwards fill in the corresponding value if available
    ents_cates_adj_matrix = np.zeros(
        [len(entity_to_id), len(category_to_id)], dtype=np.float32
    )

    for ent, _, cate in cate_triples:
        # row define entity, and column the cate
        try:
            ents_cates_adj_matrix[entity_to_id[ent], category_to_id[cate]] = 1
        except:
            # There are some entities not in training set
            pass

    return ents_cates_adj_matrix


class TripleswithCategory(TriplesFactory):
    file_name_cate_to_id: ClassVar[str] = "cate_to_id"
    file_name_cates: ClassVar[str] = "categories"

    def __init__(
        self,
        *,
        ents_cates_adj_matrix: np.ndarray,
        categories_to_ids: Mapping[str, int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.categories_to_ids = categories_to_ids
        self.ents_cates_adj_matrix = torch.from_numpy(ents_cates_adj_matrix)

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples=False,
        *,
        cate_triples: LabeledTriples = None,
        **kwargs,
    ) -> "TriplescatesFactory":
        if cate_triples is None:
            raise ValueError(f"{cls.__name__} requires cate_triples.")
        base = TriplesFactory.from_labeled_triples(
            triples=triples, create_inverse_triples=create_inverse_triples, **kwargs
        )

        # get entity and relation adjacency matrix
        categories_to_ids = category_to_id(cate_triples)
        ents_cates_adj_matrix = create_adjacency_matrix_of_entities_categories(
            cate_triples=cate_triples,
            entity_to_id=base.entity_to_id,
            category_to_id=categories_to_ids,
        )

        # Calculate the proportion of each cate.
        ents_cates_adj_matrix = L1_normalize_each_rows_of_matrix(ents_cates_adj_matrix)

        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            ents_cates_adj_matrix=ents_cates_adj_matrix,
            categories_to_ids=categories_to_ids,
        )

    @property
    def cate_shape(self) -> Tuple[int, ...]:
        """Return the shape of the cates."""
        return self.ents_cates_adj_matrix.shape[1:]

    @property
    def num_category(self) -> int:
        """Return the number of cates."""
        return self.ents_cates_adj_matrix.shape[1]
