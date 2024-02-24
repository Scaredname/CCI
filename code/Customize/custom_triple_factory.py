"""
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-07-28 17:05:45
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-08-20 14:05:54
FilePath: /ESETC/code/Custom/CustomTripleFactory.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import logging
from typing import ClassVar, Dict, Mapping, Tuple

import numpy as np
import torch
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.typing import EntityMapping, LabeledTriples

logger = logging.getLogger(__name__)


def cal_propor(data):
    """Calculate the proportion of each cate."""
    for i in range(data.shape[0]):
        if np.sum(data[i], dtype=np.float32) > 0:
            data[i] = data[i] / np.sum(data[i], dtype=np.float32)
    return data


def create_matrix_of_cates(
    cate_triples: np.array,
    entity_to_id: EntityMapping,
):
    """
    Create matrix of literals where each row corresponds to an entity and each column to a cate.
    """
    data_cates = np.unique(np.ndarray.flatten(cate_triples[:, 2]))
    data_cate_to_id: Dict[str, int] = {
        value: key for key, value in enumerate(data_cates)
    }
    # Prepare literal matrix, set every cate to zero, and afterwards fill in the corresponding value if available
    ents_cates = np.zeros([len(entity_to_id), len(data_cate_to_id)], dtype=np.float32)

    for ent, _, cate in cate_triples:
        # row define entity, and column the cate
        try:
            ents_cates[entity_to_id[ent], data_cate_to_id[cate]] = 1
        except:
            # There are some entities not in training set
            pass

    return ents_cates, data_cate_to_id


class TripleswithCategory(TriplesFactory):
    file_name_cate_to_id: ClassVar[str] = "cate_to_id"
    file_name_cates: ClassVar[str] = "categories"

    def __init__(
        self,
        *,
        ents_cates: np.ndarray,
        cates_to_id: Mapping[str, int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cates_to_id = cates_to_id
        self.ents_cates = torch.from_numpy(ents_cates)

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

        # get entity and relation adjacence matrix

        ents_cates, cates_to_id = create_matrix_of_cates(
            cate_triples=cate_triples,
            entity_to_id=base.entity_to_id,
        )

        # Calculate the proportion of each cate.
        ents_cates = cal_propor(ents_cates)

        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            ents_cates=ents_cates,
            cates_to_id=cates_to_id,
        )

    @property
    def cate_shape(self) -> Tuple[int, ...]:
        """Return the shape of the cates."""
        return self.ents_cates.shape[1:]

    @property
    def num_cates(self) -> int:
        """Return the number of cates."""
        return self.ents_cates.shape[1]
