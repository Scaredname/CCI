"""
Author: Ni Runyu & MonkeyDC
Date: 2024-03-12 15:59:39
LastEditors: Ni Runyu & MonkeyDC
LastEditTime: 2024-03-12 16:02:53
Description:

Copyright (c) 2024 by Ni Runyu, All Rights Reserved.
"""

import logging
import pathlib
from typing import Any, ClassVar, Dict, Mapping, MutableMapping, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.typing import EntityMapping, LabeledTriples
from collections import defaultdict

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

def create_ent_average_matrix(ents_cates_adj_matrx: np.ndarray):
    
    A = ents_cates_adj_matrx
    e_n = A.shape[0]
    degree = (A != 0).sum(axis=1)
    offsets = np.concatenate(([0], np.cumsum(degree)))[:-1]
    ent_average_matrix = np.zeros((e_n, degree.sum()), dtype=A.dtype)
    
    for i in range(e_n):
        nnz = degree[i]
        if nnz == 0:
            continue
        values = A[i, A[i] != 0]
        ent_average_matrix[i, offsets[i]:offsets[i]+nnz] = values
    return ent_average_matrix

def generate_ent_rel_fre(entity_to_id:dict[str, int], ent_pair_set:dict[str, set], relation_frequency:dict):
    # ent_rel_frequency = defaultdict(int)
    ent_bal_wei = torch.zeros(len(entity_to_id), dtype=float)
    for ent, id in entity_to_id.items():
        for p in ent_pair_set[ent]:
            _,r = p
            ent_bal_wei[id] += relation_frequency[r]
            
    return ent_bal_wei
    


class TripleswithCategory(TriplesFactory):
    file_name_category_to_id: ClassVar[str] = "cates_to_id"
    file_name_ents_cates: ClassVar[str] = "ents_cates_adj_matrix"
    cate_triples_file_name: ClassVar[str] = "cate_triples"
    file_name_entity_to_id: ClassVar[str] = "entity_to_id"
    file_name_relation_to_id: ClassVar[str] = "relation_to_id"

    def __init__(
        self,
        *,
        ents_cates_adj_matrix: np.ndarray,
        categories_to_ids: Mapping[str, int],
        ent_average_matrix: np.ndarray,
        ent_rel_fre: torch.FloatTensor,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.categories_to_ids = categories_to_ids
        self.ents_cates_adj_matrix = torch.from_numpy(ents_cates_adj_matrix)
        self.ent_average_matrix = torch.from_numpy(ent_average_matrix)
        self.ent_rel_fre = ent_rel_fre

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples=False,
        *,
        cate_triples: LabeledTriples = None,
        ent_pair_set=None,
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
        
        
        ent_average_matrix = create_ent_average_matrix(ents_cates_adj_matrix)
        
        relations = np.array(triples)[:, 1]
        labels, counts = np.unique(relations, return_counts=True)
        relations_frequency = dict(zip(labels, counts))
        
        ent_rel_fre = generate_ent_rel_fre(base.entity_to_id, ent_pair_set=ent_pair_set, relation_frequency=relations_frequency)

        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            ents_cates_adj_matrix=ents_cates_adj_matrix,
            categories_to_ids=categories_to_ids,
            ent_average_matrix = ent_average_matrix,
            ent_rel_fre = ent_rel_fre,
        )

    @property
    def category_shape(self) -> Tuple[int, ...]:
        """Return the shape of the cates."""
        return self.ents_cates_adj_matrix.shape[1:]

    @property
    def num_category(self) -> int:
        """Return the number of cates."""
        return self.ents_cates_adj_matrix.shape[1]

    def to_path_binary(
        self, path: Union[str, pathlib.Path, TextIO]
    ) -> pathlib.Path:  # noqa: D102
        path = super().to_path_binary(path=path)
        # store entity/relation to ID
        for name, data in (
            (
                self.file_name_category_to_id,
                self.categories_to_ids,
            ),
        ):
            pd.DataFrame(
                data=data.items(),
                columns=["label", "id"],
            ).sort_values(
                by="id"
            ).set_index("id").to_csv(
                path.joinpath(f"{name}.tsv.gz"),
                sep="\t",
            )

        np.savez_compressed(
            path.joinpath(f"{self.file_name_ents_cates}.npz"),
            self.ents_cates_adj_matrix,
        )

        return path

    @classmethod
    def _from_path_binary(cls, path: pathlib.Path) -> MutableMapping[str, Any]:
        data = super()._from_path_binary(path)
        # load entity/relation to ID
        # for name in [
        #     cls.file_name_category_to_id,
        # ]:
        df = pd.read_csv(
            path.joinpath(f"{cls.file_name_category_to_id}.tsv.gz"),
            sep="\t",
        )
        data["categories_to_ids"] = dict(zip(df["label"], df["id"]))

        data["ents_cates_adj_matrix"] = np.load(
            path.joinpath(f"{cls.file_name_ents_cates}.npz")
        )["arr_0"]

        print(data)

        return data
