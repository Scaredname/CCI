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
import pathlib
from collections import defaultdict
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from pykeen.constants import COLUMN_LABELS, TARGET_TO_INDEX
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.utils import load_triples
from pykeen.typing import (
    COLUMN_HEAD,
    COLUMN_RELATION,
    COLUMN_TAIL,
    LABEL_HEAD,
    LABEL_RELATION,
    LABEL_TAIL,
    EntityMapping,
    LabeledTriples,
    MappedTriples,
    RelationMapping,
)

logger = logging.getLogger(__name__)


def maximum_reciprocal(
    df: pd.DataFrame,
    source: str,
    target: str,
    mu: int = 3,
) -> Tuple[int, float]:
    """
    description:
        calculate the reciprocal of maximum of target per source
    param df:
        The dataframe.
    param source:
        The source column.
    param target:
        The target column.
    return {*}
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    return 1 / n_unique.max() ** mu


def average_target(
    df: pd.DataFrame,
    source: str,
    target: str,
    mu: int = 3,
) -> Tuple[int, float]:
    """
    description:
        calculate the reciprocal of maximum of target per source
    param df:
        The dataframe.
    param source:
        The source column.
    param target:
        The target column.
    return {*}
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    return n_unique.mean()


def calculate_injective_confidence(
    df: pd.DataFrame,
    source: str,
    target: str,
) -> Tuple[int, float]:
    """
    Calcualte the confidence of wheter there is injective mapping from source to target.

    :param df:
        The dataframe.
    :param source:
        The source column.
    :param target:
        The target column.

    :return:
        the relative frequency of unique target per source.
    """
    grouped = df.groupby(by=source)
    n_unique = grouped.agg({target: "nunique"})[target]
    conf = (n_unique <= 1).mean()
    return conf


def apply_label_smoothing(
    labels: torch.FloatTensor,
    epsilon: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> torch.FloatTensor:
    """Apply label smoothing to a target tensor.

    Redistributes epsilon probability mass from the true target uniformly to the remaining classes by replacing
        * a hard one by (1 - epsilon)
        * a hard zero by epsilon / (num_classes - 1)

    :param labels:
        The one-hot label tensor.
    :param epsilon:
        The smoothing parameter. Determines how much probability should be transferred from the true class to the
        other classes.
    :param num_classes:
        The number of classes.
    :returns: A smoothed label tensor
    :raises ValueError: if epsilon is negative or if num_classes is None
    """
    if not epsilon:  # either none or zero
        return labels
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be positive, but is {epsilon}")
    if num_classes is None:
        raise ValueError("must pass num_classes to perform label smoothing")

    new_label_true = 1.0 - epsilon
    new_label_false = epsilon / (num_classes - 1)
    return new_label_true * labels + new_label_false * (1.0 - labels)


def create_adjacency_matrix(
    triples: np.ndarray,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
    if_reverse: bool = False,
) -> np.ndarray:
    """Create adjacency matrix from triples."""
    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    if if_reverse:
        num_relations *= 2
    adjacency_matrix = np.zeros((2, num_entities, num_relations), dtype=np.float32)
    for head_id, relation_id, tail_id in triples:
        adjacency_matrix[0, entity_to_id[head_id], relation_to_id[relation_id]] = 1
        adjacency_matrix[1, entity_to_id[tail_id], relation_to_id[relation_id]] = 1
        if if_reverse:
            adjacency_matrix[
                0,
                entity_to_id[tail_id],
                relation_to_id[relation_id] + num_relations // 2,
            ] = 1
            adjacency_matrix[
                1,
                entity_to_id[head_id],
                relation_to_id[relation_id] + num_relations // 2,
            ] = 1
    return adjacency_matrix


def create_relation_injective_confidence(
    mapped_triples: Collection[Tuple[int, int, int]]
):
    """
    return: injective_confidence: [head2tail, tail2head]
    """
    injective_confidence = list()
    df = pd.DataFrame(data=mapped_triples, columns=COLUMN_LABELS)
    see_df = defaultdict(list)
    for relation, group in df.groupby(by=LABEL_RELATION):
        h_IJC = calculate_injective_confidence(
            df=group, source=LABEL_TAIL, target=LABEL_HEAD
        )
        t_IJC = calculate_injective_confidence(
            df=group, source=LABEL_HEAD, target=LABEL_TAIL
        )

        h_RMT = maximum_reciprocal(df=group, source=LABEL_TAIL, target=LABEL_HEAD)
        t_RMT = maximum_reciprocal(df=group, source=LABEL_HEAD, target=LABEL_TAIL)
        h_AT = average_target(df=group, source=LABEL_TAIL, target=LABEL_HEAD)
        t_AT = average_target(df=group, source=LABEL_HEAD, target=LABEL_TAIL)

        if h_IJC > (1 - h_RMT):
            h_confi = h_IJC
        else:
            h_confi = min(h_IJC, h_RMT)
        if t_IJC > (1 - t_RMT):
            t_confi = t_IJC
        else:
            t_confi = min(t_IJC, t_RMT)

        injective_confidence.append((h_confi, t_confi))
        see_df["h_IJC"].append(h_IJC)
        see_df["h_AT"].append(h_AT)
        see_df["h_confi"].append(h_confi)
        see_df["t_IJC"].append(t_IJC)
        see_df["t_AT"].append(t_AT)
        see_df["t_confi"].append(t_confi)

    return np.array(injective_confidence), pd.DataFrame(see_df)


def create_matrix_of_types(
    type_triples: np.array,
    ents_rels: np.array,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
    type_position: int = 0,
    if_reverse: bool = False,
):
    """
    Create matrix of literals where each row corresponds to an entity and each column to a type.
    """
    data_types = np.unique(np.ndarray.flatten(type_triples[:, type_position]))
    data_type_to_id: Dict[str, int] = {
        value: key for key, value in enumerate(data_types)
    }
    # Prepare literal matrix, set every type to zero, and afterwards fill in the corresponding value if available
    ents_types = np.zeros([len(entity_to_id), len(data_type_to_id)], dtype=np.float32)
    # 使用64位浮点数，因为后面要做除法，这样保证每个关系的类型比例之和为1
    if if_reverse:
        rels_types = np.zeros(
            [2, len(relation_to_id) * 2, len(data_type_to_id)], dtype=np.float64
        )
    else:
        rels_types = np.zeros(
            [2, len(relation_to_id), len(data_type_to_id)], dtype=np.float64
        )

    # TODO vectorize code
    if type_position == 0:
        for typ, rel, ent in type_triples:
            try:
                # row define entity, and column the type.
                ents_types[entity_to_id[ent], data_type_to_id[typ]] = 1
                # 所以实体作为尾实体共现时的关系也要计算
                for rel_id in np.where(ents_rels[0, entity_to_id[ent], :] == 1)[
                    0
                ]:  # 获得所有和该实体共现的关系id
                    rels_types[0, rel_id, data_type_to_id[typ]] += 1
                for rel_id in np.where(ents_rels[1, entity_to_id[ent], :] == 1)[0]:
                    rels_types[1, rel_id, data_type_to_id[typ]] += 1
            except KeyError:
                # 注意：存在只拥有实体类型却没有出现在训练数据的实体
                # logger.info(f"There is entity {ent} not in train triple")
                continue
    elif type_position == 2:
        for ent, rel, typ in type_triples:
            try:
                # row define entity, and column the type
                ents_types[entity_to_id[ent], data_type_to_id[typ]] = 1
                # 一个关系针对头尾实体应该拥有两个不同的类型权重矩阵
                # 令所有和这个实体相关的关系的类型都加1
                for rel_id in np.where(ents_rels[1, entity_to_id[ent], :] == 1)[0]:
                    rels_types[1, rel_id, data_type_to_id[typ]] += 1
                for rel_id in np.where(ents_rels[0, entity_to_id[ent], :] == 1)[0]:
                    rels_types[0, rel_id, data_type_to_id[typ]] += 1
            except KeyError:
                # 存在只拥有实体类型却没有出现在训练数据的实体
                # logger.info(f"There is entity {ent} not in train triple")
                continue

    return ents_types, rels_types, data_type_to_id


def crate_rel_type_related_ent(ents_types, rels_types):
    """
    return LongTensor
    """
    rels_related_h_ents = list()
    rels_related_t_ents = list()
    h_ent_max_num = 0
    t_ent_max_num = 0

    for rel_id in range(rels_types.shape[1]):
        rel_related_h_type = np.where(rels_types[0, rel_id] > 0)[0]
        rel_related_t_type = np.where(rels_types[1, rel_id] > 0)[0]

        t_related_h_ent = []
        t_related_t_ent = []
        for t_h_id in rel_related_h_type:
            type_related_h_ent = np.where(ents_types.T[t_h_id] > 0)[0]
            t_related_h_ent += type_related_h_ent.tolist()

        for t_t_id in rel_related_t_type:
            type_related_t_ent = np.where(ents_types.T[t_t_id] > 0)[0]
            t_related_t_ent += type_related_t_ent.tolist()

        h_ent_max_num = (
            len(t_related_h_ent)
            if h_ent_max_num < len(t_related_h_ent)
            else h_ent_max_num
        )
        rels_related_h_ents.append(t_related_h_ent)
        t_ent_max_num = (
            len(t_related_t_ent)
            if t_ent_max_num < len(t_related_t_ent)
            else t_ent_max_num
        )
        rels_related_t_ents.append(t_related_t_ent)

    rels_related_h_ents = torch.LongTensor(
        np.array(
            [
                np.pad(
                    ents, (0, h_ent_max_num - len(ents)), "constant", constant_values=-1
                )
                for ents in rels_related_h_ents
            ]
        )
    )
    rels_related_t_ents = torch.LongTensor(
        np.array(
            [
                np.pad(
                    ents, (0, t_ent_max_num - len(ents)), "constant", constant_values=-1
                )
                for ents in rels_related_t_ents
            ]
        )
    )

    return rels_related_h_ents, rels_related_t_ents


class TriplesTypesFactory(TriplesFactory):
    file_name_type_to_id: ClassVar[str] = "type_to_id"
    file_name_types: ClassVar[str] = "types"

    def __init__(
        self,
        *,
        ents_types: np.ndarray,
        rels_types: np.ndarray,
        rels_inj_conf: np.ndarray,
        rel_related_ent: list[torch.tensor, torch.tensor],
        types_to_id: Mapping[str, int],
        type_smoothing: float = 0.0,
        use_random_weights: bool = False,
        select_one_type: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ents_types = ents_types
        self.ents_types_mask = torch.where(
            torch.tensor(ents_types) != 0, torch.tensor(1), torch.tensor(0)
        )
        self.rels_types = rels_types
        self.rels_types_mask = torch.where(
            torch.tensor(rels_types) != 0, torch.tensor(1), torch.tensor(0)
        )
        self.types_to_id = types_to_id
        self.assignments = self._get_assignment(ents_types)
        self.rels_inj_conf = rels_inj_conf
        self.rel_related_ent = rel_related_ent
        # Calculate the proportion of each type.
        self.ents_types = self._cal_propor(self.ents_types)
        self.rels_types[0] = self._cal_propor(self.rels_types[0])
        self.rels_types[1] = self._cal_propor(self.rels_types[1])

        if type_smoothing:
            self.ents_types = apply_label_smoothing(
                self.ents_types, type_smoothing, ents_types.shape[1]
            )
            self.rels_types = apply_label_smoothing(
                self.rels_types, type_smoothing, rels_types.shape[2]
            )
            print("type smoothing applied")

        self.ents_types = torch.from_numpy(self.ents_types)
        self.rels_types = torch.from_numpy(self.rels_types)

        if use_random_weights:
            self.ents_types = torch.rand_like(self.ents_types)
            self.rels_types = torch.rand_like(self.rels_types)

        if select_one_type:
            # 只保留最相关的实体类型
            max_values_ent, _ = torch.max(self.ents_types, dim=1)
            max_values_rel, _ = torch.max(self.rels_types, dim=2)

            ones_ents_types = torch.ones_like(self.ents_types)
            ones_rels_types = torch.ones_like(self.rels_types)

            # 将全1矩阵中每一行的最大值所在位置的元素设为1，其他元素保持为0
            ents_type_result = torch.where(
                self.ents_types == max_values_ent.unsqueeze(1),
                ones_ents_types,
                torch.zeros_like(self.ents_types),
            )
            rels_type_result = torch.where(
                self.rels_types == max_values_rel.unsqueeze(2),
                ones_rels_types,
                torch.zeros_like(self.rels_types),
            )

            # 如果存在多个最大值，随机选择一个最大值的位置，并将其设为1
            max_indices_ent = torch.multinomial(ents_type_result, num_samples=1)
            max_indices_rel_h = torch.multinomial(rels_type_result[0], num_samples=1)
            max_indices_rel_t = torch.multinomial(rels_type_result[1], num_samples=1)
            ents_type_result.zero_()
            rels_type_result.zero_()
            ents_type_result.scatter_(1, max_indices_ent, 1)
            rels_type_result[0].scatter_(1, max_indices_rel_h, 1)
            rels_type_result[1].scatter_(1, max_indices_rel_t, 1)

            self.ents_types = ents_type_result
            self.rels_types = rels_type_result

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples=False,
        type_smoothing=0.0,
        *,
        type_triples: LabeledTriples = None,
        type_position: int = 0,
        use_random_weights: bool = False,
        select_one_type: bool = False,
        strict_confidence: bool = False,
        **kwargs,
    ) -> "TriplesTypesFactory":
        if type_triples is None:
            raise ValueError(f"{cls.__name__} requires type_triples.")
        base = TriplesFactory.from_labeled_triples(
            triples=triples, create_inverse_triples=create_inverse_triples, **kwargs
        )

        # get entity and relation adjacence matrix
        ents_rels_adj = create_adjacency_matrix(
            triples=triples,
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            if_reverse=create_inverse_triples,
        )

        ents_types, rels_types, types_to_id = create_matrix_of_types(
            type_triples=type_triples,
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            type_position=type_position,
            ents_rels=ents_rels_adj,
            if_reverse=create_inverse_triples,
        )

        rel_related_ent = crate_rel_type_related_ent(
            ents_types=ents_types, rels_types=rels_types
        )

        (
            relation_injective_confidence,
            see_confindence,
        ) = create_relation_injective_confidence(base.mapped_triples)

        # 为了展示使用的数据
        # see_confindence.insert(see_confindence.shape[1], 'r', list(base.relation_to_id.keys()))
        # see_confindence.to_csv('/home/ni/confi_examples.csv', index = False)

        if strict_confidence:
            relation_injective_confidence[relation_injective_confidence < 1] = 0

        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            ents_types=ents_types,
            rels_types=rels_types,
            types_to_id=types_to_id,
            rels_inj_conf=relation_injective_confidence,
            type_smoothing=type_smoothing,
            use_random_weights=use_random_weights,
            select_one_type=select_one_type,
            rel_related_ent=rel_related_ent,
        )

    def class_calculate_injective_confidence(self, mapped_triples, stricit=False):
        # 这里额外用一个函数来方便我们使用所有数据来判断cardinality.
        self.rels_inj_conf, _ = create_relation_injective_confidence(mapped_triples)
        if stricit:
            self.rels_inj_conf[self.rels_inj_conf < 1] = 0
        print("using all data to generate injective confidence")

    @property
    def type_shape(self) -> Tuple[int, ...]:
        """Return the shape of the types."""
        return self.ents_types.shape[1:]

    def _cal_propor(self, data):
        """Calculate the proportion of each type."""
        for i in range(data.shape[0]):
            if np.sum(data[i], dtype=np.float32) > 0:
                data[i] = data[i] / np.sum(data[i], dtype=np.float32)
        return data

    @property
    def num_types(self) -> int:
        """Return the number of types."""
        return self.ents_types.shape[1]

    def _get_assignment(self, ents_types) -> np.ndarray:
        """Mean the ents_type matrix and Return the assignment, shape:[max_id, 2]."""
        assignment_list = list()
        for i in range(self.ents_types.shape[0]):
            types_id_ent = np.where(self.ents_types[i] == 1)[0]

            if len(types_id_ent) > 0:
                assignment_list.append(0)
            else:
                assignment_list.append(1)

        # self.ents_types = torch.as_tensor(self.ents_types, dtype=torch.cfloat)

        return torch.as_tensor(np.array(assignment_list))
