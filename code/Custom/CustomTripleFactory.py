import logging
import pathlib
from typing import (Any, ClassVar, Dict, Iterable, Mapping, MutableMapping,
                    Optional, TextIO, Tuple, Union)

import numpy as np
import pandas
import torch
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.triples.utils import load_triples
from pykeen.typing import (EntityMapping, LabeledTriples, MappedTriples,
                           RelationMapping)

logger = logging.getLogger(__name__)

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
            adjacency_matrix[0, entity_to_id[tail_id], relation_to_id[relation_id] + num_relations // 2] = 1
            adjacency_matrix[1, entity_to_id[head_id], relation_to_id[relation_id] + num_relations // 2] = 1
    return adjacency_matrix

def create_matrix_of_types(
    type_triples: np.array,
    ents_rels: np.array,
    entity_to_id: EntityMapping,
    relation_to_id: RelationMapping,
    type_position: int = 0,
    if_reverse: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create matrix of literals where each row corresponds to an entity and each column to a type."""
    data_types = np.unique(np.ndarray.flatten(type_triples[:, type_position]))
    data_type_to_id: Dict[str, int] = {value: key for key, value in enumerate(data_types)}
    # Prepare literal matrix, set every type to zero, and afterwards fill in the corresponding value if available
    ents_types = np.zeros([len(entity_to_id), len(data_type_to_id)], dtype=np.float32)
    #使用64位浮点数，因为后面要做除法，这样保证每个关系的类型比例之和为1
    if if_reverse:
        rels_types = np.zeros([2, len(relation_to_id)*2, len(data_type_to_id)], dtype=np.float64)
    else:
        rels_types = np.zeros([2, len(relation_to_id), len(data_type_to_id)], dtype=np.float64)

    # TODO vectorize code
    if type_position == 0:
        for typ, rel, ent in type_triples:
            try:
                # row define entity, and column the type.
                ents_types[entity_to_id[ent], data_type_to_id[typ]] = 1
                # 所以实体作为尾实体共现时的关系也要计算
                for rel_id in np.where(ents_rels[0, entity_to_id[ent], :] == 1)[0]:
                    rels_types[0, rel_id, data_type_to_id[typ]] += 1
                for rel_id in np.where(ents_rels[1, entity_to_id[ent], :] == 1)[0]:
                    rels_types[1, rel_id, data_type_to_id[typ]] += 1
            except KeyError:
                logger.info("Either entity or relation to type doesn't exist.")
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
                logger.info("Either entity or relation to type doesn't exist.")
                continue

    return ents_types, rels_types, data_type_to_id

class TriplesTypesFactory(TriplesFactory):
    file_name_type_to_id: ClassVar[str] = "type_to_id"
    file_name_types: ClassVar[str] = "types"

    def __init__(
        self,
        *,
        ents_types: np.ndarray,
        rels_types: np.ndarray,
        types_to_id: Mapping[str, int],
        type_smoothing: float = 0.0,
        use_random_weights: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ents_types = ents_types
        self.ents_types_mask = torch.where(torch.tensor(ents_types) != 0, torch.tensor(1), torch.tensor(0))
        self.rels_types = rels_types
        self.rels_types_mask = torch.where(torch.tensor(rels_types) != 0, torch.tensor(1), torch.tensor(0))
        self.types_to_id = types_to_id
        self.assignments = self._get_assignment(ents_types)

        # Calculate the proportion of each type.
        self.ents_types = self._cal_propor(self.ents_types)
        self.rels_types[0] = self._cal_propor(self.rels_types[0])
        self.rels_types[1] = self._cal_propor(self.rels_types[1])

        if type_smoothing:
            self.ents_types = apply_label_smoothing(self.ents_types, type_smoothing, ents_types.shape[1])
            self.rels_types = apply_label_smoothing(self.rels_types, type_smoothing, rels_types.shape[2])
            print("type smoothing applied")
        
        self.ents_types = torch.from_numpy(self.ents_types)
        self.rels_types = torch.from_numpy(self.rels_types)

        if use_random_weights:
            self.ents_types = torch.rand_like(self.ents_types)
            self.rels_types = torch.rand_like(self.rels_types)
            

    @classmethod
    def from_labeled_triples(
        cls,
        triples: LabeledTriples,
        create_inverse_triples = False,
        type_smoothing = 0.0,
        *,
        type_triples: LabeledTriples = None,
        type_position: int = 0,
        use_random_weights: bool = False,
        **kwargs,
    ) -> "TriplesTypesFactory":
        if type_triples is None:
            raise ValueError(f"{cls.__name__} requires type_triples.")
        base = TriplesFactory.from_labeled_triples(triples=triples, create_inverse_triples=create_inverse_triples, **kwargs)
        
        # get entity and relation adjacence matrix
        ents_rels_adj = create_adjacency_matrix(triples=triples, entity_to_id=base.entity_to_id, relation_to_id=base.relation_to_id, if_reverse=create_inverse_triples)

        ents_types, rels_types, types_to_id = create_matrix_of_types(
            type_triples=type_triples, entity_to_id=base.entity_to_id, relation_to_id=base.relation_to_id, type_position=type_position, ents_rels=ents_rels_adj, if_reverse=create_inverse_triples
        )

        return cls(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
            ents_types=ents_types,
            rels_types=rels_types,
            types_to_id=types_to_id,
            type_smoothing=type_smoothing,
            use_random_weights=use_random_weights,
        )
    
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