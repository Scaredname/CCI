'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-07-12 13:14:46
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-13 11:15:28
FilePath: /ESETC/code/Custom/HAKE.py
Description: 基于pykeen实现HAKE

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from typing import Any, ClassVar, Mapping, MutableMapping

import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs
from pykeen.models.nbase import ERModel
from pykeen.nn.modules import FunctionalInteraction
from pykeen.regularizers import Regularizer
from pykeen.triples import KGInfo
from pykeen.typing import Constrainer, Hint, Initializer
from pykeen.utils import estimate_cost_of_sequence


def HAKE_entity_initialize(tensor, bound):
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor

def HAKE_relation_initialize(tensor, bound):
    torch.nn.init.uniform_(tensor, -bound, bound)
    
    try:
        dim = tensor.shape[1] // 3
    except:
        ValueError('relation shape is not correct, is ', tensor.shape)
    torch.nn.init.ones_(tensor[:, dim : 2*dim])
    torch.nn.init.zeros_(tensor[:, 2*dim : 3*dim])
    return tensor

def hake_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
    bound: torch.FloatTensor,
    phase_weight: torch.FloatTensor,
    modulus_weight: torch.FloatTensor,
) -> torch.FloatTensor:
    """the HAKE interaction function.

    :param h: shape: (`*batch_dims`, dim * 2)
        The head representations.
    :param r: shape: (`*batch_dims`, dim * 3)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim * 2)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """
    pi = np.pi
    if h.shape[-1] != 2:
        h = h.view(*h.shape[:-1], -1, 2)
    if t.shape[-1] != 2:
        t = t.view(*t.shape[:-1], -1, 2)
    if r.shape[-1] != 3:
        r = r.view(*r.shape[:-1], -1, 3)
    phase_head, mod_head = torch.chunk(h, 2, dim=-1)
    phase_relation, mod_relation, bias_relation = torch.chunk(r, 3, dim=-1)
    phase_tail, mod_tail = torch.chunk(t, 2, dim=-1)

    phase_head = phase_head / (bound / pi)
    phase_relation = phase_relation / (bound  / pi)
    phase_tail = phase_tail / (bound / pi)


    # 适配1-N测试方法的shape
    if len(phase_head.shape) > 2:
        phase_relation.unsqueeze(dim=1)
        phase_tail.unsqueeze(dim=1)
    if len(phase_tail.shape) > 2:
        phase_relation.unsqueeze(dim=1)
        phase_head.unsqueeze(dim=1)

    if estimate_cost_of_sequence(h.shape, r.shape) < estimate_cost_of_sequence(r.shape, t.shape):
    # 当h和r的计算量小于r和t的计算量时，说明此时我们替换的是尾实体，也就是原Rotate代码中的tail-batch
        phase_score = (phase_head + phase_relation) - phase_tail
    else:
        phase_score = phase_head + (phase_relation - phase_tail)
    
    mod_relation = torch.abs(mod_relation)
    bias_relation = torch.clamp(bias_relation, max=1)
    indicator = (bias_relation < -mod_relation)
    bias_relation[indicator] = -mod_relation[indicator]
    
    r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

    phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * phase_weight
    r_score = torch.norm(r_score, dim=2) * modulus_weight
    return - (phase_score + r_score).sum(dim=-1)

class HAKEInteraction(FunctionalInteraction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    func = hake_interaction

    def __init__(self, bound, phase_weight, modulus_weight):
        super().__init__()
        self.bound = torch.tensor(bound)
        self.phase_weight = torch.tensor(phase_weight)
        self.modulus_weight = torch.tensor(modulus_weight)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        return dict(bound=self.bound, phase_weight=self.phase_weight, modulus_weight=self.modulus_weight)
    
class HAKEModel(ERModel):
    """
    HAKE model.
    """
    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = HAKE_entity_initialize,
        relation_initializer: Hint[Initializer] = HAKE_relation_initialize,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        lm : float = 9.0,
        phase_weight : float = 0.5,
        modulus_weight : float = 0.5,
        **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param embedding_dim:
            the embedding dimension
        :param entity_initializer:
            the entity representation initializer
        :param relation_initializer:
            the relation representation initializer
        :param relation_constrainer:
            the relation representation constrainer
        :param regularizer:
            the regularizer
        :param regularizer_kwargs:
            additional keyword-based parameters passed to the regularizer
        :param kwargs:
            additional keyword-based parameters passed to :meth:`ERModel.__init__`
        """
        self.epsilon = 2.0
        bound = (lm + self.epsilon) / embedding_dim
        phase_weight = phase_weight * bound
        super().__init__(
            interaction=HAKEInteraction(bound=bound, phase_weight=phase_weight, modulus_weight=modulus_weight),
            entity_representations_kwargs=dict(
                shape=embedding_dim*2, 
                initializer=entity_initializer,
                initializer_kwargs=dict(bound=bound),
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=torch.float,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim*3,
                initializer=relation_initializer,
                initializer_kwargs=dict(bound=bound),
                dtype=torch.float,
            ),
            **kwargs,
        )