'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-28 14:23:33
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-02-22 15:58:01
FilePath: /code/Custom/TypeModels/ESETCwithRotate.py
Description: 

Copyright (c) 2022 by error: git config user.name && git config user.email & please set dead value or install git, All Rights Reserved. 
'''


import logging
from typing import Any, ClassVar, Mapping, Type

import torch
from class_resolver import HintOrType, OptionalKwargs
from pykeen.losses import Loss, NSSALoss
from pykeen.nn.init import init_phases, xavier_uniform_, xavier_uniform_norm_
from pykeen.nn.modules import RotatEInteraction, TransEInteraction
from pykeen.regularizers import Regularizer
from pykeen.typing import Constrainer, Hint, Initializer
from pykeen.utils import complex_normalize
from torch.nn import functional

from .ESETC import TypeFramework

logger = logging.getLogger(__name__)


name_to_index = {name: index for index, name in enumerate("hrt")}
    


class ESETCwithRotate(TypeFramework):
    loss_default: ClassVar[Type[Loss]] = NSSALoss
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
    )
    def __init__(
            self,
            *,
            ent_dim: int = 50,
            rel_dim: int = 50,
            type_dim: int = 20,
            entity_initializer: Hint[Initializer] = xavier_uniform_,
            type_initializer: Hint[Initializer] = xavier_uniform_,
            relation_initializer: Hint[Initializer] = init_phases,
            relation_constrainer: Hint[Constrainer] = complex_normalize,
            regularizer: HintOrType[Regularizer] = None,
            regularizer_kwargs: OptionalKwargs = None,
            bias = False,
            data_type = torch.cfloat,
            dropout = 0.3,
            **kwargs,) -> None:
        super().__init__(
            dropout=dropout,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            data_type=data_type,
            interaction=RotatEInteraction,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=data_type,
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                dtype=data_type,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
                dtype=data_type,
            ),
            **kwargs)


class ESETCwithTransE(TypeFramework):
    loss_default: ClassVar[Type[Loss]] = NSSALoss
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
    )
    def __init__(
            self,
            *,
            ent_dim: int = 50,
            rel_dim: int = 50,
            type_dim: int = 20,
            scoring_fct_norm: int = 1,
            entity_initializer: Hint[Initializer] = xavier_uniform_,
            entity_constrainer: Hint[Constrainer] = functional.normalize,
            relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
            relation_constrainer: Hint[Constrainer] = None,
            type_initializer: Hint[Initializer] = xavier_uniform_,
            regularizer: HintOrType[Regularizer] = None,
            regularizer_kwargs: OptionalKwargs = None,
            bias = False,
            dropout = 0.3,
            **kwargs,) -> None:
        super().__init__(
            dropout=dropout,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            data_type=torch.float,
            interaction=TransEInteraction,
            interaction_kwargs=dict(p=scoring_fct_norm),
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
            ),
            **kwargs)


        