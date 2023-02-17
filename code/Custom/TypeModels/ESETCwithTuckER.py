'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-02-17 11:58:01
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-02-17 12:18:17
FilePath: /undefined/home/ni/code/ESETC/code/Custom/TypeModels/ESETCwithTuckER.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from class_resolver import OptionalKwargs
from pykeen.constants import (DEFAULT_DROPOUT_HPO_RANGE,
                              DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE)
from pykeen.losses import BCEAfterSigmoidLoss, Loss
from pykeen.models.nbase import ERModel
from pykeen.nn import TuckerInteraction
from pykeen.nn.init import xavier_normal_
from pykeen.typing import Hint, Initializer

from .ESETC import TypeFramework

__all__ = [
    "TuckER",
]

class ESETCwithTuckER(TypeFramework):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        ent_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        rel_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        dropout_0=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_1=DEFAULT_DROPOUT_HPO_RANGE,
        dropout_2=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = BCEAfterSigmoidLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        *,
        ent_dim: int = 200,
        rel_dim: int = 200,
        type_dim: int = 100,
        dropout_0: float = 0.3,
        dropout_1: float = 0.4,
        dropout_2: float = 0.5,
        apply_batch_normalization: bool = True,
        entity_initializer: Hint[Initializer] = xavier_normal_,
        relation_initializer: Hint[Initializer] = xavier_normal_,
        type_initializer: Hint[Initializer] = xavier_normal_,
        core_tensor_initializer: Hint[Initializer] = None,
        core_tensor_initializer_kwargs: OptionalKwargs = None,
        bias = False,
        dropout = 0.3,
        data_type = torch.float,
        **kwargs,
    ) -> None:

        super().__init__(
            dropout=dropout,
            data_type = data_type,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            interaction=TuckerInteraction,
            interaction_kwargs=dict(
                embedding_dim=ent_dim,
                relation_dim=rel_dim,
                head_dropout=dropout_0,  # TODO: rename
                relation_dropout=dropout_1,
                head_relation_dropout=dropout_2,
                apply_batch_normalization=apply_batch_normalization,
                core_initializer=core_tensor_initializer,
                core_initializer_kwargs=core_tensor_initializer_kwargs,
            ),
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                dtype=torch.float,
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                dtype=torch.float,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
                dtype=torch.float,
            ),
            **kwargs,
            )
