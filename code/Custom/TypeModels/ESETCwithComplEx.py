'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-02-17 11:58:01
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-02-24 13:32:37
FilePath: /ESETC/code/Custom/TypeModels/ESETCwithComplEx.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from class_resolver import OptionalKwargs
from class_resolver.api import HintOrType
from pykeen.constants import (DEFAULT_DROPOUT_HPO_RANGE,
                              DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE)
from pykeen.losses import Loss, SoftplusLoss
from pykeen.nn import ComplExInteraction, DistMultInteraction
from pykeen.nn.init import (init_phases, xavier_normal_, xavier_normal_norm_,
                            xavier_uniform_)
from pykeen.regularizers import LpRegularizer, Regularizer
from pykeen.typing import Constrainer, Hint, Initializer
from torch.nn import functional
from torch.nn.init import normal_

from .ESETC import TypeFramework

__all__ = [
    "TuckER",
]

class ESETCwithComplEx(TypeFramework):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    #: The default loss function class
    loss_default: ClassVar[Type[Loss]] = SoftplusLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = dict(reduction="mean")
    #: The LP settings used by [trouillon2016]_ for ComplEx.
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.01,
        p=3.0, # 使用N3norm
        normalize=True,
    )
    def __init__(
        self,
        *,
        ent_dim: int = 200,
        rel_dim: int = 200,
        type_dim: int = 100,
        entity_initializer: Hint[Initializer] = normal_,
        relation_initializer: Hint[Initializer] = normal_,
        type_initializer: Hint[Initializer] = normal_,
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        bias = False,
        dropout = 0.3,
        data_type = torch.float,
        **kwargs,
    ) -> None:
        regularizer_kwargs = regularizer_kwargs or ESETCwithComplEx.regularizer_default_kwargs
        super().__init__(
            dropout=dropout,
            data_type = data_type,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            interaction=ComplExInteraction,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                dtype=data_type,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                dtype=data_type,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
                dtype=data_type,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            **kwargs,
            )
        

class ESETCwithDistMult(TypeFramework):
    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )
    regularizer_default: ClassVar[Type[Regularizer]] = LpRegularizer
    #: The LP settings used by [yang2014]_ for DistMult
    regularizer_default_kwargs: ClassVar[Mapping[str, Any]] = dict(
        weight=0.1,
        p=3.0, # 使用N3norm
        normalize=True,
    )
    def __init__(
        self,
        *,
        ent_dim: int = 200,
        rel_dim: int = 200,
        type_dim: int = 100,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_normal_norm_,
        type_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        regularizer: HintOrType[Regularizer] = LpRegularizer,
        regularizer_kwargs: Optional[Mapping[str, Any]] = None,
        bias = False,
        dropout = 0.3,
        data_type = torch.float,
        **kwargs,
    ) -> None:
        regularizer_kwargs = regularizer_kwargs or ESETCwithDistMult.regularizer_default_kwargs
        super().__init__(
            dropout=dropout,
            data_type = data_type,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            interaction=DistMultInteraction,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                dtype=data_type,
                constrainer=entity_constrainer,
                # note: DistMult only regularizes the relation embeddings;
                #       entity embeddings are hard constrained insteadd
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                dtype=data_type,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
                dtype=data_type,
                constrainer=entity_constrainer,
            ),
            **kwargs,
            )
