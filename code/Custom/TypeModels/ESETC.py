'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-28 16:19:48
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-04 14:30:01
FilePath: /ESETC/code/Custom/TypeModels/ESETC.py
Description: "Entity Specific Entity and entity Type Combination" (ESETC)

Copyright (c) 2022 by error: git config user.name && git config user.email & please set dead value or install git, All Rights Reserved. 
'''

import logging
from typing import (Any, ClassVar, Mapping, Optional, Sequence, Tuple, Type,
                    cast)

import numpy as np
import torch
from class_resolver import HintOrType, OptionalKwargs
from class_resolver.contrib.torch import activation_resolver
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from pykeen.losses import Loss, NSSALoss
from pykeen.models import ERModel
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn import Representation
from pykeen.nn.combination import Combination
from pykeen.nn.init import LabelBasedInitializer, init_phases, xavier_uniform_
from pykeen.nn.modules import (Interaction, RotatEInteraction,
                               interaction_resolver, parallel_unsqueeze)
from pykeen.nn.representation import Representation
from pykeen.regularizers import Regularizer
from pykeen.triples import KGInfo
from pykeen.typing import (Constrainer, HeadRepresentation, Hint,
                           InductiveMode, Initializer, RelationRepresentation,
                           TailRepresentation)
from pykeen.utils import complex_normalize
from torch import nn

from ..CustomTripleFactory import TriplesTypesFactory

logger = logging.getLogger(__name__)

name_to_index = {name: index for index, name in enumerate("hrt")}

def repeat_if_necessary(
    scores: torch.FloatTensor,
    representations: Sequence[Representation],
    num: Optional[int],
) -> torch.FloatTensor:
    """
    Repeat score tensor if necessary.

    If a model does not have entity/relation representations, the scores for
    `score_{h,t}` / `score_r` are always the same. For efficiency, they are thus
    only computed once, but to meet the API, they have to be brought into the correct shape afterwards.

    :param scores: shape: (batch_size, ?)
        the score tensor
    :param representations:
        the representations. If empty (i.e. no representations for this 1:n scoring), repetition needs to be applied
    :param num:
        the number of times to repeat, if necessary.

    :return:
        the score tensor, which has been repeated, if necessary
    """
    if representations:
        return scores
    return scores.repeat(1, num)

class TypeFramework(ERModel):
    """Base class for models with entity types that uses combinations."""
    def __init__(self,
        triples_factory: TriplesTypesFactory,
        interaction: HintOrType[Interaction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]],
        entity_representations: OneOrManyHintOrType[Representation] = None,
        entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        relation_representations: OneOrManyHintOrType[Representation] = None,
        relation_representations_kwargs: OneOrManyOptionalKwargs = None,
        relation_constrainer: Hint[Constrainer] = complex_normalize,
        type_representations: OneOrManyHintOrType[Representation] = None,
        type_representations_kwargs: OneOrManyOptionalKwargs = None,
        combination: HintOrType[Combination] = None,
        combination_kwargs: OptionalKwargs = None,
        skip_checks: bool = False,
        ent_dim: int = 50,
        rel_dim: int = 50,
        type_dim: int = 20,
        data_type = float,
        bias = True,
        dropout = 0.0,
        shape: Sequence[str] = ('d',),
        activation: HintOrType[nn.Module] = nn.Identity,
        activation_kwargs: OptionalKwargs = None,
        usepretrained = None,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        relation_initializer: Hint[Initializer] = xavier_uniform_,
        type_initializer: Hint[Initializer] = xavier_uniform_,
        freeze_matrix = False,
        freeze_type_emb = False,
        activation_weight = False,
        weight_mask = False,
        type_weight_temperature = 1.0,
        **kwargs,) -> None:

        self.triples_factory = triples_factory
        self.shape = shape
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.type_dim = type_dim

        self.entity_representations_kwargs = entity_representations_kwargs
        self.data_type = data_type
        self.activation_weight = activation_weight
        self.weight_mask = weight_mask
        self.type_weight_temperature = type_weight_temperature

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations_kwargs=self.entity_representations_kwargs,
            relation_representations_kwargs=relation_representations_kwargs,
            **kwargs,
        )

        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (type_dim+ent_dim, ent_dim)), dtype=data_type, device='cuda', requires_grad=True))

        if data_type == torch.cfloat:
            self.dropout = dropout
            dropout = 0.0

        # 冻结与可训练是相反的
        type_representations_kwargs['trainable'] = not freeze_type_emb
        # Using pre-trained embeddings for type representations
        if usepretrained:
            types = list(triples_factory.types_to_id.keys())
            encoder_kwargs=dict(
					pretrained_model_name_or_path = usepretrained,
                    max_length = 512,
                    )
            type_init = LabelBasedInitializer(labels=types, encoder='transformer', encoder_kwargs=encoder_kwargs)
            type_representations_kwargs['initializer'] = type_init
            type_representations_kwargs['shape'] = type_init.as_embedding().shape[0]
            self.type_dim = type_init.as_embedding().shape[0]

        self.projection = torch.nn.Sequential(
                torch.nn.Linear(self.type_dim+ent_dim, ent_dim, bias=bias, dtype=data_type),
                torch.nn.Dropout(dropout),
                activation_resolver.make(activation, activation_kwargs),)
        
        self.type_representations = self._build_type_representations(
            triples_factory=triples_factory,
            shape=shape,
            representations=type_representations,
            representations_kwargs=type_representations_kwargs,
            skip_checks=skip_checks,
        )

        self.ents_types = torch.nn.parameter.Parameter(torch.as_tensor(self.triples_factory.ents_types, dtype=self.data_type, device=self.device), requires_grad= not freeze_matrix) #令获得实体对应的实体类型嵌入时的权重为可训练参数
        self.activation_function = nn.Softmax(dim=-1)
        self.ents_types_mask = self.triples_factory.ents_types_mask
        self.rels_types_mask = self.triples_factory.rels_types_mask
    def _build_type_representations(self, triples_factory: KGInfo, shape: Sequence[str], representations: OneOrManyHintOrType[Representation] = None, representations_kwargs: OneOrManyOptionalKwargs = None, **kwargs) -> Sequence[Representation]:
        return _prepare_representation_module_list(
            representations=representations,
            representations_kwargs=representations_kwargs,
            max_id=triples_factory.num_types,
            shapes=shape,
            label= 'type',
            **kwargs,
    )
    @property
    def num_parameter_bytes(self) -> int:
        """Different from origin function. Calculate the number of bytes used for gradient enabled parameters of the model."""
        return sum(param.numel() * param.element_size() for param in self.parameters() if param.requires_grad)

    def _replace_emb(self, emb_list, assignments):
        """
        emb_list: [ent_s_type_emb, ent_emb]
        """
        if len(emb_list[0].shape) != 2:
            shape = emb_list[0].shape
            new_emb = torch.squeeze(emb_list[0]) + torch.squeeze(emb_list[1])*assignments.view(-1, 1)
            new_emb = new_emb.view(shape)
        else:
            new_emb = emb_list[0] + emb_list[1]*assignments.view(-1, 1)
        return new_emb
    
    def _complex_dropout(self, input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
        mask = torch.ones(*input.shape, dtype = torch.float32).to(self.device)
        mask = nn.functional.dropout(mask, p, training)*1/(1-p)
        mask.type(input.dtype)
        return mask*input
    
    def _get_representations(
        self,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails."""
        head_representations = tail_representations = self._get_entity_representations_from_inductive_mode(mode=mode)
        head_representations = [head_representations[i] for i in self.interaction.head_indices()]
        tail_representations = [tail_representations[i] for i in self.interaction.tail_indices()]
        hr, rr, tr = [
            [representation(indices=indices) for representation in representations]
            for indices, representations in (
                (h, head_representations),
                (r, self.relation_representations),
                (t, tail_representations),
            )
        ]
        # normalization
        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )

    def _get_enttype_representations(
        self,
        h: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple:
        """Get representations for head ent type emb and tail ent type emb."""
        
        assignments = self.triples_factory.assignments.to(self.device)
        self.ents_types_mask = self.ents_types_mask.to(self.device)
        ents_types = self.ents_types.to(self.device)
        # type_emb = self.type_representations[0]._embeddings.weight.to(self.device)
        type_emb = self.type_representations[0](indices = torch.arange(self.ents_types.shape[1]).long().to(self.device)) #取出所有的type embedding

        #通过邻接矩阵与类型嵌入矩阵的矩阵乘法可以快速每个实体对应的类型嵌入，如果是多个类型则是多个类型嵌入的加权和，权重为邻接矩阵中的值。如果值都为1则相当于sum操作，为平均值则是mean操作。
        if self.weight_mask:
            ents_types = self.ents_types*self.ents_types_mask
        if self.activation_weight:
            head_type_emb_tensor = torch.matmul(self.activation_function(self.type_weight_temperature * ents_types[h]), type_emb)
            tail_type_emb_tensor = torch.matmul(self.activation_function(self.type_weight_temperature * ents_types[t]), type_emb)
        else:
            head_type_emb_tensor = torch.matmul(ents_types[h], type_emb)
            tail_type_emb_tensor = torch.matmul(ents_types[t], type_emb)
        h_assignments = assignments[h]
        t_assignments = assignments[t]

        return torch.squeeze(head_type_emb_tensor), torch.squeeze(tail_type_emb_tensor), torch.squeeze(h_assignments), torch.squeeze(t_assignments)

    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        """Forward pass.

        This method takes head, relation and tail of each triple and calculates the corresponding score.

        :param hrt_batch: shape: (batch_size, 3), dtype: long
            The indices of (head, relation, tail) triples.
        :param mode:
            The pass mode, which is None in the transductive setting and one of "training",
            "validation", or "testing" in the inductive setting.

        :return: shape: (batch_size, 1), dtype: float
            The score for each triple.
        """
        # Note: slicing cannot be used here: the indices for score_hrt only have a batch
        # dimension, and slicing along this dimension is already considered by sub-batching.
        # Note: we do not delegate to the general method for performance reasons
        # Note: repetition is not necessary here
        h_index = hrt_batch[:, 0]
        r_index = hrt_batch[:, 1]
        t_index = hrt_batch[:, 2]
        
        
        h, r, t = self._get_representations(h=h_index, r=r_index, t=t_index, mode=mode)
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=h_index, t=t_index, mode=mode)
        h_s_type_emb = self.projection(torch.cat([head_type_emb, h],dim=-1)).to(self.device)
        t_s_type_emb = self.projection(torch.cat([tail_type_emb, t],dim=-1)).to(self.device)
        # h_s_type_emb = torch.matmul(torch.cat([head_type_emb, h],dim=-1), self.W).to(self.device)
        # t_s_type_emb = torch.matmul(torch.cat([tail_type_emb, t],dim=-1), self.W).to(self.device)
        if self.data_type == torch.cfloat:
            h_s_type_emb = self._complex_dropout(h_s_type_emb, p=self.dropout)
            t_s_type_emb = self._complex_dropout(t_s_type_emb, p=self.dropout)
        
        h_emb_list = [h_s_type_emb, h]
        h = self._replace_emb(h_emb_list, h_assig)

        # h = h_s_type_emb
    
        t_emb_list = [t_s_type_emb, t]
        t = self._replace_emb(t_emb_list, t_assig)
        # t = t_s_type_emb

        return self.interaction.score_hrt(h=h, r=r, t=t)
    
    def score_t(
        self,
        hr_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        tails: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        hr_batch = hr_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1], t=tails, mode=mode)
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=hr_batch[..., 0], t=tails, mode=mode)
        
        tail_type_emb = tail_type_emb.view(t.shape[0], -1)
        t_assig = t_assig.view(t.shape[0], -1)
        head_type_emb = head_type_emb.view(h.shape[0], h.shape[1], -1)

        h_s_type_emb = self.projection(torch.cat([head_type_emb, h],dim=-1)).to(self.device)
        t_s_type_emb = self.projection(torch.cat([tail_type_emb, t],dim=-1)).to(self.device)

        if self.data_type == torch.cfloat:
            h_s_type_emb = self._complex_dropout(h_s_type_emb, p=self.dropout)
            t_s_type_emb = self._complex_dropout(t_s_type_emb, p=self.dropout)

        h_emb_list = [h_s_type_emb, h]
        h = self._replace_emb(h_emb_list, h_assig)    
        t_emb_list = [t_s_type_emb, t]
        t = self._replace_emb(t_emb_list, t_assig)
        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            t = parallel_unsqueeze(t, dim=0)

        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1],
        )
    
    def score_h(
        self,
        rt_batch: torch.LongTensor,
        *,
        slice_size: Optional[int] = None,
        mode: Optional[InductiveMode] = None,
        heads: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        self._check_slicing(slice_size=slice_size)
        # add broadcast dimension
        rt_batch = rt_batch.unsqueeze(dim=1)
        h, r, t = self._get_representations(h=heads, r=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode)
        
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=heads, t=rt_batch[..., 1], mode=mode)

        head_type_emb = head_type_emb.view(h.shape[0], -1)
        h_assig = h_assig.view(h.shape[0], -1)
        tail_type_emb = tail_type_emb.view(t.shape[0], t.shape[1], -1)
        
        
        h_s_type_emb = self.projection(torch.cat([head_type_emb, h],dim=-1)).to(self.device)    
        t_s_type_emb = self.projection(torch.cat([tail_type_emb, t],dim=-1)).to(self.device)
        if self.data_type == torch.cfloat:
            h_s_type_emb = self._complex_dropout(h_s_type_emb, p=self.dropout)
            t_s_type_emb = self._complex_dropout(t_s_type_emb, p=self.dropout)

        h_emb_list = [h_s_type_emb, h]
        h = self._replace_emb(h_emb_list, h_assig)    
        t_emb_list = [t_s_type_emb, t]
        t = self._replace_emb(t_emb_list, t_assig)
        # unsqueeze if necessary
        if heads is None or heads.ndimension() == 1:
            h = parallel_unsqueeze(h, dim=0)


        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1),
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )

    


        
