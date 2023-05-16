# The name of this new framework is "Relation Specific Entity and entity Type Combination" (RSETC).
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
from pykeen.nn.init import init_phases, xavier_uniform_, xavier_uniform_norm_
from pykeen.nn.modules import (Interaction, RotatEInteraction,
                               TransEInteraction, interaction_resolver,
                               parallel_unsqueeze)
from pykeen.nn.representation import Representation
from pykeen.regularizers import Regularizer
from pykeen.triples import KGInfo
from pykeen.typing import (Constrainer, HeadRepresentation, Hint,
                           InductiveMode, Initializer, RelationRepresentation,
                           TailRepresentation)
from pykeen.utils import complex_normalize
from torch import nn
from torch.nn import functional

from ..CustomTripleFactory import TriplesTypesFactory
from .ESETC import TypeFramework, repeat_if_necessary


class RSETC(TypeFramework):
    """
    
    """
    def __init__(self,freeze_matrix = False, add_ent_type = True, **kwargs) -> None:
        super().__init__(**kwargs)

        # ents_types requires_grad=False, rels_types requires_grad=True
        # self.ents_types = torch.nn.parameter.Parameter(torch.as_tensor(self.triples_factory.ents_types, dtype=self.data_type, device=self.device), requires_grad=False)
        # self.ents_types = torch.as_tensor(self.triples_factory.ents_types, dtype=self.data_type, device=self.device)
        self.ents_types = None
        self.add_ent_type = add_ent_type
        self.rels_types = torch.nn.parameter.Parameter(torch.as_tensor(self.triples_factory.rels_types, dtype=self.data_type, device=self.device), requires_grad= not freeze_matrix)

    
    def _get_enttype_representations(
        self,
        h: Optional[torch.LongTensor],
        r_h: Optional[torch.LongTensor],
        r_t: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        *,
        mode: Optional[InductiveMode],
    ) -> Tuple:
        """Get representations for head ent type emb and tail ent type emb."""
        
        assignments = self.triples_factory.assignments.to(self.device)
        # type_emb = self.type_representations[0]._embeddings.weight.to(self.device)
        type_emb = self.type_representations[0](indices = torch.arange(self.triples_factory.ents_types.shape[1]).long().to(self.device)) #取出所有的type embedding

        #通过邻接矩阵与类型嵌入矩阵的矩阵乘法可以快速每个实体对应的类型嵌入，如果是多个类型则是多个类型嵌入的加权和，权重为邻接矩阵中的值。如果值都为1则相当于sum操作，为平均值则是mean操作。
        rels_types_h = self.rels_types[0]
        rels_types_t = self.rels_types[1]
        ents_types = self.triples_factory.ents_types.to(self.device)

        if self.add_ent_type:
            head_type_emb_tensor = torch.matmul(ents_types[h]+rels_types_h[r_h], type_emb)
            tail_type_emb_tensor = torch.matmul(ents_types[t]+rels_types_t[r_t], type_emb)
        else:
            head_type_emb_tensor = torch.matmul(rels_types_h[r_h], type_emb)
            tail_type_emb_tensor = torch.matmul(rels_types_t[r_t], type_emb)

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
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=h_index, r_h=r_index, r_t=r_index, t=t_index, mode=mode)
        h_s_type_emb = self.projection(torch.cat([head_type_emb, h],dim=-1)).to(self.device)
        t_s_type_emb = self.projection(torch.cat([tail_type_emb, t],dim=-1)).to(self.device)
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

        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=hr_batch[..., 0], r_h=hr_batch[..., 1], r_t=hr_batch[..., 1], t=tails, mode=mode)
        
        
        t_assig = t_assig.view(t.shape[0], -1)
        head_type_emb = head_type_emb.view(h.shape[0], h.shape[1], -1)

        h_s_type_emb = self.projection(torch.cat([head_type_emb, h],dim=-1)).to(self.device)
        # 确保t和t_type_emb的shape一致
        t = t.unsqueeze(dim=0).repeat(h.shape[0], 1, 1)
        # 会出现测试批度为1的特例，所以调整一下tail_type_emb的shape
        tail_type_emb = tail_type_emb.view(t.shape[0], t.shape[1], -1)
        
        t_s_type_emb = self.projection(torch.cat([tail_type_emb, t],dim=-1)).to(self.device)

        if self.data_type == torch.cfloat:
            h_s_type_emb = self._complex_dropout(h_s_type_emb, p=self.dropout)
            t_s_type_emb = self._complex_dropout(t_s_type_emb, p=self.dropout)

        h_emb_list = [h_s_type_emb, h]
        # 当采用1N打分时，广播机制存在所以也可以实现嵌入的替换。
        h = self._replace_emb(h_emb_list, h_assig)    
        t_emb_list = [t_s_type_emb, t]
        t = self._replace_emb(t_emb_list, t_assig)
        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            if not len(t.shape) > 2:
                t = parallel_unsqueeze(t, dim=0)

        return repeat_if_necessary(
            # score shape: (batch_size, num_entities)
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1).view(-1, self.num_entities), # 会出现测试批度为1的特例，所以调整一下score的shape
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
        
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=heads, r_h=rt_batch[..., 0], r_t=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode)

        h_assig = h_assig.view(h.shape[0], -1)
        tail_type_emb = tail_type_emb.view(t.shape[0], t.shape[1], -1)
        
        # 确保h和h_type_emb的shape一致
        h = h.unsqueeze(dim=0).repeat(t.shape[0], 1, 1)
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
        if heads is None or heads.ndimension() == 1:
            h = parallel_unsqueeze(h, dim=0)


        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1).view(-1, self.num_entities), # 会出现测试批度为1的特例，所以调整一下score的shape
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )




class RSETCwithTransE(RSETC):
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
