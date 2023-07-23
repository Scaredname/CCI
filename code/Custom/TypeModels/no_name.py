from typing import Any, ClassVar, Mapping, Optional, Type

import torch
from class_resolver import HintOrType, OptionalKwargs
from pykeen.losses import Loss, NSSALoss
from pykeen.nn.init import init_phases, xavier_uniform_, xavier_uniform_norm_
from pykeen.nn.modules import (RotatEInteraction, TransEInteraction,
                               parallel_unsqueeze)
from pykeen.regularizers import Regularizer
from pykeen.typing import Constrainer, Hint, InductiveMode, Initializer
from pykeen.utils import complex_normalize
from torch.nn import functional

from .CatRSETC import CatRSETC
from .ESETC import repeat_if_necessary


class NoNameYet(CatRSETC):
    
    def __init__(self, type_score_weight, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rels_types_h = self.rel_type_h_weights[0]._embeddings.weight
        self.rels_types_t = self.rel_type_t_weights[0]._embeddings.weight
        self.type_score_weight = type_score_weight

    def score_hrt(self, hrt_batch: torch.LongTensor, *, mode: Optional[InductiveMode] = None) -> torch.FloatTensor:
        
        """
        add type relatedness into socre
        """
        
        h_index = hrt_batch[:, 0]
        r_index = hrt_batch[:, 1]
        t_index = hrt_batch[:, 2]


        self.ents_types_weight = self.ents_types_weight.to(self.device)
        self.rels_inj_conf = self.rels_inj_conf.clone().detach().to(self.device)

        h_type_weight = self.ents_types_weight[h_index]
        r_h_type_weight = self.rels_types_h[r_index]
        r_t_type_weight = self.rels_types_t[r_index]
        t_type_weight = self.ents_types_weight[t_index]

        
        injective_confidence = self.rels_inj_conf[r_index]

        r_h_type_score = (h_type_weight * r_h_type_weight).sum(-1)
        r_t_type_score = (t_type_weight * r_t_type_weight).sum(-1)
        type_rel = torch.stack([r_h_type_score, r_t_type_score], dim=-1)
        
        h, r, t = self._get_representations(h=h_index, r=r_index, t=t_index, mode=mode)
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=h_index, r_h=r_index, r_t=r_index, t=t_index, mode=mode)
        h_s_type_emb = torch.cat([head_type_emb, h],dim=-1).to(self.device)
        t_s_type_emb = torch.cat([tail_type_emb, t],dim=-1).to(self.device)
        
        h = h_s_type_emb
        t = t_s_type_emb

        type_score = self.type_score_weight * (r_h_type_score.unsqueeze(dim=-1) + r_t_type_score.unsqueeze(dim=-1))

        return self.interaction.score_hrt(h=h, r=r, t=t) + type_score, injective_confidence, type_rel
    
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


        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=hr_batch[..., 0], r_h=hr_batch[..., 1], r_t=hr_batch[..., 1].repeat(1, self.triples_factory.num_entities), t=tails, mode=mode)
        
        h_type_weight = self.ents_types_weight[hr_batch[..., 0]]
        r_h_type_weight = self.rels_types_h[hr_batch[..., 1]]
        r_t_type_weight = self.rels_types_t[hr_batch[..., 1]] # [batch_num, 1, type_num]
        t_type_weight = self.ents_types_weight[tails] # [1, ent_num, type_num]

        r_h_type_score = (r_h_type_weight * h_type_weight).sum(-1)
        r_t_type_score = (r_t_type_weight * t_type_weight).sum(-1)
        
        t_assig = t_assig.view(t.shape[0], -1)
        head_type_emb = head_type_emb.view(h.shape[0], h.shape[1], -1)

        h_s_type_emb = torch.cat([head_type_emb, h],dim=-1).to(self.device)
        # 确保t和t_type_emb的shape一致
        t = t.unsqueeze(dim=0).repeat(h.shape[0], 1, 1)
        # 会出现测试批度为1的特例，所以调整一下tail_type_emb的shape
        tail_type_emb = tail_type_emb.view(t.shape[0], t.shape[1], -1)

        t_s_type_emb = torch.cat([tail_type_emb, t],dim=-1).to(self.device)

        h = h_s_type_emb
        t = t_s_type_emb

        type_score = self.type_score_weight * (r_h_type_score.unsqueeze(dim=-1) + r_t_type_score.unsqueeze(dim=-1))
        # unsqueeze if necessary
        if tails is None or tails.ndimension() == 1:
            if not len(t.shape) > 2:
                t = parallel_unsqueeze(t, dim=0)
        return repeat_if_necessary(
            # score shape: (batch_size, num_entities)
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1).view(-1, self.num_entities) + type_score.view(-1, self.num_entities), # 会出现测试批度为1的特例，所以调整一下score的shape
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
        
        

        # 发现问题，当不使用ent-type权重时，head_type_emb无法通过广播机制来得到合适的shape。解决方法，令r_h重复num_ent次。
        head_type_emb, tail_type_emb, h_assig, t_assig = self._get_enttype_representations(h=heads, r_h=rt_batch[..., 0].repeat(1, self.triples_factory.num_entities), r_t=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode)

        t_type_weight = self.ents_types_weight[rt_batch[..., 1]]
        r_h_type_weight = self.rels_types_h[rt_batch[..., 0]]
        r_t_type_weight = self.rels_types_t[rt_batch[..., 0]] # [batch_num, 1, type_num]
        h_type_weight = self.ents_types_weight[heads] # [1, ent_num, type_num]

        r_h_type_score = (r_h_type_weight * h_type_weight).sum(-1)
        r_t_type_score = (r_t_type_weight * t_type_weight).sum(-1)

        h_assig = h_assig.view(h.shape[0], -1)
        tail_type_emb = tail_type_emb.view(t.shape[0], t.shape[1], -1)
        # 确保h和h_type_emb的shape一致
        h = h.unsqueeze(dim=0).repeat(t.shape[0], 1, 1)
        # 会出现测试批度为1的特例，所以调整一下head_type_emb的shape
        head_type_emb = head_type_emb.view(h.shape[0], h.shape[1], -1)

        h_s_type_emb = torch.cat([head_type_emb, h],dim=-1).to(self.device)    
        t_s_type_emb = torch.cat([tail_type_emb, t],dim=-1).to(self.device)

        h = h_s_type_emb
        t = t_s_type_emb


        type_score = self.type_score_weight * (r_h_type_score.unsqueeze(dim=-1) + r_t_type_score.unsqueeze(dim=-1))
        # unsqueeze if necessary
        if heads is None or heads.ndimension() == 1:
            h = parallel_unsqueeze(h, dim=0)

        return repeat_if_necessary(
            scores=self.interaction.score(h=h, r=r, t=t, slice_size=slice_size, slice_dim=1).view(-1, self.num_entities) + type_score.view(-1, self.num_entities), # 会出现测试批度为1的特例，所以调整一下score的shape
            representations=self.entity_representations,
            num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1],
        )
    

class NNYwithTransE(NoNameYet):
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
            type_score_weight = 1.0, 
            **kwargs,) -> None:
        super().__init__(
            dropout=dropout,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            data_type=torch.float,
            type_score_weight= type_score_weight,
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
        

class NNYwithRotatE(NoNameYet):
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
            ent_dtype = torch.float,
            rel_dtype = torch.float,
            type_dtype = torch.float,
            dropout = 0.3,
            type_score_weight = 1.0, 
            **kwargs,) -> None:
        super().__init__(
            dropout=dropout,
            bias=bias,
            ent_dim=ent_dim,
            rel_dim=rel_dim,
            type_dim=type_dim,
            data_type=ent_dtype,
            type_score_weight= type_score_weight,
            interaction=RotatEInteraction,
            entity_representations_kwargs=dict(
                shape=ent_dim,
                initializer=entity_initializer,
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=ent_dtype,
            ),
            relation_representations_kwargs=dict(
                shape=rel_dim,
                initializer=relation_initializer,
                constrainer=relation_constrainer,
                dtype=rel_dtype,
            ),
            type_representations_kwargs=dict(
                shape=type_dim,
                initializer=type_initializer,
                dtype=type_dtype,
            ),
            **kwargs)
