'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-06-20 11:26:31
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-06 17:05:50
FilePath: /ESETC/code/Custom/OriginRotatE.py
Description: 在pykeen中引入不使用complex张量的RotatE

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from typing import Any, ClassVar, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from class_resolver import HintOrType, OptionalKwargs
from pykeen.models.nbase import ERModel
from pykeen.nn.init import init_phases, xavier_uniform_
from pykeen.nn.modules import FunctionalInteraction
from pykeen.regularizers import Regularizer
from pykeen.typing import Constrainer, Hint, Initializer
from pykeen.utils import (complex_normalize, ensure_complex,
                          estimate_cost_of_sequence, negative_norm)
from torch.nn.functional import normalize


def rotate_initialize(tensor, bound):
    torch.nn.init.uniform_(tensor, -bound, bound)
    return tensor

def rotate_relation_initialize(tensor, bound):
    return rotate_initialize(tensor, bound)/(bound/np.pi)

def rotate_origin_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    """the RotatE interaction function without complex tensor.

    .. note::
        this method expects all tensors to be of complex datatype, i.e., `torch.is_complex(x)` to evaluate to `True`.

    :param h: shape: (`*batch_dims`, dim * 2)
        The head representations.
    :param r: shape: (`*batch_dims`, dim)
        The relation representations.
    :param t: shape: (`*batch_dims`, dim * 2)
        The tail representations.

    :return: shape: batch_dims
        The scores.
    """

    if h.shape[-1] != 2:
        h = h.view(*h.shape[:-1], -1, 2)
    if t.shape[-1] != 2:
        t = t.view(*t.shape[:-1], -1, 2)
    
    re_h, im_h = torch.chunk(h, 2, dim=-1)
    re_t, im_t = torch.chunk(t, 2, dim=-1)

    # if r.shape[-1] != 2:
    #     r = r.view(*r.shape[:-1], -1, 2)
    # re_r, im_r = torch.chunk(r, 2, dim=-1)

    # phase_relation = r / (r.abs().clamp_min(torch.finfo(r.dtype).eps) / np.pi)
    # phase_relation = F.normalize(r) * np.pi
    # phase_relation = r * np.pi
    # r = r/(bound/np.pi)
    re_r = torch.cos(r).unsqueeze(-1)
    im_r = torch.sin(r).unsqueeze(-1)

    if estimate_cost_of_sequence(h.shape, r.shape) < estimate_cost_of_sequence(r.shape, t.shape):
    # 当h和r的计算量小于r和t的计算量时，说明此时我们替换的是尾实体，也就是原Rotate代码中的tail-batch
        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r

        re_score = re_score - re_t
        im_score = im_score - im_t

    else:
        re_score = re_r * re_t + im_r * im_t
        im_score = re_r * im_t - im_r * re_t

        re_score = re_score - re_h
        im_score = im_score - im_h


    score = torch.stack([re_score, im_score], dim = 0)
    score = score.norm(dim = 0).squeeze()

    # 测试时使用1-N scoring方法会出现维度不匹配的问题，这里消除一下
    if len(score.shape) > 3:
        score = score.view(score.shape[0], score.shape[1], -1)

    return -score.sum(dim = -1)


class RotatEOriginInteraction(FunctionalInteraction[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]):
    """A module wrapper for the stateless RotatE interaction function.

    .. seealso:: :func:`pykeen.nn.functional.rotate_interaction`
    """

    func = rotate_origin_interaction

class FloatRotatE(ERModel):
    r"""An implementation of RotatE using Float tensor.
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=32, high=1024, q=16),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        entity_initializer: Hint[Initializer] = rotate_initialize,
        # relation_initializer: Hint[Initializer] = init_phases,
        relation_initializer: Hint[Initializer] = rotate_relation_initialize,
        relation_constrainer: Hint[Constrainer] = normalize,
        regularizer: HintOrType[Regularizer] = None,
        regularizer_kwargs: OptionalKwargs = None,
        lm : float = 9.0,
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
        super().__init__(
            interaction=RotatEOriginInteraction,
            entity_representations_kwargs=dict(
                shape=embedding_dim*2, 
                initializer=entity_initializer,
                initializer_kwargs=dict(bound=bound),
                regularizer=regularizer,
                regularizer_kwargs=regularizer_kwargs,
                dtype=torch.float,
            ),
            relation_representations_kwargs=dict(
                shape=embedding_dim,
                initializer=relation_initializer,
                initializer_kwargs=dict(bound=bound),
                constrainer=relation_constrainer,
                # constrainer_kwargs=dict(bound=bound),
                dtype=torch.float,
            ),
            **kwargs,
        )     