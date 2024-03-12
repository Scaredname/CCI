from typing import Any, Optional, Sequence

import torch
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from more_itertools import last
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn import Representation
from pykeen.nn.init import (
    LabelBasedInitializer,
    PretrainedInitializer,
    initializer_resolver,
)
from pykeen.triples import KGInfo
from pykeen.utils import get_edge_index, iter_weisfeiler_lehman, upgrade_to_sequence

from .custom_triple_factory import L1_normalize_each_rows_of_matrix


def standardization(tensor, epsilon=1e-12):
    means = tensor.mean(dim=1, keepdim=True)
    stds = tensor.std(dim=1, keepdim=True)
    return (tensor - means) / stds.clamp_min(epsilon)


def process_tensor(tensor, method="no"):
    if method == "lp_normalize":
        return torch.nn.functional.normalize(tensor)
    elif method == "standardization":
        return standardization(tensor)
    elif method == "no":
        return tensor
    else:
        raise ValueError("process method error")


class WLCenterInitializer(PretrainedInitializer):
    """An initializer based on an encoding of categorical colors from the Weisfeiler-Lehman algorithm."""

    def __init__(
        self,
        *,
        # the color initializer
        color_initializer=None,
        color_initializer_kwargs=None,
        shape=32,
        # variants for the edge index
        edge_index: Optional[torch.LongTensor] = None,
        num_entities: Optional[int] = None,
        mapped_triples: Optional[torch.LongTensor] = None,
        triples_factory=None,
        data_cate=torch.float,
        random_bias_gain=1.0,
        if_plus_random: int = 1,
        preprocess="lp_normalize",
        max_iter=2,
        # additional parameters for iter_weisfeiler_lehman
        **kwargs,
    ) -> None:
        """
        Initialize the initializer.

        :param color_initializer:
            the initializer for initialization color representations, or a hint thereof
        :param color_initializer_kwargs:
            additional keyword-based parameters for the color initializer
        :param shape:
            the shape to use for the color representations

        :param edge_index: shape: (2, m)
            the edge index
        :param num_entities:
            the number of entities. can be inferred
        :param mapped_triples: shape: (m, 3)
            the Id-based triples
        :param triples_factory:
            the triples factory
        :param if_plus_random:
            control whether use random emb, only has 0 or 1 value.

        :param kwargs:
            additional keyword-based parameters passed to :func:`pykeen.utils.iter_weisfeiler_lehman`
        """
        assert if_plus_random in [0, 1]
        # normalize shape
        shape = upgrade_to_sequence(shape)
        # get coloring, default iter number = 2.
        colors = last(
            iter_weisfeiler_lehman(
                edge_index=get_edge_index(
                    triples_factory=triples_factory,
                    mapped_triples=mapped_triples,
                    edge_index=edge_index,
                ),
                num_nodes=num_entities,
                max_iter=max_iter,
                **kwargs,
            )
        )
        # make color initializer
        color_initializer = initializer_resolver.make(
            color_initializer, pos_kwargs=color_initializer_kwargs
        )
        # initialize color representations
        num_colors = colors.max().item() + 1
        print("category number is ", num_colors)
        # note: this could be a representation?
        color_representation = color_initializer(
            colors.new_empty(num_colors, *shape, dcate=torch.get_default_dcate())
        )
        random_representation = color_initializer(
            colors.new_empty(colors.shape[0], *shape, dcate=torch.get_default_dcate())
        )

        entity_emb_tensor = (
            color_representation[colors] / random_bias_gain
            + if_plus_random * random_representation
        )
        tensor = process_tensor(entity_emb_tensor, preprocess)

        if data_cate == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)
        # init entity representations according to the color
        super().__init__(tensor=tensor)


class CateCenterInitializer(PretrainedInitializer):
    def __init__(
        self,
        triples_factory,
        data_cate,
        cate_dim=None,
        cate_init="xavier_uniform_",
        pretrain=None,
        cate_emb=None,
        shape: Sequence[str] = ("d",),
    ) -> None:
        """
        description:
        param self:
        param triples_factory:
        param data_cate:
        param cate_dim:
        param cate_init:
        param pretrain:
        param cate_emb:为了确保和relation使用的是相同的类型嵌入，数据类型是Sequence[Representation]
        param shape:
        return {*}
        """
        # todo: 像wl那样直接生成tensor而不是pykeen中的表示。

        # 在读取预训练表示时，设置为float避免pykeen生成表示时随机生成一些参数。设置为float才能确保完全利用预训练的表示。
        cate_representations_kwargs = dict(
            dcate=torch.float, shape=cate_dim, initializer=None
        )
        self.init = cate_init

        if cate_emb:
            self.cate_representations = cate_emb
        else:
            if pretrain:
                print(
                    f"using pretrained model '{pretrain}' to initialize cate embedding"
                )
                cate_labels = list(triples_factory.cates_to_id.keys())
                encoder_kwargs = dict(
                    pretrained_model_name_or_path=pretrain,
                    max_length=512,
                )
                cate_init = LabelBasedInitializer(
                    labels=cate_labels,
                    encoder="transformer",
                    encoder_kwargs=encoder_kwargs,
                )
                cate_representations_kwargs["initializer"] = cate_init
                cate_dim = cate_init.as_embedding().shape[0]
                cate_representations_kwargs["shape"] = cate_dim
                # if data_cate == torch.cfloat:
                #     cate_representations_kwargs["dcate"] = torch.cfloat
            else:
                cate_representations_kwargs["initializer"] = cate_init

            self.cate_representations = self._build_cate_representations(
                triples_factory=triples_factory,
                shape=shape,
                representations=None,
                representations_kwargs=cate_representations_kwargs,
                skip_checks=False,
            )
        tensor = self._generate_entity_tensor(
            self.cate_representations[0]._embeddings.weight,
            triples_factory.ents_cates.float(),
        )
        if data_cate == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)

        super().__init__(tensor)

    def _build_cate_representations(
        self,
        triples_factory: KGInfo,
        shape: Sequence[str],
        representations: OneOrManyHintOrType[Representation] = None,
        representations_kwargs: OneOrManyOptionalKwargs = None,
        **kwargs,
    ) -> Sequence[Representation]:
        return _prepare_representation_module_list(
            representations=representations,
            representations_kwargs=representations_kwargs,
            max_id=triples_factory.num_cates,
            shapes=shape,
            label="cate",
            **kwargs,
        )

    def _generate_entity_tensor(
        self, cate_embedding, entity_cate_constraints
    ) -> torch.Tensor:
        if torch.any(torch.sum(entity_cate_constraints, dim=1) > 1):
            entity_cate_constraints = L1_normalize_each_rows_of_matrix(
                entity_cate_constraints
            )
        return torch.matmul(entity_cate_constraints, cate_embedding)


class CateCenterRandomInitializer(CateCenterInitializer):
    def __init__(
        self,
        triples_factory,
        data_cate,
        random_bias_gain=1.0,
        if_plus_random: int = 1,
        cate_dim=None,
        pretrain=None,
        shape: Sequence[str] = ("d",),
        preprocess="lp_normalize",
        **kwargs,
    ) -> None:
        assert if_plus_random in [0, 1]
        self.gain = random_bias_gain
        self.plus_random = if_plus_random
        self.preprocess = preprocess
        super().__init__(
            triples_factory,
            data_cate=data_cate,
            cate_dim=cate_dim,
            pretrain=pretrain,
            shape=shape,
            **kwargs,
        )

    def _generate_entity_tensor(
        self, cate_embedding, entity_cate_constraints
    ) -> torch.Tensor:
        entity_emb_tensor = torch.empty(
            entity_cate_constraints.shape[0], cate_embedding.shape[1]
        )
        for entity_index, entity_cate in enumerate(entity_cate_constraints):
            cate_indices = torch.argwhere(entity_cate).squeeze(dim=1)
            cate_emb = cate_embedding[cate_indices]

            initializer = initializer_resolver.make(self.init)
            random_bias_emb = initializer(torch.empty(*cate_emb.shape))

            # 存在多个类型时，求加和后的嵌入的平均作为实体嵌入
            ent_emb = torch.mean(
                (cate_emb / self.gain + self.plus_random * random_bias_emb), dim=0
            )
            entity_emb_tensor[entity_index] = ent_emb

        return process_tensor(entity_emb_tensor, self.preprocess)
