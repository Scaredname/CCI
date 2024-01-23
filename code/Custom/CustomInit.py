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
    xavier_uniform_,
)
from pykeen.triples import KGInfo
from pykeen.utils import get_edge_index, iter_weisfeiler_lehman, upgrade_to_sequence

# class PretrainedInitializer:
#     """
#     Initialize tensor with pretrained weights.

#     Example usage:

#     .. code-block::

#         import torch
#         from pykeen.pipeline import pipeline
#         from pykeen.nn.init import PretrainedInitializer

#         # this is usually loaded from somewhere else
#         # the shape must match, as well as the entity-to-id mapping
#         pretrained_embedding_tensor = torch.rand(14, 128)

#         result = pipeline(
#             dataset="nations",
#             model="transe",
#             model_kwargs=dict(
#                 embedding_dim=pretrained_embedding_tensor.shape[-1],
#                 entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor),
#             ),
#         )
#     """

#     def __init__(self, tensor: torch.FloatTensor) -> None:
#         """
#         Initialize the initializer.

#         :param tensor:
#             the tensor of pretrained embeddings.
#         """
#         self.tensor = tensor

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         """Initialize the tensor with the given tensor."""
#         if x.shape != self.tensor.shape:
#             raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
#         return self.tensor.to(device=x.device, dtype=x.dtype)


#     def as_embedding(self, **kwargs: Any):
#         """Get a static embedding from this pre-trained initializer.

#         :param kwargs: Keyword arguments to pass to :class:`pykeen.nn.representation.Embedding`
#         :returns: An embedding
#         :rtype: pykeen.nn.representation.Embedding
#         """
#         from pykeen.nn.representation import Embedding

#         max_id, *shape = self.tensor.shape
#         return Embedding(max_id=max_id, shape=shape, initializer=self, trainable=True, **kwargs)


def cal_propor(data):
    """Calculate the proportion of each type."""
    for i in range(data.shape[0]):
        if torch.sum(data[i], dtype=torch.float32) > 0:
            data[i] = data[i] / torch.sum(data[i], dtype=torch.float32)
    return data


def standardization(tensor):
    means = tensor.mean(dim=1, keepdim=True)
    stds = tensor.std(dim=1, keepdim=True)
    return (tensor - means) / stds


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
        data_type=torch.float,
        random_bias_gain=1.0,
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

        :param kwargs:
            additional keyword-based parameters passed to :func:`pykeen.utils.iter_weisfeiler_lehman`
        """
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
                **kwargs,
            )
        )
        # make color initializer
        color_initializer = initializer_resolver.make(
            color_initializer, pos_kwargs=color_initializer_kwargs
        )
        # initialize color representations
        num_colors = colors.max().item() + 1
        # note: this could be a representation?
        color_representation = color_initializer(
            colors.new_empty(num_colors, *shape, dtype=torch.get_default_dtype())
        )
        random_representation = color_initializer(
            colors.new_empty(colors.shape[0], *shape, dtype=torch.get_default_dtype())
        )
        tensor = torch.nn.functional.normalize(
            color_representation[colors] + random_representation
        )
        if data_type == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)
        # init entity representations according to the color
        super().__init__(tensor=tensor)


class TypeCenterInitializer(PretrainedInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        type_dim=None,
        type_init="xavier_uniform_",
        pretrain=None,
        type_emb=None,
        shape: Sequence[str] = ("d",),
    ) -> None:
        """
        description:
        param self:
        param triples_factory:
        param data_type:
        param type_dim:
        param type_init:
        param pretrain:
        param type_emb:为了确保和relation使用的是相同的类型嵌入，数据类型是Sequence[Representation]
        param shape:
        return {*}
        """
        # todo: 像wl那样直接生成tensor而不是pykeen中的表示。

        # 在读取预训练表示时，设置为float避免pykeen生成表示时随机生成一些参数。设置为float才能确保完全利用预训练的表示。
        type_representations_kwargs = dict(
            dtype=torch.float, shape=type_dim, initializer=None
        )
        self.init = type_init

        if type_emb:
            self.type_representations = type_emb
        else:
            if pretrain:
                print(
                    f"using pretrained model '{pretrain}' to initialize type embedding"
                )
                type_labels = list(triples_factory.types_to_id.keys())
                encoder_kwargs = dict(
                    pretrained_model_name_or_path=pretrain,
                    max_length=512,
                )
                type_init = LabelBasedInitializer(
                    labels=type_labels,
                    encoder="transformer",
                    encoder_kwargs=encoder_kwargs,
                )
                type_representations_kwargs["initializer"] = type_init
                type_dim = type_init.as_embedding().shape[0]
                type_representations_kwargs["shape"] = type_dim
                # if data_type == torch.cfloat:
                #     type_representations_kwargs["dtype"] = torch.cfloat
            else:
                type_representations_kwargs["initializer"] = type_init

            self.type_representations = self._build_type_representations(
                triples_factory=triples_factory,
                shape=shape,
                representations=None,
                representations_kwargs=type_representations_kwargs,
                skip_checks=False,
            )
        tensor = self._generate_entity_tensor(
            self.type_representations[0]._embeddings.weight,
            triples_factory.ents_types.float(),
        )
        if data_type == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)

        super().__init__(tensor)

    def _build_type_representations(
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
            max_id=triples_factory.num_types,
            shapes=shape,
            label="type",
            **kwargs,
        )

    def _generate_entity_tensor(
        self, type_embedding, entity_type_constraints
    ) -> torch.Tensor:
        if torch.any(torch.sum(entity_type_constraints, dim=1) > 1):
            entity_type_constraints = cal_propor(entity_type_constraints)
        return torch.matmul(entity_type_constraints, type_embedding)


class TypeCenterRandomInitializer(TypeCenterInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        random_bias_gain=1.0,
        type_dim=None,
        pretrain=None,
        shape: Sequence[str] = ("d",),
        preprocess="lp_normalize",
        **kwargs,
    ) -> None:
        self.gain = random_bias_gain
        self.preprocess = preprocess
        super().__init__(
            triples_factory,
            data_type=data_type,
            type_dim=type_dim,
            pretrain=pretrain,
            shape=shape,
            **kwargs,
        )

    def _generate_entity_tensor(
        self, type_embedding, entity_type_constraints
    ) -> torch.Tensor:
        entity_emb_tensor = torch.empty(
            entity_type_constraints.shape[0], type_embedding.shape[1]
        )
        for entity_index, entity_type in enumerate(entity_type_constraints):
            type_indices = torch.argwhere(entity_type).squeeze(dim=1)
            type_emb = type_embedding[type_indices]

            initializer = initializer_resolver.make(self.init)
            random_bias_emb = initializer(torch.empty(*type_emb.shape))

            # 存在多个类型时，求加和后的嵌入的平均作为实体嵌入
            ent_emb = torch.mean((type_emb + self.gain * random_bias_emb), dim=0)
            entity_emb_tensor[entity_index] = ent_emb

        if self.preprocess == "lp_normalize":
            return torch.nn.functional.normalize(entity_emb_tensor)
        elif self.preprocess == "standardization":
            # return standardization(entity_emb_tensor)
            return entity_emb_tensor
        else:
            raise ValueError("process method error")


class TypeCenterFrequencyRandomInitializer(TypeCenterInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        type_dim=None,
        pretrain=None,
        shape: Sequence[str] = ("d",),
        **kwargs,
    ) -> None:
        self.entities_gain = self.entity_specific_gains(triples_factory)
        super().__init__(
            triples_factory,
            data_type=data_type,
            type_dim=type_dim,
            pretrain=pretrain,
            shape=shape,
            **kwargs,
        )

    def entity_specific_gains(self, triples_factory):
        entities_gain = torch.zeros(triples_factory.num_entities)
        for triple in triples_factory.mapped_triples:
            entities_gain[triple[0]] += 1
            entities_gain[triple[2]] += 1

        return entities_gain / torch.mean(entities_gain)

    def _generate_entity_tensor(
        self, type_embedding, entity_type_constraints
    ) -> torch.Tensor:
        entity_emb_tensor = torch.empty(
            entity_type_constraints.shape[0], type_embedding.shape[1]
        )
        for entity_index, entity_type in enumerate(entity_type_constraints):
            type_indices = torch.argwhere(entity_type).squeeze(dim=1)
            type_emb = type_embedding[type_indices]

            initializer = initializer_resolver.make(self.init)

            # 实体在训练集中的频率作为gain
            random_bias_emb = self.entities_gain[entity_index] * initializer(
                torch.empty(*type_emb.shape)
            )

            # 存在多个类型时，求加和后的嵌入的平均作为实体嵌入
            ent_emb = torch.mean((type_emb + random_bias_emb), dim=0)
            entity_emb_tensor[entity_index] = ent_emb

        return torch.nn.functional.normalize(entity_emb_tensor)


class TypeCenterProductRandomInitializer(TypeCenterRandomInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        random_bias_gain=1,
        type_dim=None,
        pretrain=None,
        preprocess="lp_normalize",
        shape: Sequence[str] = ("d",),
        **kwargs,
    ) -> None:
        super().__init__(
            triples_factory,
            data_type,
            random_bias_gain,
            type_dim,
            pretrain,
            shape,
            preprocess,
            **kwargs,
        )

    def _generate_entity_tensor(
        self, type_embedding, entity_type_constraints
    ) -> torch.Tensor:
        entity_emb_tensor = torch.empty(
            entity_type_constraints.shape[0], type_embedding.shape[1]
        )
        for entity_index, entity_type in enumerate(entity_type_constraints):
            type_indices = torch.argwhere(entity_type).squeeze(dim=1)
            type_emb = type_embedding[type_indices]

            initializer = initializer_resolver.make(self.init)
            random_bias_emb = initializer(torch.empty(*type_emb.shape))

            # 存在多个类型时，求加和后的嵌入的平均作为实体嵌入
            ent_emb = torch.mean((type_emb * self.gain * random_bias_emb), dim=0)
            entity_emb_tensor[entity_index] = ent_emb

        if self.preprocess == "lp_normalize":
            return torch.nn.functional.normalize(entity_emb_tensor)
        elif self.preprocess == "standardization":
            return standardization(entity_emb_tensor)
        else:
            raise ValueError("process method error")


class TypeCenterRelationInitializer(PretrainedInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        type_emb,
        type_dim=None,
        type_init="xavier_uniform_",
        pretrain=None,
        shape: Sequence[str] = ("d",),
    ) -> None:
        """
        description: TransE 思想下的relation 初始化
        param self:
        param triples_factory:
        param data_type:
        param type_dim:
        param type_init:
        param pretrain:
        param type_emb:确保和entity使用的是相同的类型嵌入，数据类型是Sequence[Representation]
        param shape:
        return {*}
        """
        if type_emb:
            self.type_representations = type_emb
        else:
            type_representations_kwargs = dict(
                dtype=torch.float, shape=type_dim, initializer=None
            )
            self.init = type_init
            if pretrain:
                print(
                    f"using pretrained model '{pretrain}' to initialize type embedding"
                )
                type_labels = list(triples_factory.types_to_id.keys())
                encoder_kwargs = dict(
                    pretrained_model_name_or_path=pretrain,
                    max_length=512,
                )
                type_init = LabelBasedInitializer(
                    labels=type_labels,
                    encoder="transformer",
                    encoder_kwargs=encoder_kwargs,
                )
                type_representations_kwargs["initializer"] = type_init
                type_dim = type_init.as_embedding().shape[0]
                type_representations_kwargs["shape"] = type_dim
                # if data_type == torch.cfloat:
                #     type_representations_kwargs["dtype"] = torch.cfloat
            else:
                type_representations_kwargs["initializer"] = type_init

            self.type_representations = self._build_type_representations(
                triples_factory=triples_factory,
                shape=shape,
                representations=None,
                representations_kwargs=type_representations_kwargs,
                skip_checks=False,
            )
        tensor = self._generate_relation_tensor(
            self.type_representations[0]._embeddings.weight,
            triples_factory.rels_types.float(),
        )
        if data_type == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)

        super().__init__(tensor)

    def _generate_relation_tensor(
        self, type_embedding, relation_type_constraints
    ) -> torch.Tensor:
        relation_emb_tensor = torch.empty(
            relation_type_constraints.shape[1], type_embedding.shape[1]
        )
        for relation_index, relation_type in enumerate(relation_type_constraints[0]):
            head_type_indices = torch.argwhere(relation_type).squeeze(dim=1)
            tail_type_indices = torch.argwhere(
                relation_type_constraints[1][relation_index]
            ).squeeze(dim=1)

            head_type_emb = torch.mean(type_embedding[head_type_indices], dim=0)
            tail_type_emb = torch.mean(type_embedding[tail_type_indices], dim=0)

            # TransE
            relation_emb = tail_type_emb - head_type_emb
            relation_emb_tensor[relation_index] = relation_emb

        return torch.nn.functional.normalize(relation_emb_tensor)
