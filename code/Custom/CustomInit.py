from typing import Any, Optional, Sequence

import torch
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn import Representation
from pykeen.nn.init import LabelBasedInitializer, PretrainedInitializer, xavier_uniform_
from pykeen.triples import KGInfo
from torch import FloatTensor

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


class TypeCenterInitializer(PretrainedInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        type_dim=None,
        type_init="xavier_uniform_",
        pretrain=None,
        shape: Sequence[str] = ("d",),
    ) -> None:
        # 在读取预训练表示时，设置为float避免pykeen生成表示时随机生成一些参数。设置为float才能确保完全利用预训练的表示。
        type_representations_kwargs = dict(
            dtype=torch.float, shape=type_dim, initializer=None
        )

        if pretrain:
            print(f"using pretrained model '{pretrain}' to initialize type embedding")
            type_labels = list(triples_factory.types_to_id.keys())
            encoder_kwargs = dict(
                pretrained_model_name_or_path=pretrain,
                max_length=512,
            )
            type_init = LabelBasedInitializer(
                labels=type_labels, encoder="transformer", encoder_kwargs=encoder_kwargs
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
            self.type_dim = type_dim // 2  # 用于像模型传递参数
        else:
            self.type_dim = type_dim

        super().__init__(tensor)

    # def __call__(self, x: torch.Tensor) -> torch.Tensor:
    #     """Initialize the tensor with the given tensor."""
    #     if len(x.shape) > 2:
    #         self.tensor = self.tensor.view(self.tensor.shape[0], -1, 2)
    #     if x.shape != self.tensor.shape:
    #         raise ValueError(
    #             f"shape does not match: expected {self.tensor.shape} but got {x.shape}"
    #         )
    #     return self.tensor.to(device=x.device, dtype=x.dtype)

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
        **kwargs,
    ) -> None:
        self.gain = random_bias_gain
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

            random_bias_emb = xavier_uniform_(
                torch.empty(*type_emb.shape), gain=self.gain
            )
            # random_bias_emb = 0.5 * torch.nn.functional.normalize(random_bias_emb)
            # print("random_bias_emb norm :", torch.norm(random_bias_emb, dim=1))
            # random_bias_emb = 0

            ent_emb = torch.mean((type_emb + random_bias_emb), dim=0)
            entity_emb_tensor[entity_index] = ent_emb

        return entity_emb_tensor
