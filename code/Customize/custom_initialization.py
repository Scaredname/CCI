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
        data_type=torch.float,
        alpha=1.0,
        if_plus_random: int = 1,
        process_function="lp_normalize",
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
            colors.new_empty(num_colors, *shape, dtype=torch.get_default_dtype())
        )
        random_representation = color_initializer(
            colors.new_empty(colors.shape[0], *shape, dtype=torch.get_default_dtype())
        )

        entity_emb_tensor = (
            color_representation[colors] / alpha
            + if_plus_random * random_representation
        )
        tensor = process_tensor(entity_emb_tensor, process_function)

        if data_type == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)
        # init entity representations according to the color
        super().__init__(tensor=tensor)


class CategoryCenterInitializer(PretrainedInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        category_dim=None,
        category_init="xavier_uniform_",
        noise_init="xavier_uniform_",
        pretrain=None,
        category_emb=None,
        shape: Sequence[str] = ("d",),
    ) -> None:
        """
        description:
        param self:
        param triples_factory:
        param data_type:
        param category_dim:
        param category_init:
        param pretrain:
        param category_emb:为了确保和relation使用的是相同的类型嵌入，数据类型是Sequence[Representation]
        param shape:
        return {*}
        """
        # todo: 像wl那样直接生成tensor而不是pykeen中的表示。

        # 在读取预训练表示时，设置为float避免pykeen生成表示时随机生成一些参数。设置为float才能确保完全利用预训练的表示。
        category_representations_kwargs = dict(
            dtype=torch.float, shape=category_dim, initializer=None
        )
        self.noise_init = noise_init

        if category_emb:
            self.category_representations = category_emb
        else:
            if pretrain:
                print(
                    f"using pretrained model 'bert-base-uncased' to initialize category embeddings"
                )
                category_labels = list(triples_factory.categories_to_ids.keys())
                encoder_kwargs = dict(
                    pretrained_model_name_or_path="bert-base-uncased",
                    max_length=512,
                )
                category_init = LabelBasedInitializer(
                    labels=category_labels,
                    encoder="transformer",
                    encoder_kwargs=encoder_kwargs,
                )
                category_representations_kwargs["initializer"] = category_init
                category_dim = category_init.as_embedding().shape[0]
                category_representations_kwargs["shape"] = category_dim
            else:
                category_representations_kwargs["initializer"] = category_init

            self.category_representations = self._build_category_representations(
                triples_factory=triples_factory,
                shape=shape,
                representations=None,
                representations_kwargs=category_representations_kwargs,
                skip_checks=False,
            )

        self.category_representations[0].reset_parameters()

        # import numpy as np

        # run_name = "bert_10_np"
        # np.save(
        #     f"../result/visualization/{run_name}/cate_emb.npy",
        #     self.category_representations[0]._embeddings.weight.detach().cpu().numpy(),
        # )

        # tensor = self._generate_entity_tensor(
        #     self.category_representations[0]._embeddings.weight,
        #     triples_factory.ents_cates_adj_matrix.float(),
        # )

        tensor = self._generate_entity_tensor(
            self.category_representations[0]._embeddings.weight,
            triples_factory.ents_cates_adj_matrix.float(),
            triples_factory.ent_rel_fre,
        )

        if data_type == torch.cfloat:
            tensor = tensor.view(tensor.shape[0], -1, 2)

        del self.category_representations
        torch.cuda.empty_cache()

        super().__init__(tensor)

    def _build_category_representations(
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
            max_id=triples_factory.num_category,
            shapes=shape,
            label="category",
            **kwargs,
        )

    def _generate_entity_tensor(
        self, category_embedding, entity_category_constraints
    ) -> torch.Tensor:
        if torch.any(torch.sum(entity_category_constraints, dim=1) > 1):
            entity_category_constraints = L1_normalize_each_rows_of_matrix(
                entity_category_constraints
            )
        return torch.matmul(entity_category_constraints, category_embedding)

    def _generate_entity_tensor_gpu():
        pass  # pragma: no cover


class CategoryCenterRandomInitializer(CategoryCenterInitializer):
    def __init__(
        self,
        triples_factory,
        data_type,
        alpha=1.0,
        if_plus_random: int = 1,
        category_dim=None,
        pretrain=None,
        shape: Sequence[str] = ("d",),
        process_function="lp_normalize",
        **kwargs,
    ) -> None:
        assert if_plus_random in [0, 1]
        self.gain = alpha
        self.plus_random = if_plus_random
        self.process_function = process_function
        super().__init__(
            triples_factory,
            data_type=data_type,
            category_dim=category_dim,
            pretrain=pretrain,
            shape=shape,
            **kwargs,
        )

    def _generate_entity_tensor(
        self, category_embedding, entity_category_constraints, ent_rel_fre
    ) -> torch.Tensor:
        entity_emb_tensor = torch.zeros(
            entity_category_constraints.shape[0], category_embedding.shape[1]
        )

        discard_threshold = torch.tensor(20000)
        ent_rel_fre[ent_rel_fre >= discard_threshold] = discard_threshold
        ent_weights = ent_rel_fre / ent_rel_fre.max()

        initializer = initializer_resolver.make(self.noise_init)
        for entity_index, entity_category in enumerate(entity_category_constraints):
            category_indices = torch.argwhere(entity_category).squeeze(dim=1)
            weight = ent_weights[entity_index]

            if category_indices.numel() == 0:
                category_emb = torch.zeros(
                    1, category_embedding.shape[1], dtype=category_embedding.dtype
                )
            else:
                category_emb = category_embedding[category_indices]

            random_bias_emb = initializer(torch.empty(*category_emb.shape))

            # 存在多个类型时，求加和后的嵌入的平均作为实体嵌入
            ent_emb = torch.mean(
                (category_emb * (1 - weight) + weight * random_bias_emb), dim=0
            )
            entity_emb_tensor[entity_index] = ent_emb

        return process_tensor(entity_emb_tensor, self.process_function)

    # 暂时没用
    def _generate_entity_tensor_matrix(
        self, category_embedding, entity_category_constraints, ent_rel_fre
    ) -> torch.Tensor:
        random_emb_tensor = torch.empty(
            entity_category_constraints.shape[0], category_embedding.shape[1]
        ).to("cuda")
        initializer = initializer_resolver.make(self.noise_init)
        category_embedding = category_embedding.to("cuda")

        entity_category_constraints = entity_category_constraints.to("cuda")
        random_emb_tensor = initializer(random_emb_tensor)

        ent_weights = (ent_rel_fre / ent_rel_fre.max()).to("cuda")

        entity_average_category_embedding = torch.matmul(
            entity_category_constraints, category_embedding
        )

        entity_emb_tensor = (
            ent_weights[:, None] * entity_average_category_embedding
            + (1 - ent_weights)[:, None] * random_emb_tensor
        )

        return process_tensor(entity_emb_tensor, self.process_function)
