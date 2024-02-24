from typing import Any, Optional, Sequence

import torch
import torch.nn
import torch.nn.init
import torch_ppr.utils
from class_resolver import Hint, HintOrType, OptionalKwargs
from more_itertools import last
from pykeen.nn.init import initializer_resolver
from pykeen.nn.text import TextEncoder, text_encoder_resolver
from pykeen.nn.utils import iter_matrix_power, safe_diagonal
from pykeen.triples import CoreTriplesFactory, TriplesFactory
from pykeen.typing import Initializer, MappedTriples, OneOrSequence
from pykeen.utils import get_edge_index, iter_weisfeiler_lehman, upgrade_to_sequence


class PretrainedInitializer:
    """
    Initialize tensor with pretrained weights.

    Example usage:

    .. code-block::

        import torch
        from pykeen.pipeline import pipeline
        from pykeen.nn.init import PretrainedInitializer

        # this is usually loaded from somewhere else
        # the shape must match, as well as the entity-to-id mapping
        pretrained_embedding_tensor = torch.rand(14, 128)

        result = pipeline(
            dataset="nations",
            model="transe",
            model_kwargs=dict(
                embedding_dim=pretrained_embedding_tensor.shape[-1],
                entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor),
            ),
        )
    """

    def __init__(self, tensor: torch.FloatTensor) -> None:
        """
        Initialize the initializer.

        :param tensor:
            the tensor of pretrained embeddings.
        """
        self.tensor = tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor with the given tensor."""
        # for some cfloat model.
        if len(x.shape) != len(self.tensor.shape):
            self.tensor = self.tensor.view(*x.shape[:-2], -1, x.shape[-1])
        if x.shape != self.tensor.shape:
            raise ValueError(
                f"shape does not match: expected {self.tensor.shape} but got {x.shape}"
            )
        return self.tensor.to(device=x.device, dtype=x.dtype)

    def as_embedding(self, **kwargs: Any):
        """Get a static embedding from this pre-trained initializer.

        :param kwargs: Keyword arguments to pass to :class:`pykeen.nn.representation.Embedding`
        :returns: An embedding
        :rtype: pykeen.nn.representation.Embedding
        """
        from pykeen.nn.representation import Embedding

        max_id, *shape = self.tensor.shape
        return Embedding(
            max_id=max_id, shape=shape, initializer=self, trainable=False, **kwargs
        )


class LabelBasedInitializer(PretrainedInitializer):
    """
    An initializer using pretrained models from the `transformers` library to encode labels.

    Example Usage:

    Initialize entity representations as Transformer encodings of their labels. Afterwards,
    the parameters are detached from the labels, and trained on the KGE task without any
    further connection to the Transformer model.

    .. code-block :: python

        from pykeen.datasets import get_dataset
        from pykeen.nn.init import LabelBasedInitializer
        from pykeen.models import ERMLPE

        dataset = get_dataset(dataset="nations")
        model = ERMLPE(
            embedding_dim=768,  # for BERT base
            entity_initializer=LabelBasedInitializer.from_triples_factory(
                triples_factory=dataset.training,
                encoder="transformer",
            ),
        )
    """

    def __init__(
        self,
        labels: Sequence[str],
        encoder: HintOrType[TextEncoder] = None,
        encoder_kwargs: OptionalKwargs = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the initializer.

        :param labels:
            the labels
        :param encoder:
            the text encoder to use, cf. `text_encoder_resolver`
        :param encoder_kwargs:
            additional keyword-based parameters passed to the encoder
        :param batch_size: >0
            the (maximum) batch size to use while encoding. If None, use `len(labels)`, i.e., only a single batch.
        """
        super().__init__(
            tensor=text_encoder_resolver.make(encoder, encoder_kwargs).encode_all(
                labels=labels,
                batch_size=batch_size,
            )
            # must be cloned if we want to do backprop
            .clone(),
        )

    @classmethod
    def from_triples_factory(
        cls,
        triples_factory: TriplesFactory,
        for_entities: bool = True,
        **kwargs,
    ) -> "LabelBasedInitializer":
        """
        Prepare a label-based initializer with labels from a triples factory.

        :param triples_factory:
            the triples factory
        :param for_entities:
            whether to create the initializer for entities (or relations)
        :param kwargs:
            additional keyword-based arguments passed to :func:`LabelBasedInitializer.__init__`
        :returns:
            A label-based initializer

        :raise ImportError:
            if the transformers library could not be imported
        """
        id_to_label = (
            triples_factory.entity_id_to_label
            if for_entities
            else triples_factory.relation_id_to_label
        )
        labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
        return cls(
            labels=labels,
            **kwargs,
        )


class WeisfeilerLehmanInitializer(PretrainedInitializer):
    """An initializer based on an encoding of categorical colors from the Weisfeiler-Lehman algorithm."""

    def __init__(
        self,
        *,
        # the color initializer
        color_initializer: Hint[Initializer] = None,
        color_initializer_kwargs: OptionalKwargs = None,
        shape: OneOrSequence[int] = 32,
        # variants for the edge index
        edge_index: Optional[torch.LongTensor] = None,
        num_entities: Optional[int] = None,
        mapped_triples: Optional[torch.LongTensor] = None,
        triples_factory: Optional[CoreTriplesFactory] = None,
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
        # get coloring
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
        # init entity representations according to the color
        super().__init__(tensor=color_representation[colors])


class RandomWalkPositionalEncodingInitializer(PretrainedInitializer):
    r"""
    Initialize nodes via random-walk positional encoding.

    The random walk positional encoding is given as

    .. math::
        \mathbf{x}_i = [\mathbf{R}_{i, i}, \mathbf{R}^{2}_{i, i}, \ldots, \mathbf{R}^{d}_{i, i}] \in \mathbb{R}^{d}

    where $\mathbf{R} := \mathbf{A}\mathbf{D}^{-1}$ is the random walk matrix, with
    $\mathbf{D} := \sum_i \mathbf{A}_{i, i}$.

    .. seealso::
        https://arxiv.org/abs/2110.07875
    """

    def __init__(
        self,
        *,
        triples_factory: Optional[CoreTriplesFactory] = None,
        mapped_triples: Optional[MappedTriples] = None,
        edge_index: Optional[torch.Tensor] = None,
        dim: int,
        num_entities: Optional[int] = None,
        space_dim: int = 0,
        skip_first_power: bool = True,
    ) -> None:
        """
        Initialize the positional encoding.

        One of `triples_factory`, `mapped_triples` or `edge_index` will be used.
        The preference order is:

        1. `triples_factory`
        2. `mapped_triples`
        3. `edge_index`

        :param triples_factory:
            the triples factory
        :param mapped_triples: shape: `(m, 3)`
            the mapped triples
        :param edge_index: shape: `(2, m)`
            the edge index
        :param dim:
            the dimensionality
        :param num_entities:
            the number of entities. If `None`, it will be inferred from `edge_index`
        :param space_dim:
            estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.
        :param skip_first_power:
            in most cases the adjacencies diagonal values will be zeros (since reflexive edges are not that common).
            This flag enables skipping the first matrix power.
        """
        edge_index = get_edge_index(
            triples_factory=triples_factory,
            mapped_triples=mapped_triples,
            edge_index=edge_index,
        )
        # create random walk matrix
        rw = torch_ppr.utils.prepare_page_rank_adjacency(
            edge_index=edge_index, num_nodes=num_entities
        )
        # stack diagonal entries of powers of rw
        tensor = torch.stack(
            [
                (i ** (space_dim / 2.0)) * safe_diagonal(matrix=power)
                for i, power in enumerate(
                    iter_matrix_power(matrix=rw, max_iter=dim), start=1
                )
                if not skip_first_power or i > 1
            ],
            dim=-1,
        )
        super().__init__(tensor=tensor)
