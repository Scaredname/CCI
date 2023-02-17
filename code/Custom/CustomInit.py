import torch
from typing import Any, Optional, Sequence


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
        if x.shape != self.tensor.shape:
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)


    def as_embedding(self, **kwargs: Any):
        """Get a static embedding from this pre-trained initializer.

        :param kwargs: Keyword arguments to pass to :class:`pykeen.nn.representation.Embedding`
        :returns: An embedding
        :rtype: pykeen.nn.representation.Embedding
        """
        from pykeen.nn.representation import Embedding

        max_id, *shape = self.tensor.shape
        return Embedding(max_id=max_id, shape=shape, initializer=self, trainable=True, **kwargs)