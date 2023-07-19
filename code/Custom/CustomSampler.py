import math
from typing import Collection, Optional

import torch
from pykeen.sampling import BasicNegativeSampler
from pykeen.typing import Target


def type_constrain_replacement_(batch: torch.LongTensor, index: int, rel_related_ent:torch.LongTensor, selection: slice, size: int, max_index: int, ) -> None:
    """
    Replace a column of a batch of indices by type related entities.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """

    print(batch[selection])

    replacement = torch.randint(
        high=max_index - 1,
        size=(size,),
        device=batch.device,
    )
    replacement += (replacement >= batch[selection, index]).long()
    batch[selection, index] = replacement

class TypeNegativeSampler(BasicNegativeSampler):
    def __init__(self, *, rel_related_ent, corruption_scheme: Collection[Target] = None, **kwargs) -> None:
        """
        Create a new negative sampler.

        :param rel_type_matrix:
            the relation ent matrix, shape:(2, num_rel, num_rel_related_ent)
        :param corruption_scheme:
            the corruption scheme ['h', 'r', 't']
        """
        self.rel_related_ent = rel_related_ent
        super().__init__(corruption_scheme=corruption_scheme, **kwargs)


    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(self.num_negs_per_pos, dim=0)

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)
            type_constrain_replacement_(
                batch=negative_batch,
                index=index,
                rel_related_ent = self.rel_related_ent[index if index == 0 else 1],
                selection=slice(start, stop),
                size=stop - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)