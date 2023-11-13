import math
from typing import Collection, Optional

import torch
from pykeen.sampling import BasicNegativeSampler
from pykeen.typing import Target


def type_constrain_replacement_(
    batch: torch.LongTensor,
    positive_batch: torch.LongTensor,
    index: int,
    rel_related_ent: torch.LongTensor,
    selection: slice,
    size: int,
    max_index: int,
    neg_num_per_pos: int,
) -> None:
    """
    Replace a column of a batch of indices by type related entities.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param rel_related_ent:
        the relation related ent matrix
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """

    candidate_ents = rel_related_ent[positive_batch[:, 1]]

    candidate_ents = torch.masked_fill(
        candidate_ents, candidate_ents == positive_batch[:, index].unsqueeze(dim=-1), -1
    )  # mask the original entity
    sampled_ent = []
    for i, ent in enumerate(candidate_ents[:]):
        ent_indices = torch.where(ent != -1)[0]
        type_specific_ent_num = len(ent_indices)
        if type_specific_ent_num < neg_num_per_pos:
            sampled_ent.append(ent[ent_indices])
            random_ent = torch.randint(
                high=max_index - 1,
                size=(neg_num_per_pos - type_specific_ent_num,),
                device=batch.device,
            )
            sampled_ent.append(random_ent)
        elif type_specific_ent_num == neg_num_per_pos:
            sampled_ent.append(ent[ent_indices])
        else:
            random_indices = ent_indices[torch.randperm(ent_indices.shape[0])][
                :neg_num_per_pos
            ]
            sampled_ent.append(ent[random_indices])

    sampled_ent = torch.cat(sampled_ent)
    sampled_ent += (
        sampled_ent == batch[selection, index]
    ).long()  # 从pykeen的>= 加 1 改为了 == 后才+1

    batch[selection, index] = sampled_ent


class TypeNegativeSampler(BasicNegativeSampler):
    def __init__(
        self, *, rel_related_ent, corruption_scheme: Collection[Target] = None, **kwargs
    ) -> None:
        """
        Create a new negative sampler.

        :param rel_type_matrix:
            the relation ent matrix, shape:(2, num_rel, num_rel_related_ent)
        :param corruption_scheme:
            the corruption scheme ['h', 'r', 't']
        """
        self.rel_related_ent = rel_related_ent
        super().__init__(corruption_scheme=corruption_scheme, **kwargs)

    def corrupt_batch(
        self, positive_batch: torch.LongTensor
    ) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat_interleave(
            self.num_negs_per_pos, dim=0
        )

        # Bind the total number of negatives to sample in this batch
        total_num_negatives = negative_batch.shape[0]

        # Equally corrupt all sides
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))

        # Do not detach, as no gradients should flow into the indices.
        for index, start in zip(
            self._corruption_indices, range(0, total_num_negatives, split_idx)
        ):
            stop = min(start + split_idx, total_num_negatives)
            type_constrain_replacement_(
                batch=negative_batch,
                positive_batch=positive_batch,
                index=index,
                rel_related_ent=self.rel_related_ent[index if index == 0 else 1],
                selection=slice(start, stop),
                size=stop - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
                neg_num_per_pos=self.num_negs_per_pos
                // 2,  # 一半头一半尾，所以neg_num_per_pos减半。
            )

        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
