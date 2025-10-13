from pykeen.triples import TriplesFactory
from pykeen.datasets import Dataset
import numpy as np
from pykeen.pipeline import pipeline


def pretrain_cate_data(cate2ids, emb_dim):
    triples = np.loadtxt(
        "/home/ni/code/CCI/data/FB_filter/cate_rel_triples.txt", dtype="str"
    )
    cat_rel_tf = TriplesFactory.from_labeled_triples(triples, entity_to_id=cate2ids)

    pretrain = pipeline(
        model="TransE",
        model_kwargs=dict(embedding_dim=emb_dim),
        dataset=Dataset.from_tf(cat_rel_tf),  # default 0.8:0.1:0.1
        training_loop="slcwa",
        training_kwargs=dict(num_epochs=100, batch_size=512),
        stopper="early",
        stopper_kwargs=dict(
            patience=5,
            metric="mrr",
            frequency=1,
        ),
        optimizer="adam",
        optimizer_kwargs=dict(lr=5e-4),
        random_seed=42,
    )

    pretrained_model = pretrain.model
    return pretrained_model.entity_representations[0]._embeddings.weight.detach()
