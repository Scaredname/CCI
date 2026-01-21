# test_fastnode2vec_initializer.py
# Minimal runnable script to test FastNode2VecInitializer with your existing pipeline.
#
# Usage example:
#   python test_fastnode2vec_initializer.py -d a -m TransE -lr 0.001 0.0005 -esf 5 --epochs 5 --dim 200
#
# Notes:
# - This script assumes:
#   1) you already have `read_data()` and `train_model()` in utilities.py
#   2) FastNode2VecInitializer is implemented in Customize/custom_initialization.py (or adjust import)
#   3) your read_data() returns PyKEEN TriplesFactory objects (training/validation/testing)

import argparse
import json

import torch
from pykeen.datasets import get_dataset

from utilities import read_data, train_model


def init_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-d",
        "--dataset",
        choices=["a", "b", "c", "d"],
        default="a",
        type=str,
        help='a="yago_new", b="NELL-995_new", c="FB_new", d="FB_filter"',
    )
    p.add_argument(
        "-m",
        "--model",
        choices=["distmult", "TransE", "RotatE", "complex"],
        required=True,
        type=str,
    )
    p.add_argument(
        "-lr",
        "--learning_rate_list",
        nargs="+",
        required=True,
        help="one or multiple learning rates, e.g., -lr 0.001 0.0005",
    )
    p.add_argument("-esf", "--early_frequency", type=int, default=5)

    # FastNode2Vec hyperparams (minimal)
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--walk_length", type=int, default=100)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--workers", type=int, default=1)

    p.add_argument(
        "-np",
        "--no_process",
        help="don't use any process function",
        action="store_true",
        default=False,
    )

    p.add_argument("-de", "--description", default="node2vec_test_", type=str)
    return p


def load_config(model: str):
    with open("base_config.json", "r") as f:
        base_config = json.load(f)

    if model in ["TransE", "RotatE"]:
        loss = "NSSALoss"
        loss_kwargs = dict(reduction="mean", adversarial_temperature=1, margin=9)
        regularizer = None
        regularizer_kwargs = None
    else:
        loss = "crossentropy"
        loss_kwargs = None
        if model == "complex":
            regularizer = "lp"
            regularizer_kwargs = dict(weight=0.1, p=3.0)
        elif model == "distmult":
            regularizer = "lp"
            regularizer_kwargs = dict(weight=0.01)

    fix_config = dict(
        model=model,
        optimizer="adam",
        optimizer_kwargs=dict(lr=0.001),
        training_loop=base_config["train"],
        training_kwargs=base_config["train_setting"],
        negative_sampler_kwargs=dict(num_negs_per_pos=base_config["num_negs_per_pos"]),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True, batch_size=base_config["test_batch_size"]),
        stopper="early",
        stopper_kwargs=dict(
            patience=6,
            relative_delta=0.0001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=base_config["test_batch_size"],
        ),
        loss=loss,
        loss_kwargs=loss_kwargs,
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
    )
    return fix_config, base_config["init_embedding_dim"]


def main():
    args = init_parser().parse_args()

    dataset_dict = dict(a="yago_new", b="NELL-995_new", c="FB_new", d="FB_filter")
    dataset_name = dataset_dict[args.dataset]

    training_data, validation, testing = read_data(data_name=dataset_name)
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    config, init_embedding_dim = load_config(args.model)
    config["stopper_kwargs"]["frequency"] = args.early_frequency

    # RotatE/ComplEx use complex-valued representations in your pipeline
    if args.model in ["RotatE", "complex"]:
        model_embedding_dim = init_embedding_dim // 2
        no_constrainer = True
        data_type = torch.cfloat
    else:
        model_embedding_dim = init_embedding_dim
        no_constrainer = False
        data_type = torch.float

    lr_list = [float(x) for x in args.learning_rate_list]
    process_function = "no" if args.no_process else "lp_normalize"

    # Import your initializer (adjust path if needed)
    from Customize.custom_initialization import (
        FastNode2VecInitializer,
        FastNode2VecParams,
    )

    n2v_params = FastNode2VecParams(
        dim=init_embedding_dim,  # IMPORTANT: match KGE init dim
        p=args.p,
        q=args.q,
        walk_length=args.walk_length,
        window=args.window,
        epochs=args.epochs,
        workers=args.workers,
        directed=False,
        weighted=False,
    )

    initializer = FastNode2VecInitializer(
        triples_factory=training_data,  # build graph from TRAIN triples (recommended)
        num_entities=training_data.num_entities,
        n2v_params=n2v_params,
        shape=init_embedding_dim,
        data_type=data_type,
        process_function=process_function,
        make_undirected=True,
    )

    exp_name = (
        f"node2vec_dim{init_embedding_dim}_p{args.p}_q{args.q}_"
        f"wl{args.walk_length}_ctx{args.window}_ep{args.epochs}_"
        f"es{args.early_frequency}_process{process_function}"
    )

    train_model(
        initializer,
        exp_name,
        args.description,
        dataset,
        dataset_name,
        config,
        model_embedding_dim,
        lr_list,
        no_constrainer=no_constrainer,
    )


if __name__ == "__main__":
    main()
