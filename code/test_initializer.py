# Get a training dataset
import argparse
import json

import torch
from pykeen.datasets import get_dataset
from utilities import read_data, train_model

INITIALIZERs = [
    "uniform_norm_",
    "normal_norm_",
    "uniform_",
    "normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "xavier_uniform_norm_",
    "xavier_normal_norm_",
    "orthogonal_",
]


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-esf", "--early_frequency", type=int, default=5)
    parser.add_argument(
        "-d",
        "--dataset",
        choices=[
            "a",
            "b",
            "c",
            "d",  # d is for FB_filter
        ],
        default="a",
        type=str,
        help='a="yago6k_103", b="NELL995", c="FB15k237", d="FB_filter"',
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["distmult", "TransE", "RotatE", "complex"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-wcg",
        "--wl_centric_gains",
        help="the list of gains for wl centric add random initializer",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-cg",
        "--CCI_gains",
        help="the list of gains for category centric add random initializer",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-lr",
        "--learning_rate_list",
        nargs="+",
        help="input one or multiple learning rate",
        required=True,
    )
    parser.add_argument(
        "-init",
        "--initializer_list",
        nargs="+",
        choices=[
            "uniform_norm_",
            "normal_norm_",
            "uniform_",
            "normal_",
            "xavier_uniform_",
            "xavier_normal_",
            "xavier_uniform_norm_",
            "xavier_normal_norm_",
            "ones_",
            "zeros_",
            "eye_",
            "orthogonal_",
        ],
        default=[
            "xavier_uniform_norm_",
            "xavier_normal_norm_",
        ],
    )

    parser.add_argument(
        "-cinit",
        "--category_initializer",
        default="normal_",
        choices=["normal_", "uniform_"],
    )

    parser.add_argument(
        "-ai",
        "--all_initializer",
        help="using all initializer, will seal initializer_list argument",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-ba", "--base", help="test base initializer", action="store_true"
    )

    parser.add_argument(
        "-np",
        "--no_process",
        help="don't use any process function",
        action="store_true",
    )
    parser.add_argument(
        "-nr", "--no_random", help="don't plus random emb", action="store_true"
    )

    parser.add_argument(
        "-de", "--description", help="additional description", default=""
    )

    parser.add_argument(
        "-pre",
        "--PreTrain",
        default=False,
        action="store_true",
        help="Whether use pre-trained type embeddings, bert-base-uncased, ",
    )

    return parser


def load_config(model: str):
    with open("base_config.json", "r") as f:
        base_config = json.load(f)

    if model in ["TransE", "RotatE"]:
        loss = "NSSALoss"
        loss_kwargs = dict(
            reduction="mean",
            adversarial_temperature=1,
            margin=9,
        )
        regularizer = None
        regularizer_kwargs = None
    else:
        loss = "crossentropy"
        loss_kwargs = None
        if model == "complex":
            regularizer = "lp"
            regularizer_kwargs = dict(
                weight=0.1,
                p=3.0,
            )
        elif model == "distmult":
            regularizer = "lp"
            regularizer_kwargs = dict(weight=0.01)

    fix_config = dict(
        model=model,
        optimizer="adam",
        optimizer_kwargs=dict(
            lr=0.001,
        ),
        training_loop=base_config["train"],
        training_kwargs=base_config["train_setting"],
        # negative_sampler=args.negative_sampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=base_config["num_negs_per_pos"],
        ),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(
            filtered=True,
            batch_size=base_config["test_batch_size"],
        ),
        # lr_scheduler="StepLR",
        # lr_scheduler_kwargs=dict(step_size=10, gamma=0.316),
        stopper="early",
        stopper_kwargs=dict(
            patience=6,  # e 为1000 的情况
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


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    dataset_dict = dict(
        a="yago_new",
        b="NELL-995_new",
        c="FB_new",
        d="FB_filter",
    )
    dataset_name = dataset_dict[args.dataset]
    training_data, validation, testing = read_data(
        data_name=dataset_name,
    )
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )
    model = args.model

    print("****************************************")
    print(dataset_name)
    print(model)
    print("****************************************")

    config, init_embedding_dim = load_config(model)
    config["stopper_kwargs"]["frequency"] = args.early_frequency

    if model not in ["RotatE", "complex"]:
        init_embedding_dim = init_embedding_dim
        model_embedding_dim = init_embedding_dim
        no_constrainer = False
        data_type = torch.float
    else:
        init_embedding_dim = init_embedding_dim
        model_embedding_dim = init_embedding_dim // 2
        no_constrainer = True
        data_type = torch.cfloat

    import torch
    from Customize.custom_initialization import (
        CategoryCenterRandomInitializer,
        WLCenterInitializer,
    )

    lr_list = [float(lr) for lr in args.learning_rate_list]

    if not args.all_initializer:
        initializer_list = args.initializer_list
    else:
        initializer_list = INITIALIZERs

    if args.no_random:
        if_plus_random = 0
        args.description += "without_random_"
    else:
        if_plus_random = 1

    if args.no_process:
        process_function = "no"
        args.description += "no_process_"
        print("no process function")
    else:
        process_function = "lp_normalize"
        print("use lp_normalize process function")

    for initializer in initializer_list:
        if args.base:
            train_model(
                initializer,
                initializer,
                args.description,
                dataset,
                dataset_name,
                config,
                model_embedding_dim,
                lr_list,
                no_constrainer=no_constrainer,
            )

        if len(args.wl_centric_gains):
            maxiter = 5  # fix for yago dataset

            for gain in args.wl_centric_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                wl_center_initializer = WLCenterInitializer(
                    color_initializer=initializer,
                    shape=init_embedding_dim,
                    triples_factory=training_data,
                    data_type=data_type,
                    alpha=gain_num,
                    max_iter=maxiter,
                    process_function=process_function,
                    if_plus_random=if_plus_random,
                )

                train_model(
                    wl_center_initializer,
                    f"wl{maxiter}_center_{gain}_initializer_" + initializer,
                    args.description,
                    dataset,
                    dataset_name,
                    config,
                    model_embedding_dim,
                    lr_list,
                    no_constrainer=no_constrainer,
                )

        if len(args.CCI_gains):
            for gain in args.CCI_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))

                random_initializer = CategoryCenterRandomInitializer(
                    training_data,
                    data_type,
                    category_dim=init_embedding_dim,
                    alpha=gain_num,
                    category_init=args.category_initializer,
                    noise_init=initializer,
                    process_function=process_function,
                    if_plus_random=if_plus_random,
                    pretrain=args.PreTrain,
                )

                if args.PreTrain:
                    initializer = "bert-base-uncased"

                train_model(
                    random_initializer,
                    f"{args.category_initializer}nCCI_{gain}_" + initializer,
                    args.description,
                    dataset,
                    dataset_name,
                    config,
                    model_embedding_dim,
                    lr_list,
                    no_constrainer=no_constrainer,
                )
