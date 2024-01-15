# Get a training dataset
import argparse
import datetime
import json
import os
import sys

import torch

sys.path.append("..")

import pykeen.datasets.utils as pdu
from pykeen.datasets import get_dataset
from utilities import init_train_model, load_dataset


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
            "d",
        ],
        default="a",
        type=str,
        help='a="yago_new_init", b="CAKE-NELL-995_new_init", c="CAKE-DBpedia-242_new_init", d="Kinships"',
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
        "-ag",
        "--add_gains",
        help="the list of gains for type centric add random initializer",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-pg",
        "--product_gains",
        help="the list of gains for type centric product random initializer",
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
        "-ai",
        "--all_initializer",
        help="using all initializer, will seal initializer_list argument",
        default="",
    )

    parser.add_argument(
        "-ba", "--base", help="test base initializer", action="store_true"
    )

    parser.add_argument(
        "-de", "--description", help="additional description", default=""
    )

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    dataset_typed_dict = dict(
        a="yago_new", b="CAKE-NELL-995_new", c="CAKE-DBpedia-242_new"
    )
    dataset_nontype_dict = dict(d="Kinships")
    if args.dataset not in dataset_typed_dict:
        dataset_name = dataset_nontype_dict[args.dataset]
        dataset = pdu.get_dataset(dataset=dataset_name)
        training_data = dataset.training
    else:
        dataset_name = dataset_typed_dict[args.dataset]
        training_data, validation, testing = load_dataset(
            dataset=dataset_name,
        )
        dataset = get_dataset(
            training=training_data, testing=testing, validation=validation
        )
    model = args.model

    test_batch_size = 4
    print("****************************************")
    print(dataset_name)
    print(model)
    print("****************************************")

    if model not in ["RotatE", "complex"]:
        init_embedding_dim = 768
        model_embedding_dim = init_embedding_dim
        no_constrainer = False
        data_type = torch.float
    else:
        init_embedding_dim = 768
        model_embedding_dim = init_embedding_dim // 2
        no_constrainer = True
        data_type = torch.cfloat
    lr = 0.001
    batch_size = 512
    num_negs_per_pos = 64
    epoch = 200
    adversarial_temperature = 1.0
    train = "slcwa"
    train_setting = dict(
        num_epochs=epoch,
        batch_size=batch_size,
        checkpoint_on_failure=True,
    )
    loss = "NSSALoss"

    loss_kwargs = dict(
        reduction="mean",
        adversarial_temperature=1,
        margin=9,
    )

    fix_config = dict(
        model=model,
        optimizer="adam",
        optimizer_kwargs=dict(
            lr=lr,
        ),
        training_loop=train,
        training_kwargs=train_setting,
        # negative_sampler=args.negative_sampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=num_negs_per_pos,
        ),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(
            filtered=True,
            batch_size=test_batch_size,
        ),
        # lr_scheduler="StepLR",
        # lr_scheduler_kwargs=dict(step_size=10, gamma=0.316),
        # 用early stop来筛选模型
        stopper="early",
        stopper_kwargs=dict(
            frequency=args.early_frequency,
            # frequency=frequency,
            # patience=args.early_stop_patience,
            patience=6,  # e 为1000 的情况
            relative_delta=0.0001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=test_batch_size,
        ),
        loss=loss,
        loss_kwargs=loss_kwargs,
    )

    # 每次调参前决定使用什么loss和训练方式
    # soft_loss = SoftTypeawareNegativeSmapling()
    # pipeline_config["training_loop"] = TypeSLCWATrainingLoop

    import torch
    from Custom.base_init import (
        LabelBasedInitializer,
        RandomWalkPositionalEncodingInitializer,
        WeisfeilerLehmanInitializer,
    )
    from Custom.CustomInit import (
        TypeCenterInitializer,
        TypeCenterProductRandomInitializer,
        TypeCenterRandomInitializer,
        WLCenterInitializer,
    )

    initializer_list = [
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
    ]
    # lr_list = [0.01, 0.001, 0.0001]
    lr_list = [float(lr) for lr in args.learning_rate_list]

    if not args.all_initializer:
        initializer_list = args.initializer_list

    for initializer in initializer_list:
        if args.base:
            init_train_model(
                initializer,
                initializer,
                dataset,
                dataset_name,
                fix_config,
                model_embedding_dim,
                lr_list,
                no_constrainer=no_constrainer,
            )

        if len(args.wl_centric_gains):
            maxiter = 5
            for gain in args.wl_centric_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                wl_center_initializer = WLCenterInitializer(
                    color_initializer=initializer,
                    shape=init_embedding_dim,
                    triples_factory=training_data,
                    data_type=data_type,
                    random_bias_gain=gain_num,
                    max_iter=maxiter,
                )

                init_train_model(
                    wl_center_initializer,
                    args.description
                    + f"wl{maxiter}_center_{gain}_initializer_"
                    + initializer,
                    dataset,
                    dataset_name,
                    fix_config,
                    model_embedding_dim,
                    lr_list,
                    no_constrainer=no_constrainer,
                )

        if len(args.add_gains):
            for gain in args.add_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                random_initializer = TypeCenterRandomInitializer(
                    training_data,
                    data_type,
                    type_dim=init_embedding_dim,
                    random_bias_gain=gain_num,
                    type_init=initializer,
                )

                init_train_model(
                    random_initializer,
                    args.description + f"random_initializer_{gain}_" + initializer,
                    dataset,
                    dataset_name,
                    fix_config,
                    model_embedding_dim,
                    lr_list,
                )

        if len(args.product_gains):
            for gain in args.product_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                random_product_initializer = TypeCenterProductRandomInitializer(
                    training_data,
                    torch.float,
                    type_dim=init_embedding_dim,
                    random_bias_gain=gain_num,
                    type_init=initializer,
                )

                init_train_model(
                    random_product_initializer,
                    args.description
                    + f"random_product_initializer_{gain}_"
                    + initializer,
                    dataset,
                    dataset_name,
                    fix_config,
                    model_embedding_dim,
                    lr_list,
                )
