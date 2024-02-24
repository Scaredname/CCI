# Get a training dataset
import argparse

import pykeen.datasets.utils as pdu
import torch
from pykeen.datasets import get_dataset
from utilities import init_train_model, read_data


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-esf", "--early_frequency", type=int, default=5)
    parser.add_argument(
        "-d",
        "--dataset",
        choices=[
            "a",
            "b",
        ],
        default="a",
        type=str,
        help='a="yago6k_103", b="NELL995"',
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
        "-wcgn",
        "--wl_centric_gains_no",
        help="the list of gains for wl centric add random initializer with no process",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-ag",
        "--add_gains",
        help="the list of gains for cate centric add random initializer",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-agn",
        "--add_gains_no",
        help="the list of gains for cate centric add random initializer for no process",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-pg",
        "--product_gains",
        help="the list of gains for cate centric product random initializer",
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
        "-nr", "--no_random", help="don't plus random emb", action="store_true"
    )

    parser.add_argument(
        "-de", "--description", help="additional description", default=""
    )

    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    dataset_cated_dict = dict(
        a="yago_new",
        b="NELL-995_new",
    )
    dataset_name = dataset_cated_dict[args.dataset]
    training_data, validation, testing = read_data(
        data_name=dataset_name,
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
        data_cate = torch.float
    else:
        init_embedding_dim = 768
        model_embedding_dim = init_embedding_dim // 2
        no_constrainer = True
        data_cate = torch.cfloat
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
        loss_kwargs = dict()
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
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
    )

    import torch
    from Customize.base_init import (
        LabelBasedInitializer,
        RandomWalkPositionalEncodingInitializer,
        WeisfeilerLehmanInitializer,
    )
    from Customize.custom_initialization import (
        CateCenterRandomInitializer,
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
        "orthogonal_",
    ]
    # lr_list = [0.01, 0.001, 0.0001]
    lr_list = [float(lr) for lr in args.learning_rate_list]

    if not args.all_initializer:
        initializer_list = args.initializer_list

    if args.no_random:
        if_plus_random = 0
        args.description += "without_random_"
    else:
        if_plus_random = 1

    for initializer in initializer_list:
        if args.base:
            init_train_model(
                initializer,
                args.description + initializer,
                dataset,
                dataset_name,
                fix_config,
                model_embedding_dim,
                lr_list,
                no_constrainer=no_constrainer,
            )

        if len(args.wl_centric_gains):
            maxiter = 5  # fix
            for gain in args.wl_centric_gains:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                wl_center_initializer = WLCenterInitializer(
                    color_initializer=initializer,
                    shape=init_embedding_dim,
                    triples_factory=training_data,
                    data_cate=data_cate,
                    random_bias_gain=gain_num,
                    max_iter=maxiter,
                    if_plus_random=if_plus_random,
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

        if len(args.wl_centric_gains_no):
            maxiter = 5  # fix
            for gain in args.wl_centric_gains_no:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                wl_center_initializer = WLCenterInitializer(
                    color_initializer=initializer,
                    shape=init_embedding_dim,
                    triples_factory=training_data,
                    data_cate=data_cate,
                    random_bias_gain=gain_num,
                    max_iter=maxiter,
                    preprocess="no",
                    if_plus_random=if_plus_random,
                )

                init_train_model(
                    wl_center_initializer,
                    args.description
                    + f"no_wl{maxiter}_center_{gain}_initializer_"
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
                random_initializer = CateCenterRandomInitializer(
                    training_data,
                    data_cate,
                    cate_dim=init_embedding_dim,
                    random_bias_gain=gain_num,
                    cate_init=initializer,
                    if_plus_random=if_plus_random,
                )

                init_train_model(
                    random_initializer,
                    args.description + f"random_initializer_{gain}_" + initializer,
                    dataset,
                    dataset_name,
                    fix_config,
                    model_embedding_dim,
                    lr_list,
                    no_constrainer=no_constrainer,
                )

        if len(args.add_gains_no):
            for gain in args.add_gains_no:
                gain_num = float(gain)
                if gain_num < 1:
                    gain = "_".join(gain.split("."))
                random_initializer = CateCenterRandomInitializer(
                    training_data,
                    data_cate,
                    cate_dim=init_embedding_dim,
                    random_bias_gain=gain_num,
                    cate_init=initializer,
                    preprocess="no",
                    if_plus_random=if_plus_random,
                )

                init_train_model(
                    random_initializer,
                    args.description + f"no_random_initializer_{gain}_" + initializer,
                    dataset,
                    dataset_name,
                    fix_config,
                    model_embedding_dim,
                    lr_list,
                    no_constrainer=no_constrainer,
                )
