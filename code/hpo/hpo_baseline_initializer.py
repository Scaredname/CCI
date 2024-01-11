# Get a training dataset
import argparse
import datetime
import json
import os
import sys

import torch

sys.path.append("..")

import pykeen.datasets.utils as pdu
from Custom.CustomTrain import TypeSLCWATrainingLoop
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.datasets import get_dataset
from utilities import init_train_model, load_dataset

# dataset_name = "yago_new"
# dataset_name = "CAKE-NELL-995_new"
# dataset_name = "CAKE-DBpedia-242_new"

parser = argparse.ArgumentParser()
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
    "-m", "--model", choices=["distmult", "TransE", "RotatE", "complex"], type=str
)

if __name__ == "__main__":
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
            frequency=5,
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
    )

    initializer_list = [
        # "uniform_norm_",
        # "normal_norm_",
        # "uniform_",
        # "normal_",
        # "xavier_uniform_",
        # "xavier_normal_",
        "xavier_uniform_norm_",
        "xavier_normal_norm_",
        # "ones_",
        # "zeros_",
        # "eye_",
        # "orthogonal_",
    ]
    lr_list = [0.01, 0.001, 0.0001]

    for initializer in initializer_list:
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

        wl_initializer = WeisfeilerLehmanInitializer(
            color_initializer=initializer,
            shape=init_embedding_dim,
            triples_factory=training_data,
        )

        init_train_model(
            wl_initializer,
            "random_wl_initializer_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            model_embedding_dim,
            lr_list,
            no_constrainer=no_constrainer,
        )

    walk_position_initializer = RandomWalkPositionalEncodingInitializer(
        dim=init_embedding_dim + 1,
        triples_factory=training_data,
    )

    init_train_model(
        walk_position_initializer,
        "random_walk_position_initializer",
        dataset,
        dataset_name,
        fix_config,
        model_embedding_dim,
        lr_list,
        no_constrainer=no_constrainer,
    )

    pre_initializer = LabelBasedInitializer.from_triples_factory(
        training_data,
        encoder="transformer",
        encoder_kwargs=dict(
            pretrained_model_name_or_path="bert-base-cased",
            max_length=512,
        ),
    )
    init_train_model(
        pre_initializer,
        "pre_bert_base_cased_initializer",
        dataset,
        dataset_name,
        fix_config,
        model_embedding_dim,
        lr_list,
        no_constrainer=no_constrainer,
    )
    pre_initializer_un = LabelBasedInitializer.from_triples_factory(
        training_data,
        encoder="transformer",
        encoder_kwargs=dict(
            pretrained_model_name_or_path="bert-base-uncased",
            max_length=512,
        ),
    )
    init_train_model(
        pre_initializer_un,
        "pre_bert_base_uncased_initializer",
        dataset,
        dataset_name,
        fix_config,
        model_embedding_dim,
        lr_list,
        no_constrainer=no_constrainer,
    )
