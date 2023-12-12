# Get a training dataset
import argparse
import datetime
import json
import os
import sys

sys.path.append("..")

from Custom.CustomTrain import TypeSLCWATrainingLoop
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.datasets import get_dataset
from utilities import init_train_model, load_dataset

dataset_name = "yago_new"
# dataset_name = "CAKE-NELL-995_new"
# dataset_name = "CAKE-DBpedia-242_new"

if __name__ == "__main__":
    print("****************************************")
    print(dataset_name)
    print("****************************************")

    test_batch_size = 4
    training_data, validation, testing = load_dataset(
        dataset=dataset_name,
    )
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    model = "complex"
    embedding_dim = 384
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
    loss = "crossentropy"

    loss_kwargs = dict(
        # reduction="mean",
        # adversarial_temperature=1,
        # margin=9,
    )

    regularizer = "lp"
    regularizer_kwargs = dict(
        weight=0.1,
        p=3.0,
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
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
    )

    # 每次调参前决定使用什么loss和训练方式
    # soft_loss = SoftTypeawareNegativeSmapling()
    # pipeline_config["training_loop"] = TypeSLCWATrainingLoop

    import torch
    from Custom.CustomInit import (
        TypeCenterInitializer,
        TypeCenterProductRandomInitializer,
        TypeCenterRandomInitializer,
    )

    initializer_list = [
        # "uniform_norm_",
        # "normal_norm_",
        "xavier_uniform_norm_",
        "xavier_normal_norm_",
    ]

    lr_list = [0.001]
    for initializer in initializer_list:
        init_train_model(
            initializer,
            initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )  # baseline

        random_initializer_50 = TypeCenterRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=50,
            type_init=initializer,
        )

        init_train_model(
            random_initializer_50,
            "random_initializer_50_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )

        random_initializer_1 = TypeCenterRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=1,
            type_init=initializer,
        )

        init_train_model(
            random_initializer_1,
            "random_initializer_1_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )

        random_product_initializer_1 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=1,
            type_init=initializer,
        )

        init_train_model(
            random_product_initializer_1,
            "random_product_initializer_1_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )

        random_product_initializer_0_1 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=0.1,
            type_init=initializer,
        )

        init_train_model(
            random_product_initializer_0_1,
            "random_product_initializer_0_1_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )

        random_product_initializer_0_5 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=0.5,
            type_init=initializer,
        )

        init_train_model(
            random_product_initializer_0_5,
            "random_product_initializer_0_5_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
            True,
        )
