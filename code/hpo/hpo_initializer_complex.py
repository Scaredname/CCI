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
from pykeen.pipeline import pipeline
from utilities import load_dataset

# dataset_name = "yago_new"
dataset_name = "CAKE-NELL-995_new"

if __name__ == "__main__":
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

    def train_model(entity_initializer, name):
        lr_lists = [0.01, 0.001, 0.0001]

        model_kwargs = dict(
            embedding_dim=embedding_dim,
            # relation_initializer="init_phases",
            # relation_constrainer="complex_normalize",
            entity_initializer=entity_initializer,
        )

        try:
            for learning_rate in lr_lists:
                fix_config["optimizer_kwargs"]["lr"] = learning_rate
                date_time = "/%s/%s/%s/%s" % (
                    f"{dataset_name}_init",
                    f"{name}_gain",
                    fix_config["model"],
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )

                pipeline_result = pipeline(
                    dataset=dataset,
                    model_kwargs=model_kwargs,
                    device="cuda",
                    result_tracker="tensorboard",
                    result_tracker_kwargs=dict(
                        experiment_path="../../result/hpo_init/" + date_time,
                    ),
                    **fix_config,
                )

                model_path = "../../models/" + date_time
                pipeline_result.metadata = fix_config
                pipeline_result.save_to_directory(model_path)
        except:
            print(f"experiment: {str(entity_initializer)} failed")

    initializer_list = [
        "uniform_norm_",
        "normal_norm_",
        "xavier_uniform_norm_",
        "xavier_normal_norm_",
    ]

    for initializer in initializer_list:
        train_model(initializer, initializer)  # baseline

        random_initializer_50 = TypeCenterRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=50,
            type_init=initializer,
        )

        train_model(random_initializer_50, "random_initializer_50_" + initializer)

        random_initializer_1 = TypeCenterRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=1,
            type_init=initializer,
        )

        train_model(random_initializer_1, "random_initializer_1_" + initializer)

        random_product_initializer_1 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=1,
            type_init=initializer,
        )

        train_model(
            random_product_initializer_1, "random_product_initializer_1_" + initializer
        )

        random_product_initializer_0_1 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=0.1,
            type_init=initializer,
        )

        train_model(
            random_product_initializer_0_1,
            "random_product_initializer_0_1_" + initializer,
        )

        random_product_initializer_0_5 = TypeCenterProductRandomInitializer(
            training_data,
            torch.cfloat,
            type_dim=768,
            random_bias_gain=0.5,
            type_init=initializer,
        )

        train_model(
            random_product_initializer_0_5,
            "random_product_initializer_0_5_" + initializer,
        )
