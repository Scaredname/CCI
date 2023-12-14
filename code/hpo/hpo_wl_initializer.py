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
    test_batch_size = 4
    print("****************************************")
    print(dataset_name)
    print("****************************************")

    training_data, validation, testing = load_dataset(
        dataset=dataset_name,
    )
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    model = "TransE"
    embedding_dim = 768
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
    from Custom.CustomInit import (
        TypeCenterInitializer,
        TypeCenterProductRandomInitializer,
        TypeCenterRandomInitializer,
    )
    from pykeen.nn.init import (
        LabelBasedInitializer,
        RandomWalkPositionalEncodingInitializer,
        WeisfeilerLehmanInitializer,
    )

    initializer_list = [
        # "uniform_norm_",
        "xavier_uniform_norm_",
        "xavier_normal_norm_",
    ]
    lr_list = [0.0001]
    for initializer in initializer_list:
        # init_train_model(
        #     initializer,
        #     initializer,
        #     dataset,
        #     dataset_name,
        #     fix_config,
        #     embedding_dim,
        #     lr_list,
        # )  # baseline

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
            "pre_bert_base_cased_initializer_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )

        wl_initializer = WeisfeilerLehmanInitializer(
            color_initializer=initializer,
            shape=embedding_dim,
            triples_factory=training_data,
        )

        init_train_model(
            wl_initializer,
            "random_wl_initializer_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )

        walk_position_initializer = RandomWalkPositionalEncodingInitializer(
            dim=embedding_dim + 1,
            triples_factory=training_data,
        )

        init_train_model(
            walk_position_initializer,
            "random_walk_position_initializer_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )
