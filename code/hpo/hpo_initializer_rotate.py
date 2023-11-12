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
from utilities import load_dataset

if __name__ == "__main__":
    training_data, validation, testing = load_dataset(
        dataset="yago_new",
    )
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    model = "RotatE"
    embedding_dim = 364
    lr = 0.0005
    batch_size = 512
    num_negs_per_pos = 64
    epoch = 1000
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
            batch_size=16,
        ),
        lr_scheduler="StepLR",
        lr_scheduler_kwargs=dict(step_size=500, gamma=0.1),
        # 用early stop来筛选模型
        stopper="early",
        stopper_kwargs=dict(
            frequency=100,
            # frequency=frequency,
            # patience=args.early_stop_patience,
            patience=9,  # e 为1000 的情况
            relative_delta=0.0001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=16,
        ),
        loss=loss,
        loss_kwargs=loss_kwargs,
    )

    # 每次调参前决定使用什么loss和训练方式
    # soft_loss = SoftTypeawareNegativeSmapling()
    # pipeline_config["training_loop"] = TypeSLCWATrainingLoop

    import torch
    from Custom.CustomInit import TypeCenterInitializer, TypeCenterRandomInitializer

    type_center_initializer_base = TypeCenterInitializer(
        training_data,
        torch.cfloat,
        type_dim=768,
        pretrain="bert-base-uncased",
    )
    type_center_initializer_random = TypeCenterInitializer(
        training_data,
        torch.cfloat,
        type_dim=768,
        # pretrain="bert-base-uncased",
    )
    type_center_random_initializer_random = TypeCenterRandomInitializer(
        training_data,
        torch.cfloat,
        type_dim=768,
        random_bias_gain=1.0,
        # pretrain="bert-base-uncased",
    )
    type_center_random_initializer_base = TypeCenterRandomInitializer(
        training_data,
        torch.cfloat,
        type_dim=768,
        pretrain="bert-base-uncased",
        random_bias_gain=1.0,
    )

    initializer_list = [
        "uniform",
        "normal",
        "orthogonal_",
        "constant_",
        "ones_",
        "zeros_",
        "eye_",
        "sparse_",
        "xavier_uniform_",
        "xavier_uniform_norm_",
        "xavier_normal_",
        "xavier_normal_norm_",
        "uniform_norm_",
        "uniform_norm_p1_",
        "normal_norm_",
    ]

    initializer_dict = dict()

    for initializer in initializer_list:
        initializer_dict[initializer] = initializer

    initializer_dict["type_center_initializer_base"] = type_center_initializer_base
    initializer_dict["type_center_initializer_random"] = type_center_initializer_random
    initializer_dict[
        "type_center_random_initializer_base_gain_1.0"
    ] = type_center_random_initializer_base
    initializer_dict[
        "type_center_random_initializer_random_gain_1.0"
    ] = type_center_random_initializer_random

    from pykeen.pipeline import pipeline

    for i, (name, entity_initializer) in enumerate(initializer_dict.items()):
        print(
            f"experiment {i} / {len(initializer_dict.items())} : {name} .............."
        )
        model_kwargs = dict(
            embedding_dim=embedding_dim,
            # relation_initializer="init_phases",
            # relation_constrainer="complex_normalize",
            entity_initializer=entity_initializer,
            # entity_constrainer=None,
        )
        date_time = "/%s/%s/%s/%s" % (
            "yago_new_init",
            f"{name}",
            "rotate",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        try:
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
            pipeline_result.save_to_directory(model_path)
        except Exception as e:
            print(f"experiment {i}: {str(entity_initializer)} failed")
            print(e)
