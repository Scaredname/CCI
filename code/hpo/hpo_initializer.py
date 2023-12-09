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

# dataset_name = "yago_new"
dataset_name = "CAKE-NELL-995_new"

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

    # lr_lists = [0.1, 0.01]
    # def init_train_model(entity_initializer, name, dataset_name, fix_config):
    #     lr_lists = [0.0001]
    #     model_kwargs = dict(
    #         embedding_dim=embedding_dim,
    #         # relation_initializer="init_phases",
    #         # relation_constrainer="complex_normalize",
    #         entity_initializer=entity_initializer,
    #         entity_constrainer=None,
    #     )
    #     try:
    #         for learning_rate in lr_lists:
    #             fix_config["optimizer_kwargs"]["lr"] = learning_rate
    #             date_time = "/%s/%s/%s/%s" % (
    #                 f"{dataset_name}_init",
    #                 f"{name}_gain",
    #                 fix_config["model"],
    #                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    #             )
    #             pipeline_result = pipeline(
    #                 dataset=dataset,
    #                 model_kwargs=model_kwargs,
    #                 device="cuda",
    #                 result_tracker="tensorboard",
    #                 result_tracker_kwargs=dict(
    #                     experiment_path="../../result/hpo_init/" + date_time,
    #                 ),
    #                 **fix_config,
    #             )
    #             model_path = "../../models/" + date_time
    #             pipeline_result.metadata = fix_config
    #             pipeline_result.save_to_directory(model_path)
    #     except:
    #         print(f"experiment: {str(entity_initializer)} failed")
    # initializer_list = [
    #     "uniform",
    #     "normal",
    #     "orthogonal_",
    #     "constant_",
    #     "ones_",
    #     "zeros_",
    #     "eye_",
    #     "sparse_",
    #     "xavier_uniform_",
    #     "xavier_uniform_norm_",
    #     "xavier_normal_",
    #     "xavier_normal_norm_",
    #     "uniform_norm_",
    #     "uniform_norm_p1_",
    #     "normal_norm_",
    # ]

    initializer_list = [
        "uniform_norm_",
        "xavier_uniform_norm_",
        "xavier_normal_norm_",
    ]
    lr_list = [0.0001]
    for initializer in initializer_list:
        init_train_model(
            initializer,
            initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )  # baseline

        random_initializer_50 = TypeCenterRandomInitializer(
            training_data,
            torch.float,
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
        )

        # bert_initializer_50 = TypeCenterRandomInitializer(
        #     training_data,
        #     torch.float,
        #     type_dim=768,
        #     pretrain="bert-base-uncased",
        #     random_bias_gain=50,
        #     type_init=initializer,
        # )

        # initializer_dict[
        #     "bert_initializer_50_" + initializer
        # ] = bert_initializer_50

        random_initializer_10 = TypeCenterRandomInitializer(
            training_data,
            torch.float,
            type_dim=768,
            random_bias_gain=10,
            type_init=initializer,
        )

        init_train_model(
            random_initializer_10,
            "random_initializer_10_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )

        # bert_initializer_10 = TypeCenterRandomInitializer(
        #     training_data,
        #     torch.float,
        #     type_dim=768,
        #     pretrain="bert-base-uncased",
        #     random_bias_gain=10,
        #     type_init=initializer,
        # )

        # initializer_dict["bert_initializer_10_" + initializer] = bert_initializer_10

        random_initializer_100 = TypeCenterRandomInitializer(
            training_data,
            torch.float,
            type_dim=768,
            random_bias_gain=100,
            type_init=initializer,
        )

        init_train_model(
            random_initializer_100,
            "random_initializer_100_" + initializer,
            dataset,
            dataset_name,
            fix_config,
            embedding_dim,
            lr_list,
        )

        # bert_initializer_100 = TypeCenterRandomInitializer(
        #     training_data,
        #     torch.float,
        #     type_dim=768,
        #     pretrain="bert-base-uncased",
        #     random_bias_gain=100,
        #     type_init=initializer,
        # )

        # initializer_dict[
        #     "bert_initializer_100_" + initializer
        # ] = bert_initializer_100

        random_initializer_1 = TypeCenterRandomInitializer(
            training_data,
            torch.float,
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
        )

        random_product_initializer_1 = TypeCenterProductRandomInitializer(
            training_data,
            torch.float,
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
        )
