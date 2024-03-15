"""
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2022-12-22 12:02:34
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-08-20 15:28:45
FilePath: /ESETC/code/utilities.py
Description: 

Copyright (c) 2023 by Ni Runyu ni-runyu@ed.tmu.ac.jp, All Rights Reserved. 
"""

import datetime
import os

import torch
from Customize.custom_triple_factory import TripleswithCategory
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def get_key(dict, va):
    return [k for k, v in dict.item() if v == va]


def split_type_data(data: TriplesFactory):
    relations = list(data.relation_to_id.keys())
    relations.remove("type")

    return (
        data.label_triples(
            data.new_with_restriction(relations=relations).mapped_triples
        ),
        data.label_triples(
            data.new_with_restriction(relations=["type"]).mapped_triples
        ),
    )


def read_data(
    data_name,
    data_pro_func=split_type_data,
    create_inverse_triples=False,
):
    """
    @Params: data_name, data_pro_func, create_inverse_triples, type_position
    @Return: Train, Test, Valid
    """
    data_path = os.path.join(os.getcwd(), "../data/")

    train_path = os.path.join(data_path, "%s/" % (data_name), "train_cate.txt")
    valid_path = os.path.join(data_path, "%s/" % (data_name), "valid.txt")
    test_path = os.path.join(data_path, "%s/" % (data_name), "test.txt")

    training = TriplesFactory.from_path(
        train_path,
        create_inverse_triples=create_inverse_triples,
    )

    (
        training_triples,
        category_triples,
    ) = data_pro_func(training)
    training_data = TripleswithCategory.from_labeled_triples(
        triples=training_triples,
        cate_triples=category_triples,
    )

    validation = TriplesFactory.from_path(
        valid_path,
        entity_to_id=training_data.entity_to_id,
        relation_to_id=training_data.relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )
    testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=training_data.entity_to_id,
        relation_to_id=training_data.relation_to_id,
        create_inverse_triples=create_inverse_triples,
    )

    return training_data, validation, testing


def train_model(
    entity_initializer,
    name,
    dataset,
    dataset_name,
    fix_config,
    embedding_dim,
    lr_list,
    no_constrainer=False,
    relation_initializer=None,
):
    """
    description: test initialization
    param entity_initializer:
    param name: the name of initializer
    param dataset:
    param dataset_name:
    param fix_config:
    param embedding_dim:
    param lr_list:
    param no_constrainer: some models don't have constrainer
    return {*}
    """
    lr_lists = lr_list

    model_kwargs = dict(
        embedding_dim=embedding_dim,
        entity_initializer=entity_initializer,
        entity_constrainer=None,
    )
    if no_constrainer:
        model_kwargs.pop("entity_constrainer")
    if relation_initializer:
        model_kwargs["relation_initializer"] = relation_initializer

    try:
        for learning_rate in lr_lists:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print("lr:", learning_rate)
            print("init:", name)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
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
                    experiment_path="../result/hpo_init/" + date_time,
                ),
                **fix_config,
            )

            model_path = "../models/" + date_time
            pipeline_result.metadata = fix_config
            pipeline_result.save_to_directory(model_path)
    except Exception as e:
        print(f"experiment: {str(entity_initializer)} failed")
        print(e)
