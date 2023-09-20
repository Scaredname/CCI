import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator
from utilities import load_dataset


def load_model(
    default_dataset_name=None,
    default_description=None,
    default_model_date=None,
    default_model_name=None,
):
    # Load the model
    if (
        default_model_name
        and default_model_date
        and default_dataset_name
        and default_description
    ):
        print("using pre-set model")

    if not default_dataset_name:
        dataset_name = input("Please input the name of the dataset:")
    else:
        dataset_name = default_dataset_name
    print("The dataset name is: ", dataset_name)

    if not default_description:
        description = input("Please input the description of this model:")
    else:
        description = default_description
    print("The description of this model is: ", description)

    if not default_model_name:
        model_name = input("Please input the name of this model:")
    else:
        model_name = default_model_name
    print("The name of this model is: ", model_name)

    if not default_model_date:
        model_date = input("Please input the date of this model:")
    else:
        model_date = default_model_date
    print("The date of this model is: ", model_date)

    model_path = (
        "../models/"
        + dataset_name
        + "/"
        + description
        + "/"
        + model_name
        + "/"
        + model_date
        + "/"
        + "trained_model.pkl"
    )
    trained_model = torch.load(model_path)

    return trained_model


def test_model(
    read_path, save_path, evaluator=RankBasedEvaluator(), test_batch_size=16
):
    dataset, description, _, _ = read_path.split("/")[-4:]

    if "TypeAsTrain" in description:
        type_as_train = True
    else:
        type_as_train = False

    training_data, validation, testing = load_dataset(
        dataset=dataset, ifTypeAsTrain=type_as_train
    )

    trained_model = torch.load(os.path.join(read_path, "trained_model.pkl"))

    trained_model.strong_constraint = True

    valid_results = evaluator.evaluate(
        model=trained_model,
        mapped_triples=validation.mapped_triples,
        additional_filter_triples=[
            training_data.mapped_triples,
            validation.mapped_triples,
            testing.mapped_triples,
        ],
        batch_size=test_batch_size,
    )

    results = evaluator.evaluate(
        model=trained_model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=[
            training_data.mapped_triples,
            validation.mapped_triples,
            testing.mapped_triples,
        ],
        batch_size=test_batch_size,
    )

    results_dict = dict(metrics=results.to_dict())
    results_dict["stopper"] = dict(
        best_metric=valid_results.to_dict()["both"]["realistic"][
            "inverse_harmonic_mean_rank"
        ]
    )
    with open(os.path.join(save_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)


def walk_folder_tree(path, depth=0):
    """
    递归遍历所有的目录
    """
    results_paths = list()
    if os.path.isdir(path):
        for folder_name in os.listdir(path):
            next_folder_path = os.path.join(path, folder_name)
            results_paths.extend(walk_folder_tree(next_folder_path, depth=depth + 1))
            if depth == 3:
                results_paths.append(next_folder_path)
    return results_paths


if __name__ == "__main__":
    all_result_path = "../models/"

    # for dataset in os.listdir(all_result_path):
    #     file_path = os.path.join(all_result_path, dataset)

    #     if os.path.isdir(file_path):
    #         for description in os.listdir(file_path):
    #             file_path1 = os.path.join(file_path, description)

    #             if os.path.isdir(file_path1):
    #                 for model in os.listdir(file_path1):
    #                     file_path2 = os.path.join(file_path1, model)
    #                     if os.path.isdir(file_path2):
    #                         for date in os.listdir(file_path2):
    #                             final_path = os.path.join(file_path2, date)

    # print(date)

    results_paths_list = walk_folder_tree(all_result_path)
    for path in results_paths_list:
        try:
            dataset, description, model, date = path.split("/")[-4:]
            prefix = path.split("/")[:-4]
        except:
            print(path)
        if "NNY" in model or "MM" in model:
            if int(date.split("-")[0]) > 20230908:
                if "StrongConstraint" in description:
                    nosc_de = description.replace("StrongConstraint", "")
                    nosc_path = os.path.join(
                        "/".join(prefix), dataset, nosc_de, model, date
                    )

                    if not os.path.exists(nosc_path) or (
                        len(os.listdir(nosc_path)) <= 1
                    ):
                        os.makedirs(nosc_path, exist_ok=True)
                        test_model(
                            read_path=path, save_path=nosc_path, test_batch_size=2
                        )
                        shutil.copy(
                            os.path.join(path, "config.json"),
                            nosc_path,
                        )

                else:
                    split_word = re.findall(r"[A-Z][^A-Z]*", description)
                    if len(split_word):
                        sc_de = description.replace(
                            split_word[0], "StrongConstraint" + split_word[0], 1
                        )
                    else:
                        sc_de = description + "StrongConstraint"
                    sc_path = os.path.join(
                        "/".join(prefix), dataset, sc_de, model, date
                    )

                    if not os.path.exists(sc_path) or (len(os.listdir(sc_path)) <= 1):
                        os.makedirs(sc_path, exist_ok=True)
                        test_model(read_path=path, save_path=sc_path, test_batch_size=2)
                        shutil.copy(
                            os.path.join(path, "config.json"),
                            sc_path,
                        )
