"""
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-05-23 14:24:50
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-08-01 11:06:56
FilePath: /code/test.py
Description: 测试1-1，1-n，n-1，n-n的结果。测试不同种类关系的结果。

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
import os
from collections import defaultdict

import pandas as pd
import pykeen
import torch
from Custom.ir_evaluation import IRRankBasedEvaluator
from pykeen.datasets import FB15k237
from pykeen.datasets import analysis as ana
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from utilities import load_dataset


def get_relation_cardinality_dict(dataset: pykeen.datasets):
    relation_cardinality_df = ana.get_relation_cardinality_types_df(dataset=dataset)
    groups = relation_cardinality_df.groupby(relation_cardinality_df.relation_id)
    relation_cardinality_dict = {
        "one-to-many": [],
        "one-to-one": [],
        "many-to-many": [],
        "many-to-one": [],
    }
    for i in range(len(groups)):
        """
        where there are many2many, this realtion is many2many
        if no many2many, relation type is many2one or one2many or one2one.
        """
        relation_instance = groups.get_group(i)

        relation_cardinality_type = "many-to-many"
        if relation_instance.shape[0] < 4:
            if "many-to-many" in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = "many-to-many"
            elif "many-to-one" in list(
                relation_instance.iloc[:, 1]
            ) and "many-to-many" not in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = "many-to-one"
            elif "one-to-many" in list(
                relation_instance.iloc[:, 1]
            ) and "many-to-many" not in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = "one-to-many"
            elif (
                "many-to-many" not in list(relation_instance.iloc[:, 1])
                and "many-to-one" not in list(relation_instance.iloc[:, 1])
                and "one-to-many" not in list(relation_instance.iloc[:, 1])
            ):
                relation_cardinality_type = "one-to-one"

        relation_cardinality_dict[relation_cardinality_type].append(i)

    return relation_cardinality_dict


def relation_cardinality_type_result(
    model, dataset, evaluator, relation_set, cardinality_type="one-to-one"
):
    test_data = dataset.testing.new_with_restriction(
        relations=relation_set[cardinality_type]
    )

    results = evaluator.evaluate(
        batch_size=8,
        model=model,
        mapped_triples=test_data.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
            dataset.testing.mapped_triples,
        ],
    )
    results = results.to_dict()
    result_dic = defaultdict(dict)
    for ht in ["head", "tail"]:
        r = results[ht]["realistic"]
        result_dic[ht]["mrr"] = r["inverse_harmonic_mean_rank"]
        result_dic[ht]["h@10"] = r["hits_at_10"]
        result_dic[ht]["h@3"] = r["hits_at_3"]
        result_dic[ht]["h@1"] = r["hits_at_1"]

    return result_dic


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


def test_examples(model, examples):
    """
    description: 测试某些例子的得分
    param model:训练后的模型
    param examples: 第一个为事实元组。都应该转化为索引的形式, 应该是batch的形式
    return {*}
    """
    model.eval()

    assert examples.dtype == torch.long

    scores = model.score_hrt(example_index)

    if len(scores) == 3:
        return scores[0]
    else:
        return scores


def get_result_dict(model, evaluator, relation_set, dataset, ir_evaluator=None):
    result_dict = defaultdict(list)

    for m in ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]:
        # print(relation_cardinality_type_result(model, dataset=dataset, evaluator=evaluator,relation_set=relation_set, head_or_tail=ht, cardinality_type=m))
        result = relation_cardinality_type_result(
            model,
            dataset=dataset,
            evaluator=evaluator,
            relation_set=relation_set,
            cardinality_type=m,
        )
        for ht in result:
            result_dict["target"].append(ht)
            result_dict["cardinality_type"].append(m)
            result_dict["mrr"].append(result[ht]["mrr"])
            result_dict["h@10"].append(result[ht]["h@10"])
            result_dict["h@3"].append(result[ht]["h@3"])
            result_dict["h@1"].append(result[ht]["h@1"])
        if ir_evaluator:
            result = relation_cardinality_type_result(
                model,
                dataset=dataset,
                evaluator=ir_evaluator,
                relation_set=relation_set,
                cardinality_type=m,
            )
            for ht in result:
                result_dict["ir_mrr"].append(result[ht]["mrr"])
                result_dict["ir_h@10"].append(result[ht]["h@10"])
                result_dict["ir_h@3"].append(result[ht]["h@3"])
                result_dict["ir_h@1"].append(result[ht]["h@1"])

    return result_dict


if __name__ == "__main__":
    # dataset_name = 'CAKE-FB15K237'
    # dataset_name = "CAKE-NELL-995"
    dataset_name = "yago5k-106"
    description = "scStrongConstraintPreTrainTypeEmbUsingSoftMaxWeightMask"
    model_name = "NNYwithRotatE"
    model_date = "20230910-013742"

    if "TypeAsTrain" in description:
        type_as_train = True
    else:
        type_as_train = False

    training_data, validation, testing = load_dataset(
        dataset=dataset_name, ifTypeAsTrain=type_as_train
    )

    # example_str_yago = "Henry_Mancini | diedIn | Los_Angeles | \
    # Om_Puri | diedIn | Los_Angeles |\
    # Orson_Welles | diedIn | Los_Angeles |\
    # Iraq | diedIn | Los_Angeles |\
    # | Henry_Mancini | diedIn | San_Francisco |\
    # | Henry_Mancini | diedIn | Joe_Mantegna |"

    examples_yago = [
        ("Henry_Mancini", "diedIn", "Los_Angeles"),
        ("Om_Puri", "diedIn", "Los_Angeles"),
        ("Orson_Welles", "diedIn", "Los_Angeles"),
        ("Iraq", "diedIn", "Los_Angeles"),
        ("Henry_Mancini", "diedIn", "San_Francisco"),
        ("Henry_Mancini", "diedIn", "Joe_Mantegna"),
    ]
    example_index = list()
    for example in examples_yago:
        example_index.append(
            (
                training_data.entity_to_id[example[0]],
                training_data.relation_to_id[example[1]],
                training_data.entity_to_id[example[2]],
            )
        )
    example_index = torch.tensor(example_index).long()

    print(
        training_data.num_entities, training_data.num_relations, training_data.num_types
    )
    print(
        "train:",
        training_data.num_triples,
        "validation:",
        validation.num_triples,
        "test:",
        testing.num_triples,
    )

    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    relation_set = get_relation_cardinality_dict(dataset=dataset)
    print(dataset_name)
    for m in ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]:
        test_data = dataset.testing.new_with_restriction(relations=relation_set[m])
        print(m, ":", test_data.mapped_triples.shape[0])

    trained_model = load_model(dataset_name, description, model_date, model_name)
    # 测试特例的得分
    trained_model.strong_constraint = False
    scores = test_examples(model=trained_model, examples=example_index)
    scores = list(scores.tolist())
    scores_str = str(round(scores[0][0], 3)) + " & "
    for score in scores[1:-1]:
        scores_str += str(round(score[0], 3)) + " & "
    scores_str += str(round(scores[-1][0], 3)) + " \\"
    print(scores_str)
    for pair in zip(examples_yago, scores):
        print(pair[0], ": ", round(pair[1][0], 3))
    breakpoint()

    # 统计每种类别的关系的数据量。

    evaluator = RankBasedEvaluator()
    ir_evaluator = IRRankBasedEvaluator()
    r = evaluator.evaluate(
        model=trained_model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
            dataset.testing.mapped_triples,
        ],
        batch_size=2,
    )

    print(r.to_dict()["both"]["realistic"]["inverse_harmonic_mean_rank"])

    # result_dic = get_result_dict(
    #     trained_model, evaluator, relation_set, dataset, ir_evaluator=ir_evaluator
    # )
    # result_df = pd.DataFrame(result_dic)

    # save_path = "../result/cardinality/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # result_df.to_csv(
    #     save_path
    #     + "%s.csv"
    #     % (
    #         dataset_name.replace("-", "")
    #         + "-"
    #         + description
    #         + "-"
    #         + model_name
    #         + "-"
    #         + model_date
    #     ),
    #     index=False,
    # )
