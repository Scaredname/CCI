import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

pattern = r"\(([^)]+)\)"
pattern1 = r"\(_embeddings\): Embedding\((.*)\)"

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
    help='a="yago_new_init", b="CAKE-NELL-995_new_init", c="CAKE-DBpedia-242_new_init", d="Kinships_init"',
)
args = parser.parse_args()

dataset_dict = dict(
    a="yago_new_init",
    b="CAKE-NELL-995_new_init",
    c="CAKE-DBpedia-242_new_init",
    d="Kinships_init",
)
dataset = dataset_dict[args.dataset]
# dataset = "CAKE-NELL-995_new_init"
result_path = "../models/%s/" % (dataset)
# dataset = args.dataset.replace("-", "")
save_path = "../result/%s.csv" % ("speed_" + dataset)

results_dict = defaultdict(list)


def determine_convergence_epoch(stopper_data, threshold_value):
    """
      stoppeer_data:
      "stopper": {
      "best_epoch": 10,
      "best_metric": 0.18443584442138672,
      "frequency": 5,
      "larger_is_better": true,
      "metric": "mean_reciprocal_rank",
      "patience": 6,
      "relative_delta": 0.0001,
      "remaining_patience": 0,
      "results": [
        0.16317647695541382,
        0.18443584442138672,
        0.18252341449260712,
        0.16714170575141907,
        0.16239221394062042,
        0.14846134185791016,
        0.14068426191806793,
        0.13593825697898865
      ],
      "stopped": true
    },
    """
    evaluation_frequency = stopper_data["frequency"]
    results_log = stopper_data["results"]

    # 如果不收敛直接使用best_epoch 和对应的测试结果
    convergence_epoch = stopper_data["best_epoch"]
    convergence_valid = round(float(results_log[-1]), 3)
    retain = False

    for i in range(len(results_log) - 1):
        if results_log[i] >= threshold_value:
            convergence_epoch = (i + 1) * evaluation_frequency
            convergence_valid = round(float(results_log[i]), 3)
            retain = True
            break

    return convergence_epoch, convergence_valid, retain


# 取每个模型的最小的验证集上的结果, mean-std的结果
threshold_dict = {
    "yago_new_init": dict(distmult=0.248, complex=0.288, RotatE=0.291, TransE=0.203),
    "CAKE-NELL-995_new_init": dict(
        distmult=0.376, complex=0.388, RotatE=0.374, TransE=0.296
    ),
}

threshold_dict = threshold_dict[dataset]

# dataset/de/model/date/results.json
for file_name in os.listdir(result_path):
    file_path = os.path.join(result_path, file_name)

    if os.path.isdir(file_path):
        for f in os.listdir(file_path):
            de_path = os.path.join(file_path, f)

            if os.path.isdir(de_path):
                model_name = f
                for ff in os.listdir(de_path):
                    date = ff
                    file_path1 = os.path.join(de_path, date)
                    if "-" in date:
                        result_path1 = os.path.join(file_path1, "results.json")
                        config_path = os.path.join(file_path1, "metadata.json")
                        with open(result_path1, "r", encoding="utf8") as fff:
                            results = json.load(fff)
                        with open(config_path, "r", encoding="utf8") as fff:
                            config = json.load(fff)

                    try:
                        threshold_value = threshold_dict[config["model"]] * 0.9
                    except:
                        threshold_value = 0
                    if "stopper" in results:
                        epoch, mrr, retain = determine_convergence_epoch(
                            results["stopper"], threshold_value
                        )
                        if not retain:
                            continue
                        results_dict["date"].append(date)
                        results_dict["initializer_name"].append(file_name)
                        results_dict["threshold_value"].append(
                            round(float(threshold_value), 3)
                        )
                        results_dict["conver-epoch"].append(epoch)
                        results_dict["conver-valid"].append(mrr)

                        if "model" in config:
                            results_dict["model"].append(config["model"])
                        else:
                            results_dict["model"].append(model_name)
                        if "loss" in config:
                            results_dict["loss"].append(config["loss"])
                        else:
                            results_dict["loss"].append("-")
                        if "optimizer_kwargs" in config:
                            results_dict["lr"].append(config["optimizer_kwargs"]["lr"])
                        else:
                            results_dict["lr"].append("-")
r = pd.DataFrame(results_dict)

r.to_csv(save_path, index=False)
