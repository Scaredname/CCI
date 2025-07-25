import argparse
import json
import os
import re
from collections import defaultdict

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
    ],
    default="a",
    type=str,
    help='a="yago6k_103", b="NELL995", c="FB15k237"',
)
args = parser.parse_args()

dataset_dict = dict(
    a="yago_new_init",
    b="NELL-995_new_init",
    c="FB_new_init",
)
dataset = dataset_dict[args.dataset]
# dataset = "CAKE-NELL-995_new_init"
result_path = "../models/%s/" % (dataset)
# dataset = args.dataset.replace("-", "")
save_path = "../result/%s.csv" % ("horizontal_" + dataset)

results_dict = defaultdict(list)

print("Note!!! Please determine the threshold value for each model in each dataset.")


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


# 取每个模型的最优的验证集结果, mean+std的结果。
threshold_dict = {
    "yago_new_init": dict(distmult=0.259, complex=0.298, RotatE=0.291, TransE=0.206),
    "NELL-995_new_init": dict(
        distmult=0.385, complex=0.391, RotatE=0.380, TransE=0.307
    ),
    "FB_new_init": dict(distmult=0.287, complex=0.301, RotatE=0.305, TransE=0.292),
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
                        threshold_value = threshold_dict[config["model"]] * 0.9 # 90% of the best validation result
                    except:
                        threshold_value = 0
                    if "stopper" in results:
                        epoch, mrr, retain = determine_convergence_epoch(
                            results["stopper"], threshold_value
                        )
                        if not retain:
                            results_dict["retain"].append(0)
                        else:
                            results_dict["retain"].append(1)
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

                    if "description" in config:
                        results_dict["description"].append(config["description"])
r = pd.DataFrame(results_dict)
r = r.sort_values(by=["date"], ascending=False)

r.to_csv(save_path, index=False)
