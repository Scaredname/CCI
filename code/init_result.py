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
    ],
    default="a",
    type=str,
    help='a="yago_new_init", b="CAKE-NELL-995_new_init", c="CAKE-DBpedia-242_new_init"',
)
args = parser.parse_args()

dataset_dict = dict(
    a="yago_new_init", b="CAKE-NELL-995_new_init", c="CAKE-DBpedia-242_new_init"
)
dataset = dataset_dict[args.dataset]
# dataset = "CAKE-NELL-995_new_init"
result_path = "../models/%s/" % (dataset)
# dataset = args.dataset.replace("-", "")
save_path = "../result/%s.csv" % (dataset)

results_dict = defaultdict(list)


def determine_convergence_epoch(stopper_data):
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

    for i in range(len(results_log) - 1):
        # 增长幅度为1%
        if 1.01 * results_log[i] > results_log[i + 1]:
            if results_log[i] > 0.9 * stopper_data["best_metric"]:
                convergence_epoch = (i + 1) * 5
                convergence_valid = round(float(results_log[i]), 3)
                break

    return convergence_epoch, convergence_valid


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
                        results_dict["date"].append(date)

                        result_path1 = os.path.join(file_path1, "results.json")
                        config_path = os.path.join(file_path1, "metadata.json")
                        with open(result_path1, "r", encoding="utf8") as fff:
                            results = json.load(fff)
                        with open(config_path, "r", encoding="utf8") as fff:
                            config = json.load(fff)

                    results_dict["initializer_name"].append(file_name)
                    if "stopper" in results:
                        results_dict["valid-mrr"].append(
                            round(float(results["stopper"]["best_metric"]), 3)
                        )

                        epoch, mrr = determine_convergence_epoch(results["stopper"])
                        results_dict["conver-epoch"].append(epoch)
                        results_dict["conver-valid"].append(mrr)
                    else:
                        results_dict["valid-mrr"].append(0)
                        results_dict["conver-epoch"].append("-")
                        results_dict["conver-valid"].append("-")

                    results_dict["test-mrr"].append(
                        round(
                            float(
                                results["metrics"]["both"]["realistic"][
                                    "inverse_harmonic_mean_rank"
                                ]
                            ),
                            3,
                        )
                    )
                    if "stopper" in results:
                        results_dict["best_epoch"].append(
                            results["stopper"]["best_epoch"]
                        )
                    else:
                        results_dict["best_epoch"].append("-")

                    results_dict["hits@1"].append(
                        round(
                            float(results["metrics"]["both"]["realistic"]["hits_at_1"]),
                            3,
                        )
                    )
                    results_dict["hits@3"].append(
                        round(
                            float(results["metrics"]["both"]["realistic"]["hits_at_3"]),
                            3,
                        )
                    )
                    results_dict["hits@10"].append(
                        round(
                            float(
                                results["metrics"]["both"]["realistic"]["hits_at_10"]
                            ),
                            3,
                        )
                    )

                    if "optimizer_kwargs" in config:
                        results_dict["lr"].append(config["optimizer_kwargs"]["lr"])
                    else:
                        results_dict["lr"].append("-")
                    if "model" in config:
                        results_dict["model"].append(config["model"])
                    else:
                        results_dict["model"].append(model_name)
r = pd.DataFrame(results_dict)

r.to_csv(save_path, index=False)
