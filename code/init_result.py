import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd

pattern = r"\(([^)]+)\)"
pattern1 = r"\(_embeddings\): Embedding\((.*)\)"

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-d",
#     "--dataset",
#     choices=[
#         "YAGO3-10-TypeLike",
#         "YAGO3-10",
#         "CAKE-FB15K237",
#         "CAKE-FB15K",
#         "CAKE-NELL-995",
#         "CAKE-DBpedia-242",
#         "fb15k-237-type",
#         "yago5k-106",
#         "CAKE-NELL-995_new",
#         "CAKE-DBpedia_hpo",
#         "CAKE-DBpedia-242_new",
#         "yago_new",
#     ],
#     default="YAGO3-10-TypeLike",
# )
# args = parser.parse_args()


# dataset = "yago_new_init"
dataset = "CAKE-NELL-995_new_init"
result_path = "../models/%s/" % (dataset)
# dataset = args.dataset.replace("-", "")
save_path = "../result/%s.csv" % (dataset)

results_dict = defaultdict(list)

# dataset/de/model/date/results.json
for file_name in os.listdir(result_path):
    file_path = os.path.join(result_path, file_name)

    if os.path.isdir(file_path):
        for f in os.listdir(file_path):
            de_path = os.path.join(file_path, f)

            if os.path.isdir(de_path):
                model_name = f
                for ff in os.listdir(de_path):
                    results_dict["model"].append(model_name)

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

                    if "stopper" in results:
                        results_dict["valid-mrr"].append(
                            round(float(results["stopper"]["best_metric"]), 3)
                        )
                    else:
                        results_dict["valid-mrr"].append(0)

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
                    if "stopper" in results:
                        results_dict["best_epoch"].append(
                            results["stopper"]["best_epoch"]
                        )
                    else:
                        results_dict["best_epoch"].append("-")
                    if "optimizer_kwargs" in config:
                        results_dict["lr"].append(config["optimizer_kwargs"]["lr"])
                    else:
                        results_dict["lr"].append("-")
                    results_dict["initializer_name"].append(file_name)
r = pd.DataFrame(results_dict)

r.to_csv(save_path, index=False)
