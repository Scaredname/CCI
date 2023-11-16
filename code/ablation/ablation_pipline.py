import sys

sys.path.append("..")

import os

from pykeen.ablation import ablation_pipeline
from pykeen.datasets import get_dataset
from utilities import load_dataset

# from Custom import triplefactory_resolver

# a = triplefactory_resolver.make("Test")

# print(a)


if __name__ == "__main__":
    # choices=[
    #         "fb15k-237-type",
    #         "YAGO3-10",
    #         "FB15k237",
    #         "CAKE-FB15K237",
    #         "CAKE-FB15K",
    #         "CAKE-NELL-995",
    #         "CAKE-DBpedia-242",
    #         "yago5k-106",
    #         "CAKE-NELL-995_new",
    #         "CAKE-DBpedia-242_new",
    #         "yago_new",
    #     ],

    dataset = "yago_new"

    results_path = os.path.join("../result/ablation/", dataset)

    test_batch_size = 16

    # training_data, validation, testing = load_dataset(
    #     dataset=dataset,
    # )
    # yago_dataset = get_dataset(
    #     training=training_data, testing=testing, validation=validation
    # )
    home_path = os.path.expanduser("~")
    yago_dataset_path = dict(
        training=f"{home_path}/code/ESETC/data/yago_new/train.txt",
        validation=f"{home_path}/code/ESETC/data/yago_new/valid.txt",
        testing=f"{home_path}/code/ESETC/data/yago_new/test.txt",
    )

    ablation_config = dict(
        datasets=[yago_dataset_path],
        models=["Distmult", "Complex"],
        model_to_model_kwargs_ranges=dict(
            Distmult=dict(
                embedding_dim=dict(type="int", scale="power_two", low=6, high=9),
            ),
            Complex=dict(
                embedding_dim=dict(type="int", scale="power_two", low=5, high=8),
            ),
        ),
        losses=["BCEAfterSigmoidLoss", "CrossEntropyLoss", "NSSALoss"],
        model_to_loss_to_loss_kwargs_ranges=dict(
            Complex=dict(
                NSSALoss=dict(
                    margin=dict(type="int", low=6, high=10),
                    adversarial_temperature=dict(
                        type="float", low=0.5, high=1.5, q=0.5
                    ),
                )
            ),
            Distmult=dict(
                NSSALoss=dict(
                    margin=dict(type="int", low=6, high=10),
                    adversarial_temperature=dict(
                        type="float", low=0.5, high=1.5, q=0.5
                    ),
                )
            ),
        ),
        training_loops=["lcwa", "slcwa"],
        model_to_training_loop_to_training_kwargs=dict(
            Complex=dict(
                lcwa=dict(num_epochs=20, batch_size=512),
                slcwa=dict(num_epochs=200, batch_size=512),
            ),
            Distmult=dict(
                lcwa=dict(num_epochs=20, batch_size=512),
                slcwa=dict(num_epochs=200, batch_size=512),
            ),
        ),
        optimizers=["adam"],
        model_to_optimizer_to_optimizer_kwargs_ranges=dict(
            Complex=dict(
                adam=dict(
                    lr=dict(
                        type="categorical", choices=[0.1, 0.01, 0.001, 0.0001, 0.00001]
                    )
                )
            ),
            Distmult=dict(
                adam=dict(
                    lr=dict(
                        type="categorical", choices=[0.1, 0.01, 0.001, 0.0001, 0.00001]
                    )
                )
            ),
        ),
        create_inverse_triples=[True, False],
        regularizers=["LpRegularizer", None],
        model_to_regularizer_to_regularizer_kwargs_ranges=dict(
            Complex=dict(
                LpRegularizer=dict(
                    p=dict(type="categorical", choices=[2.0, 3.0]),
                    weight=dict(type="categorical", choices=[0.1, 0.5, 1.0]),
                )
            ),
            Distmult=dict(
                LpRegularizer=dict(
                    p=dict(type="categorical", choices=[2.0, 3.0]),
                    weight=dict(type="categorical", choices=[0.1, 0.5, 1.0]),
                )
            ),
        ),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=10,
            relative_delta=0.0001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=test_batch_size,
        ),
    )

    ablation_result = ablation_pipeline(
        directory=results_path,
        evaluator_kwargs=dict(filtered=True, batch_size=test_batch_size),
        best_replicates=3,
        n_trials=50,
        metric="mean_reciprocal_rank",
        direction="maximize",
        sampler="TPESampler",
        pruner="nop",
        **ablation_config,
    )
