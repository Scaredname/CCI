import sys

sys.path.append("..")

# from Custom import triplefactory_resolver

# a = triplefactory_resolver.make("Test")

# print(a)


from pykeen.ablation import ablation_pipeline
from pykeen.datasets import get_dataset
from utilities import load_dataset

if __name__ == "__main__":
    results_path = "../result/ablation/"
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

    training_data, validation, testing = load_dataset(
        dataset="yago_new",
    )
    dataset = get_dataset(
        training=training_data, testing=testing, validation=validation
    )

    model = "Rotate"
    embedding_dim = 768
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
        lr_scheduler_kwargs=dict(step_size=500),
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
    )

    ablation_config = dict()
