# Get a training dataset
import argparse
import datetime
import json
import os

from Custom.CustomTrain import TypeSLCWATrainingLoop
from pykeen.constants import PYKEEN_CHECKPOINTS
from utilities import load_dataset

parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model_index", type=int, default=0)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    choices=[
        "fb15k-237-type",
        "YAGO3-10",
        "FB15k237",
        "CAKE-FB15K237",
        "CAKE-FB15K",
        "CAKE-NELL-995",
        "CAKE-DBpedia-242",
        "yago5k-106",
        "CAKE-NELL-995_new",
        "CAKE-DBpedia-242_new",
        "yago_new",
    ],
    default="fb15k-237-type",
)
# parser.add_argument("-o", "--optimizer", type=str, default="adam")
parser.add_argument("-cpu", "--num_workers", type=int, default=1)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-lm", "--loss_margin", type=float, default=9.0)
parser.add_argument("-at", "--adversarial_temperature", type=float, default=1.0)
parser.add_argument("-twt", "--type_weight_temperature", type=float, default=1.0)
parser.add_argument("-tsw", "--type_score_weight", type=float, default=1.0)

# HAKE 才用 -------------
# parser.add_argument("-mw", "--modulus_weight", type=float, default=1.0)
# parser.add_argument("-pw", "--phase_weight", type=float, default=1.0)
# -----------------

parser.add_argument("-b", "--batch_size", type=int, default=256)
# parser.add_argument("-e", "--epochs", type=int, default=1000)
# parser.add_argument("-esf", "--early_stop_frequency", type=int, default=50)
# parser.add_argument("-esp", "--early_stop_patience", type=int, default=3)
parser.add_argument("-train", "--training_loop", type=str, default="lcwa")
# parser.add_argument("-neg", "--negative_sampler", type=str, default=None)
parser.add_argument("-nen", "--num_negs_per_pos", type=int, default=None)
parser.add_argument("-ef", "--filtered", type=bool, default=True)
parser.add_argument("-eb", "--evaluator_batch_size", type=int, default=128)
parser.add_argument("-med", "--model_ent_dim", type=int, default=100)
parser.add_argument("-mrd", "--model_rel_dim", type=int, default=100)  # = med+mtd
parser.add_argument("-mtd", "--model_type_dim", type=int, default=100)

## 可以考虑删掉
parser.add_argument("-tsm", "--type_smoothing", type=float, default=0.0)
# ——————————————————
# parser.add_argument("-drop", "--dropout", type=float, default=0.0)
# parser.add_argument("-rw", "--ReglurizerWeight", type=float, default=0.001)
# parser.add_argument("-rp", "--ReglurizerNorm", type=float, default=3.0)


parser.add_argument("-pb", "--project_with_bias", action="store_true", default=False)
parser.add_argument(
    "-nstal", "--not_soft_type_aware_loss", action="store_true", default=False
)
parser.add_argument("-ipo", "--init_preference_one", action="store_true", default=False)
parser.add_argument("-sc", "--strong_constraint", action="store_true", default=False)
parser.add_argument("-let", "--learn_ents_types", action="store_true", default=False)

parser.add_argument("-de", "--description", type=str, default=None)
parser.add_argument(
    "-reverse", "--CreateInverseTriples", action="store_true", default=False
)
parser.add_argument("-t", "--IfUseTypeLike", action="store_true", default=False)
parser.add_argument(
    "-pre",
    "--IfUsePreTrainTypeEmb",
    default=None,
    type=str,
    choices=[None, "bert-base-uncased", "bert-large-uncased"],
    help="If use pre-trained type embeddings, the name of pretrained model",
)
parser.add_argument("-hnt", "--ifHasNoneType", action="store_true", default=False)
parser.add_argument("-tes", "--ifTestEarlyStop", action="store_true", default=False)
parser.add_argument("-tat", "--ifTypeAsTrain", action="store_true", default=False)
parser.add_argument("-fw", "--ifFreezeWeights", action="store_true", default=False)
parser.add_argument(
    "-shp", "--ifSearchHyperParameters", action="store_true", default=False
)
parser.add_argument(
    "-fte",
    "--ifFreezeTypeEmb",
    action="store_true",
    default=False,
    help="If freeze type embeddings, default is False",
)
parser.add_argument(
    "-src",
    "--ifStrictRelationCardinality",
    action="store_true",
    default=False,
    help="if strict relation cardinality, default is False",
)
parser.add_argument(
    "-randw",
    "--ifRandomWeight",
    action="store_true",
    default=False,
    help="If use random weight, default is False",
)
parser.add_argument(
    "-naf",
    "--ifNoActivationFuncion",
    action="store_true",
    default=False,
    help="If use ActivationFuncion for type weight",
)
parser.add_argument(
    "-wm",
    "--ifWeightMask",
    action="store_true",
    default=False,
    help="If use entity types constrains for type weight",
)
parser.add_argument(
    "-ot",
    "--ifOneType",
    action="store_true",
    default=False,
    help="If only use the most related entity type",
)
parser.add_argument(
    "-naet",
    "--ifNotAddEntType",
    help="When get entity's type embedding, whether add entity_type weight to relation_type weight",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-stop", "--stopper", type=str, choices=["early", "nop"], default="early"
)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")


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

    args = parser.parse_args()
    # load data
    dataset = args.dataset
    training_data, validation, testing = load_dataset(
        dataset=dataset,
        IfUseTypeLike=args.IfUseTypeLike,
        CreateInverseTriples=args.CreateInverseTriples,
        type_smoothing=args.type_smoothing,
        ifHasNoneType=args.ifHasNoneType,
        ifTypeAsTrain=args.ifTypeAsTrain,
        use_random_weights=args.ifRandomWeight,
        select_one_type=args.ifOneType,
        strict_confidence=args.ifStrictRelationCardinality,
    )

    # todo: the wired logger of _split_triples
    big_validation, small_validation = validation.split(0.9)

    # setting pipeline

    training_setting = dict(
        num_epochs=5000,
        batch_size=args.batch_size,  # 取决于数据集
        checkpoint_on_failure=True,
        num_workers=1,
    )

    optimizer_kwargs = dict(
        lr=args.learning_rate,  # 手动设置
    )
    negative_sampler_kwargs = dict(
        num_negs_per_pos=args.num_negs_per_pos,
    )

    pipeline_config = dict(
        optimizer="adam",
        training_loop=args.training_loop,
        training_kwargs=training_setting,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(
            filtered=args.filtered,
            batch_size=args.evaluator_batch_size,  #
        ),
        stopper=args.stopper,
        stopper_kwargs=dict(
            frequency=5,
            patience=3,
            relative_delta=0.001,
            metric="mean_reciprocal_rank",
            evaluation_batch_size=args.evaluator_batch_size,
            evaluation_triples_factory=small_validation,
        ),
    )

    # if args.model_index in [41, 42, 51, 52] and args.description == 'final':
    #     args.description = 'STNS-'

    if args.strong_constraint:
        args.description += "StrongConstraint"

    if args.IfUsePreTrainTypeEmb:
        args.description += "PreTrainTypeEmb"

    if args.ifHasNoneType:
        args.description += "HasNoneType"

    if args.type_smoothing:
        args.description += "TypeSmoothing"

    if args.ifTypeAsTrain:
        args.description += "TypeAsTrain"

    if args.ifFreezeWeights:
        args.description += "FreezeWeights"

    if args.ifFreezeTypeEmb:
        args.description += "FreezeTypeEmb"

    if args.ifNotAddEntType:
        args.description += "NotAddEntType"

    if args.ifRandomWeight:
        args.description += "RandomWeight"

    if args.ifWeightMask:
        args.description += "WeightMask"

    if args.ifOneType:
        args.description += "OneType"

    if args.ifStrictRelationCardinality:
        args.description += "StrictRelationCardinality"

    from Custom.CustomLoss import SoftTypeawareNegativeSmapling
    from pykeen.losses import NSSALoss

    # 每次调参前决定使用什么loss和训练方式
    soft_loss = SoftTypeawareNegativeSmapling()
    pipeline_config["training_loop"] = TypeSLCWATrainingLoop
    #     soft_loss = NSSALoss()

    loss_kwargs = dict(
        reduction="mean",
        adversarial_temperature=args.adversarial_temperature,
        margin=args.loss_margin,
    )

    import torch
    from Custom.TypeModels.no_name import NNYwithRotatE

    model = NNYwithRotatE(
        triples_factory=training_data,
    )
    model_kwargs = dict(
        ent_dtype=torch.float,
        rel_dtype=torch.cfloat,
        type_dtype=torch.float,
        ent_dim=args.model_ent_dim,
        rel_dim=args.model_rel_dim // 2,  # relation的数据类型的cfloat
        type_dim=args.model_type_dim,
        freeze_matrix=args.ifFreezeWeights,
        freeze_type_emb=args.ifFreezeTypeEmb,
        add_ent_type=not args.ifNotAddEntType,
        type_initializer="xavier_uniform_",
        entity_initializer="uniform",
        relation_initializer="init_phases",
        relation_constrainer="complex_normalize",
        strong_constraint=args.strong_constraint,
        usepretrained=args.IfUsePreTrainTypeEmb,
        activation_weight=not args.ifNoActivationFuncion,
        weight_mask=args.ifWeightMask,
        type_weight_temperature=args.type_weight_temperature,
        type_score_weight=args.type_score_weight,
        init_preference_one=args.init_preference_one,
        learn_ents_types=args.learn_ents_types,
    )

    from torch.optim import Adam

    optimizer = Adam(params=model.get_grad_params())

    if args.description is not None:
        model_name = args.description + "/" + type(model).__name__
    else:
        model_name = type(model).__name__

    date_time = "/%s/%s/%s" % (
        dataset,
        model_name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    # 设置需要优化的超参数
    model_kwargs_range = dict(
        # type_dim=dict(type=int, scale="power", base=2, low=4, high=10),
        # ent_dim=dict(type=int, scale="power_two", low=6, high=10),
    )
    loss_kwargs_ranges = dict()
    # regularizer_kwargs_ranges = dict()
    optimizer_kwargs_ranges = dict(
        lr=dict(type=float, low=0.0001, high=0.0009, step=0.0001)
    )
    # lr_scheduler_kwargs_ranges = dict()
    negative_sampler_kwargs_ranges = dict(
        num_negs_per_pos=dict(type=int, scale="power_two", low=1, high=9)
    )
    # training_kwargs_ranges = dict()

    # 从参数列表中删除设置的超参数
    if len(model_kwargs_range):
        for kwarg in model_kwargs_range:
            model_kwargs.pop(kwarg)
    if len(loss_kwargs_ranges):
        for kwarg in loss_kwargs_ranges:
            loss_kwargs.pop(kwarg)
    if len(optimizer_kwargs_ranges):
        for kwarg in optimizer_kwargs_ranges:
            optimizer_kwargs.pop(kwarg)
    if len(negative_sampler_kwargs_ranges):
        for kwarg in negative_sampler_kwargs_ranges:
            negative_sampler_kwargs.pop(kwarg)

    from Custom.Custom_hpo import hpo_pipeline
    from optuna.samplers import RandomSampler
    from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler

    # loss 和 模型应该分别初始化
    pipeline_result = hpo_pipeline(
        sampler=RandomSampler,
        n_trials=60,
        training=training_data,
        validation=validation,
        testing=testing,
        model=model,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_range,
        loss=soft_loss,
        loss_kwargs=loss_kwargs,
        loss_kwargs_ranges=loss_kwargs_ranges,
        device=args.device,
        result_tracker="tensorboard",
        result_tracker_kwargs=dict(
            experiment_path="../result/tensorBoard_log/hpo/"
            + args.description
            + "/"
            + date_time,
        ),
        study_name=args.description + date_time,
        storage="sqlite:///../models/{}.db".format(dataset),
        load_if_exists=True,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_kwargs_ranges=optimizer_kwargs_ranges,
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges,
        **pipeline_config,
    )

    model_path = "../models/hpo" + date_time
    pipeline_result.save_to_directory(model_path)
