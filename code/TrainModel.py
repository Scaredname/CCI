# Get a training dataset
import argparse
import datetime
import json
import os

from Custom.CustomTripleFactory import TriplesTypesFactory
from pykeen.triples import TriplesFactory
from pykeen.typing import LabeledTriples
from utilities import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_index',type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, choices=['fb15k-237-type', 'YAGO3-10', 'FB15k237', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242', 'yago5k-106'], default='fb15k-237-type')
parser.add_argument('-o', '--optimizer', type=str, default='adam')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-lm', '--loss_margin', type=float, default=9.0)
parser.add_argument('-at', '--adversarial_temperature', type=float, default=1.0)
parser.add_argument('-twt', '--type_weight_temperature', type=float, default=1.0)
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('-train', '--training_loop', type=str, default='lcwa')
parser.add_argument('-neg', '--negative_sampler', type=str, default=None)
parser.add_argument('-nen', '--num_negs_per_pos', type=int, default=None)
parser.add_argument('-ef', '--filtered', type=bool, default=True)
parser.add_argument('-eb', '--evaluator_batch_size', type=int, default=128)
parser.add_argument('-med', '--model_ent_dim', type=int, default=100)
parser.add_argument('-mrd', '--model_rel_dim', type=int, default=100)
parser.add_argument('-mtd', '--model_type_dim', type=int, default=100)
parser.add_argument('-tsm', '--type_smoothing', type=float, default=0.0)

parser.add_argument('-pb', '--project_with_bias', action='store_true', default=False)
parser.add_argument('-drop', '--dropout', type=float, default=0.0)

parser.add_argument('-de', '--description', type=str, default='noDescription')
parser.add_argument('-reverse', '--CreateInverseTriples', action='store_true', default=False)
parser.add_argument('-t', '--IfUseTypeLike', action='store_true', default=False)
parser.add_argument('-pre', '--IfUsePreTrainTypeEmb', default=None, type=str, choices=[None, 'bert-base-uncased', 'bert-large-uncased'], help='If use pre-trained type embeddings, the name of pretained model')
parser.add_argument('-rw', '--ReglurizerWeight', type=float, default=0.001)
parser.add_argument('-rp', '--ReglurizerNorm', type=float, default=3.0)
parser.add_argument('-hnt', '--ifHasNoneType', action='store_true', default=False)
parser.add_argument('-tes', '--ifTestEarlyStop', action='store_true', default=False)
parser.add_argument('-tat', '--ifTypeAsTrain', action='store_true', default=False)
parser.add_argument('-fw', '--ifFreezeWeights', action='store_true', default=False)
parser.add_argument('-shp', '--ifSearchHyperParameters', action='store_true', default=False)
parser.add_argument('-fte', '--ifFreezeTypeEmb', action='store_true', default=False, help='If freeze type embeddings, default is False')
parser.add_argument('-randw', '--ifRandomWeight', action='store_true', default=False, help='If use random weight, default is False')
parser.add_argument('-af', '--ifActivationFuncion', action='store_true', default=False, help='If use ActivationFuncion for type weight')
parser.add_argument('-wm', '--ifWeightMask', action='store_true', default=False, help='If use entity types constrains for type weight')
parser.add_argument('-ot', '--ifOneType', action='store_true', default=False, help='If only use the most related entity type')
parser.add_argument('-naet', '--ifNotAddEntType', help="When get entity's type embedding, whether add entity_type weight to relation_type weight", action='store_true', default=False)
parser.add_argument('-stop', '--stopper', type=str, choices=['early', 'nop'], default='early')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
args = parser.parse_args()


if args.ifTestEarlyStop:
    frequency = 1
elif args.ifSearchHyperParameters:
    frequency = 5
else:
    frequency = 50
pipeline_config = dict(
    optimizer=args.optimizer,
    optimizer_kwargs=dict(
        lr=args.learning_rate,
    ),
    training_loop=args.training_loop,
    training_kwargs=dict(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    ),
    negative_sampler=args.negative_sampler,
    negative_sampler_kwargs=dict(
        num_negs_per_pos=args.num_negs_per_pos,
    ),
    evaluator = 'RankBasedEvaluator',
    evaluator_kwargs=dict(
        filtered=args.filtered,
        batch_size=args.evaluator_batch_size,
    ),
    stopper=args.stopper,
    stopper_kwargs=dict(
        frequency=frequency,
        patience=3,
        relative_delta=0.0001,
        metric='mean_reciprocal_rank',
        evaluation_batch_size=args.evaluator_batch_size,
    ),
)



dataset = args.dataset
training_data, validation, testing = load_dataset(dataset=dataset, IfUseTypeLike=args.IfUseTypeLike, CreateInverseTriples=args.CreateInverseTriples, type_smoothing=args.type_smoothing, ifHasNoneType=args.ifHasNoneType, ifTypeAsTrain=args.ifTypeAsTrain, use_random_weights=args.ifRandomWeight, select_one_type=args.ifOneType)


import torch
from Custom.OriginRotatE import FloatRotatE
from Custom.TypeModels.CatESETC import CatESETCwithRotate, CatESETCwithTransE
from Custom.TypeModels.CatRSETC import CatRSETCwithRotate, CatRSETCwithTransE
from Custom.TypeModels.ESETCwithComplEx import (DistMult, ESETCwithComplEx,
                                                ESETCwithDistMult)
from Custom.TypeModels.ESETCwithRotate import ESETCwithRotate, ESETCwithTransE
from Custom.TypeModels.ESETCwithTuckER import ESETCwithTuckER
from Custom.TypeModels.RSETC import RSETCwithTransE
# Pick a model
# from Custom.CustomModel import EETCRLwithRotate
from pykeen.models import ComplEx, DistMultLiteral, RotatE, TransE
from pykeen.nn.init import xavier_uniform_
from pykeen.nn.modules import RotatEInteraction, TransEInteraction

if args.IfUsePreTrainTypeEmb:
    args.description+='PreTrainTypeEmb'

if args.ifHasNoneType:
    args.description+='HasNoneType'

if args.type_smoothing:
    args.description+='TypeSmoothing'

if args.ifTypeAsTrain:
    args.description+='TypeAsTrain'

if args.ifFreezeWeights:
    args.description+='FreezeWeights'

if args.ifFreezeTypeEmb:
    args.description+='FreezeTypeEmb'

if args.ifNotAddEntType:
    args.description+='NotAddEntType'

if args.ifRandomWeight:
    args.description+='RandomWeight'

if args.ifActivationFuncion:
    args.description+='ActivationFuncion'

if args.ifWeightMask:
    args.description+='WeightMask'

if args.ifSearchHyperParameters:
    args.description+='SearchHP'

if args.ifOneType:
    args.description+='OneType'

if args.model_index == 0:
    model = ESETCwithTransE(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )
elif args.model_index == 1:
    # 因为没有使用cfloat，通过乘2来确保参数维度和实际维度相同
    # 现在发现使用cfloat会降低模型性能
    model = ESETCwithRotate(
            triples_factory=training_data,
            dropout=args.dropout,
            data_type=torch.float,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim*2,
            rel_dim=args.model_rel_dim*2,
            type_dim=args.model_type_dim*2,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            type_initializer='xavier_uniform_',
            entity_initializer='uniform',
            relation_initializer='init_phases',
            relation_constrainer= 'complex_normalize',
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )
elif args.model_index == 2:

    model = ESETCwithTuckER(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            loss='BCEAfterSigmoidLoss',
            dropout_0 = 0.3,
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
    )

elif args.model_index == 3:
    model = ESETCwithComplEx(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim*2,
            rel_dim=args.model_rel_dim*2,
            type_dim=args.model_type_dim*2,
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
            type_initializer=xavier_uniform_,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            regularizer_kwargs = dict(
                weight=args.ReglurizerWeight,
                p=args.ReglurizerNorm, # 使用N3norm
                normalize=True,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,

            type_weight_temperature = args.type_weight_temperature,
            )
elif args.model_index == 4:
    model = ESETCwithDistMult(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
            type_initializer=xavier_uniform_,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            regularizer_kwargs = dict(
                weight=args.ReglurizerWeight,
                p=args.ReglurizerNorm, # 使用N3norm
                normalize=True,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )

elif args.model_index == 11:
    model = TransE(
            triples_factory=training_data,
            embedding_dim=args.model_ent_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            # loss = 'BCEAfterSigmoidLoss',
    )

elif args.model_index == 12:
    model = RotatE(
            triples_factory=training_data,
            embedding_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            entity_initializer='uniform',
            relation_initializer='init_phases',
            relation_constrainer= 'complex_normalize',
            # relation_constrainer=None,
            # relation_constrainer='normalize',
            # relation_constrainer_kwargs = dict(
            #     p = 1.0,
            # ),
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
    )

elif args.model_index == 13:
    model = ComplEx(
            triples_factory=training_data,
            entity_initializer=xavier_uniform_,
            relation_initializer=xavier_uniform_,
            embedding_dim=args.model_ent_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            regularizer_kwargs = dict(
                weight=args.ReglurizerWeight,
                p=args.ReglurizerNorm, # 使用N3norm
                normalize=True,
            ),
            )

elif args.model_index == 14:
    model = DistMult(
            triples_factory=training_data,
            # entity_initializer='uniform',
            # relation_initializer='uniform',
            # entity_initializer=xavier_uniform_,
            # relation_initializer=xavier_uniform_,
            embedding_dim=args.model_ent_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            regularizer_kwargs = dict(
                weight=args.ReglurizerWeight,
                p=args.ReglurizerNorm, # 使用N3norm
                normalize=True,
            ),
            )
    
elif args.model_index == 15:
    model = FloatRotatE(
            triples_factory=training_data,
            embedding_dim=args.model_ent_dim,
            lm=args.loss_margin,
            # entity_initializer='uniform',
            # relation_initializer='init_phases',
            # relation_constrainer= 'complex_normalize',
            relation_constrainer=None,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
    )

elif args.model_index == 21:
    model = RSETCwithTransE(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            add_ent_type = not args.ifNotAddEntType,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )

elif args.model_index == 31:
    model = CatESETCwithTransE(
            triples_factory=training_data,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )

elif args.model_index == 32:
    model = CatESETCwithRotate(
            triples_factory=training_data,
            dropout=args.dropout,
            data_type=torch.cfloat,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            type_initializer='xavier_uniform_',
            entity_initializer='uniform',
            relation_initializer='init_phases',
            relation_constrainer= 'complex_normalize',
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )

elif args.model_index == 41:
    model = CatRSETCwithTransE(
            triples_factory=training_data,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            add_ent_type = not args.ifNotAddEntType,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )
    
elif args.model_index == 42:
    model = CatRSETCwithRotate(
            triples_factory=training_data,
            dropout=args.dropout,
            ent_dtype = torch.float,
            rel_dtype = torch.cfloat,
            type_dtype = torch.float,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim // 2, # relation的数据类型的cfloat
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            add_ent_type = not args.ifNotAddEntType,
            type_initializer='xavier_uniform_',
            entity_initializer='uniform',
            relation_initializer='init_phases',
            relation_constrainer= 'complex_normalize',
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = args.ifActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )

if torch.cuda.is_available():
    print('Using GPU')
    model.to('cuda')
    

# Pick an optimizer from Torch
from torch.optim import Adam

optimizer = Adam(params=model.get_grad_params())

from pykeen.pipeline import pipeline

if args.description is not None:
    model_name = args.description + '/' + type(model).__name__ 
else:
    model_name = type(model).__name__

date_time = '/%s/%s/%s'%(dataset, model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
pipeline_result = pipeline(
    training=training_data,
    validation=validation,
    testing=testing,
    model=model,
    result_tracker='tensorboard',
    result_tracker_kwargs=dict(
        experiment_path='../result/tensorBoard_log' + date_time,
    ),
    **pipeline_config, 
)

modelpath = '../models' + date_time
for config in pipeline_result.configuration:
    pipeline_result.configuration[config] = str(pipeline_result.configuration[config])

pipeline_result.configuration['model_kwargs'] = str(model.__dict__)
pipeline_result.configuration['num_parameter_bytes'] = str(model.num_parameter_bytes / 125000) + 'MB'

pipeline_result.configuration['loss_kwargs'] = str(model.loss.__dict__)
pipeline_result.configuration['loss'] = type(model.loss).__name__
pipeline_result.configuration['type_smoothing'] = args.type_smoothing
if args.training_loop == 'slcwa':
    pipeline_result.configuration['negative_sampler'] = args.negative_sampler
    pipeline_result.configuration['num_negs_per_pos'] = args.num_negs_per_pos

if args.IfUsePreTrainTypeEmb:
    pipeline_result.configuration['pre_trained_type_name'] = args.IfUsePreTrainTypeEmb

pipeline_result.configuration['type_weight_temperature'] = args.type_weight_temperature

pipeline_result.save_to_directory(modelpath)
with open(os.path.join(modelpath, 'config.json'), 'w') as f:
    json.dump(pipeline_result.configuration, f, indent=1)

        