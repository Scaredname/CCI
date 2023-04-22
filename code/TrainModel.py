# Get a training dataset
import argparse
import datetime
import json
import os

from Custom.CustomTripleFactory import TriplesTypesFactory
from pykeen.datasets import YAGO310, FB15k237, Nations, get_dataset
from pykeen.triples import TriplesFactory
from pykeen.typing import LabeledTriples
from utilities import get_white_list_relation, readTypeData

HEAD = 0
TAIL = 2

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_index',type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, choices=['fb15k-237-type', 'YAGO3-10', 'FB15k237', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242'], default='fb15k-237-type')
parser.add_argument('-o', '--optimizer', type=str, default='adam')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-lm', '--loss_margin', type=float, default=9.0)
parser.add_argument('-at', '--adversarial_temperature', type=float, default=1.0)
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
parser.add_argument('-pre', '--IfUsePreTrainTypeEmb', action='store_true', default=False)
parser.add_argument('-rw', '--ReglurizerWeight', type=float, default=0.001)
parser.add_argument('-rp', '--ReglurizerNorm', type=float, default=3.0)
parser.add_argument('-hnt', '--ifHasNoneType', action='store_true', default=False)
parser.add_argument('-tes', '--ifTestEarlyStop', action='store_true', default=False)
parser.add_argument('-stop', '--stopper', type=str, choices=['early', 'nop'], default='early')
args = parser.parse_args()


if args.ifTestEarlyStop:
    frequency = 1
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

def splitTypeData(data:TriplesFactory, type_position = 0) -> "LabeledTriples, LabeledTriples":
    unlike_type_rel, like_type_rel = get_white_list_relation(data, type_position=type_position)
    return data.label_triples(data.new_with_restriction(relations=unlike_type_rel).mapped_triples), data.label_triples(data.new_with_restriction(relations=like_type_rel).mapped_triples), unlike_type_rel, like_type_rel

dataset = args.dataset
if dataset == 'fb15k-237-type':
    training_data, validation, testing = readTypeData(dataset, data_pro_func=splitTypeData, type_position=HEAD, create_inverse_triples=args.CreateInverseTriples, hasNoneType=args.ifHasNoneType, type_smoothing=args.type_smoothing)
elif 'CAKE' in dataset:
    training_data, validation, testing = readTypeData(dataset, data_pro_func=splitTypeData, type_position=TAIL, create_inverse_triples=args.CreateInverseTriples, hasNoneType=args.ifHasNoneType, type_smoothing=args.type_smoothing)
else:
    if dataset == 'FB15k237':
        data = FB15k237(create_inverse_triples = args.CreateInverseTriples)
    elif dataset == 'YAGO3-10':
        data = YAGO310(create_inverse_triples = args.CreateInverseTriples)

    if args.IfUseTypeLike:
        training_triples, training_type_triples, unlike_type_rel, like_type_rel = splitTypeData(data.training, type_position=TAIL)
        training_data = TriplesTypesFactory.from_labeled_triples(triples=training_triples, type_triples=training_type_triples, type_position=TAIL, create_inverse_triples=args.CreateInverseTriples, type_smoothing=args.type_smoothing)

        
        dataset += '-TypeLike'

        validation = TriplesFactory.from_labeled_triples(
                data.validation.label_triples(data.validation.mapped_triples), 
                entity_to_id=training_data.entity_to_id, 
                relation_to_id=training_data.relation_to_id,
                create_inverse_triples=args.CreateInverseTriples,)
        validation = validation.new_with_restriction(relations=unlike_type_rel)
        testing = TriplesFactory.from_labeled_triples(
                data.testing.label_triples(data.testing.mapped_triples), 
                entity_to_id=training_data.entity_to_id, 
                relation_to_id=training_data.relation_to_id,
                create_inverse_triples=args.CreateInverseTriples,)
        testing = testing.new_with_restriction(relations=unlike_type_rel)

        # a = get_dataset(training = training_data, validation = validation, testing = testing)
        # print(a.summary_str())
    else:
        training_data = data.training
        validation = data.validation
        testing = data.testing





import torch
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
    args.description='PreTrainTypeEmb'

if args.ifHasNoneType:
    args.description+='HasNoneType'

if args.type_smoothing:
    args.description +='TypeSmoothing'

if args.model_index == 0:
    model = ESETCwithTransE(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
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
            )
elif args.model_index == 2:

    model = ESETCwithTuckER(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            loss='BCEAfterSigmoidLoss',
            dropout_0 = 0.3,
            usepretrained = args.IfUsePreTrainTypeEmb,
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
            )

elif args.model_index == 11:
    model = TransE(
            triples_factory=training_data,
            embedding_dim=args.model_ent_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=1.0,
                margin=args.loss_margin,
            ),
            # loss = 'BCEAfterSigmoidLoss',
    )

elif args.model_index == 12:
    model = RotatE(
            triples_factory=training_data,
            embedding_dim=args.model_ent_dim,
            entity_initializer='uniform',
            relation_initializer='init_phases',
            relation_constrainer= 'complex_normalize',
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=1.0,
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

elif args.model_index == 21:
    model = RSETCwithTransE(
            triples_factory=training_data,
            dropout=args.dropout,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            )

elif args.model_index == 31:
    model = CatESETCwithTransE(
            triples_factory=training_data,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
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
            )

elif args.model_index == 41:
    model = CatRSETCwithTransE(
            triples_factory=training_data,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            loss='NSSALoss',
            loss_kwargs=dict(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            ),
            usepretrained = args.IfUsePreTrainTypeEmb,
            )
    
elif args.model_index == 42:
    model = CatRSETCwithRotate(
            triples_factory=training_data,
            dropout=args.dropout,
            data_type=torch.cfloat,
            bias = args.project_with_bias,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
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
            )

if torch.cuda.is_available():
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

pipeline_result.save_to_directory(modelpath)
with open(os.path.join(modelpath, 'config.json'), 'w') as f:
    json.dump(pipeline_result.configuration, f, indent=1)

        