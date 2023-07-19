'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2022-12-26 11:19:42
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-19 16:07:11
FilePath: /ESETC/code/STNS_train.py
Description: 

Copyright (c) 2023 by Ni Runyu ni-runyu@ed.tmu.ac.jp, All Rights Reserved. 
'''
import argparse
import datetime
import json
import os

from Custom.CustomLoss import SoftTypeawareNegativeSmapling
from Custom.CustomSampler import TypeNegativeSampler
from Custom.CustomTrain import TypeSLCWATrainingLoop
from Custom.TypeModels.CatRSETC import CatRSETCwithRotate, CatRSETCwithTransE
from utilities import load_dataset

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_index',type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, choices=['fb15k-237-type', 'YAGO3-10', 'FB15k237', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242', 'yago5k-106'], default='fb15k-237-type')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-lm', '--loss_margin', type=float, default=9.0)
    parser.add_argument('-at', '--adversarial_temperature', type=float, default=1.0)
    parser.add_argument('-twt', '--type_weight_temperature', type=float, default=1.0)
    parser.add_argument('-mw', '--modulus_weight', type=float, default=1.0)
    parser.add_argument('-pw', '--phase_weight', type=float, default=1.0)
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
    parser.add_argument('-ch', '--checkpoint', type=str, default=None)
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
    parser.add_argument('-naf', '--ifNoActivationFuncion', action='store_true', default=False, help='If use ActivationFuncion for type weight')
    parser.add_argument('-wm', '--ifWeightMask', action='store_true', default=False, help='If use entity types constrains for type weight')
    parser.add_argument('-ot', '--ifOneType', action='store_true', default=False, help='If only use the most related entity type')
    parser.add_argument('-naet', '--ifNotAddEntType', help="When get entity's type embedding, whether add entity_type weight to relation_type weight", action='store_true', default=False)
    parser.add_argument('-stop', '--stopper', type=str, choices=['early', 'nop'], default='early')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()


    
    training_data, validation, testing = load_dataset(dataset=args.dataset, IfUseTypeLike=args.IfUseTypeLike, CreateInverseTriples=args.CreateInverseTriples, type_smoothing=args.type_smoothing, ifHasNoneType=args.ifHasNoneType, ifTypeAsTrain=args.ifTypeAsTrain, use_random_weights=args.ifRandomWeight, select_one_type=args.ifOneType)

    loss = SoftTypeawareNegativeSmapling(
                reduction='mean',
                adversarial_temperature=args.adversarial_temperature,
                margin=args.loss_margin,
            )

    model = CatRSETCwithTransE(
            triples_factory=training_data,
            ent_dim=args.model_ent_dim,
            rel_dim=args.model_rel_dim,
            type_dim=args.model_type_dim,
            freeze_matrix = args.ifFreezeWeights,
            freeze_type_emb = args.ifFreezeTypeEmb,
            add_ent_type = not args.ifNotAddEntType,
            loss=loss,
            usepretrained = args.IfUsePreTrainTypeEmb,
            activation_weight = not args.ifNoActivationFuncion,
            weight_mask = args.ifWeightMask,
            type_weight_temperature = args.type_weight_temperature,
            )
    
    model.to(args.device)
    train_loop = TypeSLCWATrainingLoop(
        model = model,
        triples_factory = training_data,
        negative_sampler = TypeNegativeSampler,
        negative_sampler_kwargs = dict(
            rel_related_ent = training_data.rel_related_ent,
            num_negs_per_pos=2,)
    )

    _ = train_loop.train(
    triples_factory=training_data,
    num_epochs=5,
    batch_size=256,
    # NEW: validation evaluation callback
    callbacks="evaluation-loop",
    callback_kwargs=dict(
        prefix="validation",
        factory=validation,
    ),
) 