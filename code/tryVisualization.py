'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-02-10 13:37:11
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-02-17 14:56:15
FilePath: /undefined/home/ni/Desktop/try/code/tryVisualization.py
Description: 

Copyright (c) 2023 by Ni Runyu ni-runyu@ed.tmu.ac.jp, All Rights Reserved. 
'''
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pykeen
import seaborn as sns
import torch
from Custom.CustomTripleFactory import TriplesTypesFactory
from matplotlib import cm
from pykeen.datasets import YAGO310, FB15k237
from pykeen.datasets import analysis as data_analysis
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from utilities import get_white_list_relation, readTypeData

HEAD = 0
TAIL = 2
pattern = r"\(([^)]+)\)"

def draw_heatmap(data, title, save_path, vmax=None, vmin=None, format='png'):
    plt.clf()
    fig, ax = plt.subplots()
    data = pd.DataFrame(data)
    im = ax.imshow(data, cmap=matplotlib.colormaps['Greys'])
    ax = ax.set_aspect('auto')
    if vmax is not None:
        im.set_clim(vmax=vmax)
    if vmin is not None:
        im.set_clim(vmin=vmin)
    fig.colorbar(im, ax=ax, label='Interactive colorbar')
    save_path = save_path + '.' + format
    plt.savefig(save_path, format=format)

def draw_example(data, save_path, format='png'):
    
    x = np.arange(0, len(data))

    plt.clf()
    plt.bar(x=x, height=data)
    save_path = save_path + '.' + format
    plt.savefig(save_path, format=format)

def find_k_revelent_type(data, id, string2id, type2id, mode = 'rel', flag = 'before', de = None, k = 1):
    # 用来查看某个特例的类型权重
    types = np.array(list(type2id.keys()))
    string = list(string2id.keys())
    rev = ''
    de = '-' + de
    if id >= len(string):
        rev = '-reverse'
        id = id - len(string)
    figure_floder = os.path.join('../models/', dataset, 'figures/')
    if not os.path.exists(figure_floder):
        os.makedirs(figure_floder)
    if mode == 'rel':
        if flag == 'before':    
            example_path_h = os.path.join(figure_floder, 'rel_type_weight_beforeTraining_h_%s'%(str(id)+rev+de))
            example_path_t = os.path.join(figure_floder, 'rel_type_weight_beforeTraining_t_%s'%(str(id)+rev+de))
        else:
            example_path_h = os.path.join(figure_floder, 'rel_type_weight_afterTraining_h_%s'%(str(id)+rev+de))
            example_path_t = os.path.join(figure_floder, 'rel_type_weight_afterTraining_t_%s'%(str(id)+rev+de))
        print('for head')
        print('for relation %s, the most revelent type is: '%(string[id]+rev), types[np.argsort(-data[0, id, :])[:k]], 'with weight: ', data[0, id, :][np.argsort(-data[0, id, :])[:k]])
        print('for relation %s, the most unrevelent type is: '%(string[id]+rev), types[np.argsort(data[0, id, :])[:k]], 'with weight: ', data[0, id, :][np.argsort(data[0, id, :])[:k]])
        print('for tail')
        print('for relation %s, the most revelent type is: '%(string[id]+rev), types[np.argsort(-data[0, id, :])[:k]], 'with weight: ', data[1, id, :][np.argsort(-data[1, id, :])[:k]])
        print('for relation %s, the most unrevelent type is: '%(string[id]+rev), types[np.argsort(data[0, id, :])[:k]], 'with weight: ', data[1, id, :][np.argsort(data[1, id, :])[:k]])
        draw_example(data[0, id, :], example_path_h, format='svg')
        draw_example(data[1, id, :], example_path_t, format='svg')
    else:
        if flag == 'before':
            example_path = os.path.join(figure_floder, 'ent_type_weight_beforeTraining_%s'%(str(id)+de))
        else:
            example_path = os.path.join(figure_floder, 'ent_type_weight_afterTraining_%s'%(str(id)+de))

        print('for entity %s, the most revelent type is: '%(string[id]), types[np.argsort(-data[id, :])[:k]], 'with weight: ', data[id, :][np.argsort(-data[id, :])[:k]])
        print('for entity %s, the most unrevelent type is: '%(string[id]), types[np.argsort(data[id, :])[:k]], 'with weight: ', data[id, :][np.argsort(data[id, :])[:k]])
        draw_example(data[id, :], example_path, format='svg')

        


def splitTypeData(data:TriplesFactory, type_position = 0) -> "LabeledTriples, LabeledTriples":
    unlike_type_rel, like_type_rel = get_white_list_relation(data, type_position=type_position)

    return data.label_triples(data.new_with_restriction(relations=unlike_type_rel).mapped_triples), data.label_triples(data.new_with_restriction(relations=like_type_rel).mapped_triples), unlike_type_rel, like_type_rel



def load_data(IfUseTypeLike, d):
    
    relevent_ent = list()
    if IfUseTypeLike:
        training_triples, training_type_triples, unlike_type_rel, like_type_rel = splitTypeData(d.training, type_position=TAIL)
        training_data = TriplesTypesFactory.from_labeled_triples(triples=training_triples, type_triples=training_type_triples, type_position=TAIL, create_inverse_triples=True)
        for triple in training_type_triples:
            try:
                training_data.entity_to_id[triple[HEAD]]
                relevent_ent.append(triple[HEAD])
            except:
                # print(triple[HEAD])
                pass
        validation = TriplesFactory.from_labeled_triples(
                    d.validation.label_triples(d.validation.mapped_triples), 
                    entity_to_id=training_data.entity_to_id, 
                    relation_to_id=training_data.relation_to_id,
                    create_inverse_triples=True)
        validation = validation.new_with_restriction(relations=unlike_type_rel)
        testing = TriplesFactory.from_labeled_triples(
                        d.testing.label_triples(d.testing.mapped_triples), 
                        entity_to_id=training_data.entity_to_id, 
                        relation_to_id=training_data.relation_to_id,
                        create_inverse_triples=True)
        testing = testing.new_with_restriction(relations=unlike_type_rel)
        testing = testing.new_with_restriction(entities=relevent_ent)

    else:
        training_data = d.training
        validation = d.validation
        testing = d.testing.new_with_restriction(entities=relevent_ent)

    return training_data, validation, testing

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', choices=['FB15k237', 'YAGO3-10', 'fb15k-237-type', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242'], default='fb15k-237-type')
parser.add_argument('-reverse', '--CreateInverseTriples', action='store_true', default=False)
parser.add_argument('-t', '--IfUseTypeLike', action='store_true', default=False)
args = parser.parse_args()
if args.dataset == 'fb15k-237-type':
    training_data, validation, testing = readTypeData(args.dataset, data_pro_func=splitTypeData, type_position=HEAD, create_inverse_triples=args.CreateInverseTriples)
elif 'CAKE' in args.dataset:
    training_data, validation, testing = readTypeData(args.dataset, data_pro_func=splitTypeData, type_position=TAIL, create_inverse_triples=args.CreateInverseTriples)
else:
    if args.dataset == 'FB15k237':
            data = FB15k237(create_inverse_triples = args.CreateInverseTriples)
    elif args.dataset == 'YAGO3-10':
        data = YAGO310(create_inverse_triples = args.CreateInverseTriples)


    training_data, validation, testing = load_data(args.IfUseTypeLike, data)

dataset = args.dataset
if args.IfUseTypeLike:
    dataset = dataset + '-TypeLike'

heatmap_before_save_path = os.path.join('../models/', dataset, 'ent_type_weight_beforeTraining')
heatmap_before_save_path_rel_h = os.path.join('../models/', dataset, 'rel_type_weight_beforeTraining_h')
heatmap_before_save_path_rel_t = os.path.join('../models/', dataset, 'rel_type_weight_beforeTraining_t')


draw_heatmap(training_data.ents_types, 'entity type weight before training', heatmap_before_save_path)

# de = 'PreTrainTypeEmb'
de = 'noDescription'
m_name = 'ESETCwithTransE'
# date = '20230225-001357'
date = '20230225-064727'
example_id = training_data.entity_to_id['/m/035qy']
# example_id = np.where(np.sum(training_data.ents_types, axis=1)== 0)[0][3] 

find_k_revelent_type(training_data.ents_types, example_id, training_data.entity_to_id, training_data.types_to_id, mode='entity', de=de, k=5)
num_relations = len(training_data.relation_to_id)

model_path = os.path.join('../models/', dataset, de, m_name, date)
model = torch.load(model_path+'/trained_model.pkl')
ents_types = model.ents_types.data.cpu().numpy()
draw_heatmap(ents_types, 'entity type weight after training', os.path.join(model_path, 'ent_type_weight_afterTraining'))

find_k_revelent_type(ents_types, example_id, training_data.entity_to_id, training_data.types_to_id, mode='ent', flag='after', de=de, k=5)
