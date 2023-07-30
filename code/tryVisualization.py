'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-02-10 13:37:11
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-30 16:56:03
FilePath: /code/tryVisualization.py
Description: 输出对于特定实体的类型权重，以及对于特定关系的类型权重

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
from matplotlib import cm
from pykeen.datasets import YAGO310, FB15k237
from pykeen.datasets import analysis as data_analysis
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory

from Custom.CustomTripleFactory import TriplesTypesFactory
from utilities import load_dataset

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

def calculate_kurtosis(vector):
    vector = torch.tensor(vector).detach()
    vector *= 100
    mean = torch.mean(vector)
    std = torch.std(vector) #样本标准差
    print(std)
    fourth_moment = torch.mean((vector - mean) ** 4) #四阶样本中心矩
    return fourth_moment / (std ** 4) - 3


def draw_example(data, save_path, format='png'):
    
    x = np.arange(0, len(data))

    plt.clf()
    plt.bar(x=x, height=data)
    save_path = save_path + '.' + format
    plt.savefig(save_path, format=format)

def find_k_relevant_type(data, id, string2id, type2id, path, mode = 'rel', flag = 'before', de = None, k = 1):
    # 用来查看某个特例的类型权重
    types = np.array(list(type2id.keys()))
    string = list(string2id.keys())
    rev = ''
    de = '-' + de
    if id >= len(string):
        rev = '-reverse'
        id = id - len(string)
    figure_floder = os.path.join(path, 'figures/')
    if not os.path.exists(figure_floder):
        os.makedirs(figure_floder)
    if mode == 'rel':
        if flag == 'before':    
            example_path_h = os.path.join(figure_floder, 'rel_type_weight_beforeTraining_h_%s'%(str(id)+rev+de))
            example_path_t = os.path.join(figure_floder, 'rel_type_weight_beforeTraining_t_%s'%(str(id)+rev+de))
        else:
            example_path_h = os.path.join(figure_floder, 'rel_type_weight_afterTraining_h_%s'%(str(id)+rev+de))
            example_path_t = os.path.join(figure_floder, 'rel_type_weight_afterTraining_t_%s'%(str(id)+rev+de))
        
        print('for head ++++++++++++')
        print('for relation %s, the most relevant type is: '%(string[id]+rev), types[np.argsort(-data[0, id, :])[:k]], 'with weight: ', data[0, id, :][np.argsort(-data[0, id, :])[:k]], 'kurtosis: ', calculate_kurtosis(data[1, id, :]))
        print('for relation %s, the most irrelevant type is: '%(string[id]+rev), types[np.argsort(data[0, id, :])[:k]], 'with weight: ', data[0, id, :][np.argsort(data[0, id, :])[:k]], 'kurtosis: ', calculate_kurtosis(data[0, id, :][np.argsort(data[0, id, :])[:k]]))
        
        print('for tail +++++++++++')
        print('for relation %s, the most relevant type is: '%(string[id]+rev), types[np.argsort(-data[1, id, :])[:k]], 'with weight: ', data[1, id, :][np.argsort(-data[1, id, :])[:k]], 'kurtosis: ', calculate_kurtosis(data[1, id, :]))
        print('for relation %s, the most irrelevant type is: '%(string[id]+rev), types[np.argsort(data[1, id, :])[:k]], 'with weight: ', data[1, id, :][np.argsort(data[1, id, :])[:k]], 'kurtosis: ', calculate_kurtosis(data[1, id, :][np.argsort(data[1, id, :])[:k]]))
        
        draw_example(data[0, id, :], example_path_h, format='svg')
        draw_example(data[1, id, :], example_path_t, format='svg')
    else:
        if flag == 'before':
            example_path = os.path.join(figure_floder, 'ent_type_weight_beforeTraining_%s'%(str(id)+de))
        else:
            example_path = os.path.join(figure_floder, 'ent_type_weight_afterTraining_%s'%(str(id)+de))

        print('for entity %s, the most relevant type is: '%(string[id]), types[np.argsort(-data[id, :])[:k]], 'with weight: ', data[id, :][np.argsort(-data[id, :])[:k]])
        print('for entity %s, the most irrelevant type is: '%(string[id]), types[np.argsort(data[id, :])[:k]], 'with weight: ', data[id, :][np.argsort(data[id, :])[:k]])
        draw_example(data[id, :], example_path, format='svg')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['FB15k237', 'YAGO3-10', 'fb15k-237-type', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242', 'yago5k-106'], default='yago5k-106')
    parser.add_argument('-reverse', '--CreateInverseTriples', action='store_true', default=False)
    parser.add_argument('-t', '--IfUseTypeLike', action='store_true', default=False)
    args = parser.parse_args()


    # de = 'PreTrainTypeEmb'
    de = 'STNS-UsingSoftMaxOneType'
    # date = '20230225-001357'
    # m_name = 'CatRSETCwithRotate'
    m_name = 'NNYwithTransE'
    date = '20230720-142135'

    dataset = args.dataset
    model_path = os.path.join('../models/', dataset, de, m_name, date)
    model = torch.load(model_path+'/trained_model.pkl')

    training_data, validation, testing = load_dataset(dataset, strict_confidence=True)

    if args.IfUseTypeLike:
        dataset = dataset + '-TypeLike'


    vis_path = os.path.join('../result/visualization/', dataset, de, m_name, date)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path) 

    # heatmap_before_save_path = os.path.join(vis_path, 'ent_type_weight')
    heatmap_before_save_path_rel_h = os.path.join(vis_path,'rel_type_weight_beforeTraining_h')
    heatmap_before_save_path_rel_t = os.path.join(vis_path, 'rel_type_weight_beforeTraining_t')


    draw_heatmap(training_data.rels_types[0], 'relation head type weight before training', heatmap_before_save_path_rel_h)
    draw_heatmap(training_data.rels_types[1], 'relation tail type weight before training', heatmap_before_save_path_rel_t)

    # heatmap_after_save_path = os.path.join('../models/', dataset, 'ent_type_weight_afterTraining')
    heatmap_after_save_path_rel_h = os.path.join(vis_path, 'rel_type_weight_afterTraining_h')
    heatmap_after_save_path_rel_t = os.path.join(vis_path, 'rel_type_weight_afterTraining_t')

    rel_types_h_after_training = torch.nn.functional.normalize(model.rel_type_h_weights[0]._embeddings.weight.data, p=1, dim=1).cpu().numpy()
    rel_types_t_after_training = model.rel_type_t_weights[0]._embeddings.weight.data.cpu().numpy()

    draw_heatmap(rel_types_h_after_training, 'relation head type weight after training', heatmap_after_save_path_rel_h)
    draw_heatmap(rel_types_h_after_training, 'relation tail type weight after training', heatmap_after_save_path_rel_t)

    rel_label = ['hasWonPrize', 'diedIn']
    for rel in rel_label:
        example_id = training_data.relation_to_id[rel]
        

        print(f'{rel}_h_before: ', training_data.rels_types[0][example_id])
        print(f'{rel}_h_after: ', rel_types_h_after_training[example_id] / np.linalg.norm(rel_types_h_after_training[example_id], ord=1))

        norm_rel_types_h_after_training = rel_types_h_after_training[example_id] / np.linalg.norm(rel_types_h_after_training[example_id], ord=1)
        norm_rel_types_t_after_training = rel_types_t_after_training[example_id] / np.linalg.norm(rel_types_t_after_training[example_id], ord=1)

        types = ['person', 'artist', 'award', 'election']
        for t in types:
            typeid = training_data.types_to_id[t]
            print(f'{t}: ')
            print(f'{rel}_h_before: ', training_data.rels_types[0][example_id][typeid])
            print(f'{rel}_h_after: ', norm_rel_types_h_after_training[typeid])
            print(f'{rel}_t_before: ', training_data.rels_types[1][example_id][typeid])
            print(f'{rel}_t_after: ', norm_rel_types_t_after_training[typeid])


            

        sub_result = training_data.rels_types[0][example_id].cpu().numpy() - rel_types_h_after_training[example_id] /np.linalg.norm(rel_types_h_after_training[example_id], ord=1)
        indices = (training_data.rels_types[0][example_id] > 0)
        print(training_data.rels_types[0][example_id][indices])
        print(rel_types_h_after_training[example_id][indices])


        breakpoint()
        print('before training-------------------------------------------')
        print('rel: ', rel , 'injective confidence: ', training_data.rels_inj_conf[example_id])
        find_k_relevant_type(training_data.rels_types, example_id, training_data.relation_to_id, training_data.types_to_id, path=vis_path, mode='rel', de=de, k=5, flag='before')
        print('after training--------------------------------------------')
        find_k_relevant_type(np.array([rel_types_h_after_training, rel_types_t_after_training]), example_id, training_data.relation_to_id, training_data.types_to_id, path=vis_path, mode='rel', de=de, k=5, flag='after')

    ent_label = ['Henry_Mancini']
    for ent in ent_label:
        e_id = training_data.entity_to_id[ent]
        print(f'{ent}: ', training_data.ents_types[e_id])



    # example_id = np.where(np.sum(training_data.ents_types, axis=1)== 0)[0][3]
    # ent_list = ['/m/01z4y', '/m/02cllz', '/m/03p41', '/m/0dnqr', '/m/09th87']
    # for ent in ent_list: 
    #     example_id = training_data.entity_to_id[ent]

        # find_k_relevant_type(training_data.ents_types, example_id, training_data.entity_to_id, training_data.types_to_id, mode='entity', de=de, k=5)
        # num_relations = len(training_data.relation_to_id)


        # ents_types = model.ents_types.data.cpu().numpy()
        # draw_heatmap(ents_types, 'entity type weight after training', os.path.join(model_path, 'ent_type_weight_afterTraining'))

        # find_k_relevant_type(ents_types, example_id, training_data.entity_to_id, training_data.types_to_id, mode='ent', flag='after', de=de, k=5) 