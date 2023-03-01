'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-02 16:32:08
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-03-01 11:35:49
FilePath: /ESETC/code/pdResult.py
Description: 

Copyright (c) 2023 by error: git config user.name && git config user.email & please set dead value or install git, All Rights Reserved. 
'''
import argparse
import ast
import json
import os
import re
from collections import defaultdict

import pandas as pd

pattern = r"\(([^)]+)\)"
pattern1 = r"\(_embeddings\): Embedding\((.*)\)"

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', choices=['YAGO3-10-TypeLike', 'YAGO3-10', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242'], default='YAGO3-10-TypeLike')
parser.add_argument('-de', choices=['LearnableAdjacency', 'noDescription', 'previousResult_noDescription', 'PreTrainTypeEmb'], default='noDescription')
args = parser.parse_args()


# result_path = '../models/YAGO3-10/'
result_path = '../models/%s/'%(args.dataset)
dataset = args.dataset.replace('-', '')
save_path = '../result/%s.csv'%(dataset)
if args.de:
    result_path = '../models/%s/%s/'%(args.dataset, args.de)
    save_path = '../result/%s_%s.csv'%(dataset, args.de)

# save_path = '/home/ni/presentation/Jan_progress/result/all_mrr.json'

results_dict = defaultdict(list)

for file_name in os.listdir(result_path):
    model_name = file_name
    
    file_path = os.path.join(result_path, file_name)

    for f in os.listdir(file_path):
        results_dict['dataset'].append(dataset)
        results_dict['model'].append(model_name)
        date = f
        file_path1 = os.path.join(file_path, date)
        if '-' in date:
            results_dict['date'].append(date)
            
            result_path1 = os.path.join(file_path1, 'results.json')
            config_path = os.path.join(file_path1, 'config.json')
            with open(result_path1, 'r', encoding='utf8') as ff:
                results = json.load(ff)
            with open(config_path, 'r', encoding='utf8') as ff:
                config = json.load(ff)
            
        results_dict['mrr'].append(results['metrics']['both']['realistic']['inverse_harmonic_mean_rank'])
        results_dict['hits@1'].append(results['metrics']['both']['realistic']['hits_at_1'])
        results_dict['hits@3'].append(results['metrics']['both']['realistic']['hits_at_3'])
        results_dict['hits@10'].append(results['metrics']['both']['realistic']['hits_at_10'])

        results_dict['model-size'].append(float(config['num_parameter_bytes'][:-2])*0.125*0.25)
        results_dict['batch-size'].append(config['batch_size'])
        results_dict['train-loop'].append(config['training_loop'])
        results_dict['optimizer'].append(config['optimizer'])
        results_dict['learning-rate'].append(config['optimizer_kwargs'].split(',')[0].split(':')[1])

        model_dims = re.findall(pattern1, config['model_kwargs'])
        # model_dims = ['14505, 200', '237, 200']
        results_dict['ent-dim'].append(int(model_dims[0].split(',')[1].strip()))
        results_dict['rel-dim'].append(int(model_dims[1].split(',')[1].strip()))
        try:
            results_dict['type-dim'].append(int(model_dims[2].split(',')[1].strip()))
        except:
            results_dict['type-dim'].append('-')

        if 'SLCWATrainingLoop' == config['training_loop']:
            results_dict['negative_sampler'].append(config['negative_sampler'])
            results_dict['num_negs_per_pos'].append(config['num_negs_per_pos'])
        else:
            results_dict['negative_sampler'].append('-')
            results_dict['num_negs_per_pos'].append('-')

        if 'loss' in config:
            results_dict['loss'].append(config['loss'])
            results_dict['loss_kwargs'].append(config['loss_kwargs'])
        else:
            results_dict['loss'].append('see config')
            results_dict['loss_kwargs'].append(config['loss_kwargs'])

r = pd.DataFrame(results_dict)

r.to_csv(save_path, index=False)

        
