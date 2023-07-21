'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-12-02 16:32:08
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-07-21 16:17:32
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
parser.add_argument('-d', '--dataset', choices=['YAGO3-10-TypeLike', 'YAGO3-10', 'CAKE-FB15K237', 'CAKE-FB15K', 'CAKE-NELL-995', 'CAKE-DBpedia-242', 'fb15k-237-type', 'yago5k-106'], default='YAGO3-10-TypeLike')
args = parser.parse_args()


result_path = '../models/%s/'%(args.dataset)
dataset = args.dataset.replace('-', '')
save_path = '../result/%s.csv'%(dataset)

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
                    results_dict['model'].append(model_name)
                    
                    date = ff
                    file_path1 = os.path.join(de_path, date)
                    if '-' in date:
                        results_dict['date'].append(date)
                        
                        result_path1 = os.path.join(file_path1, 'results.json')
                        config_path = os.path.join(file_path1, 'config.json')
                        with open(result_path1, 'r', encoding='utf8') as fff:
                            results = json.load(fff)
                        with open(config_path, 'r', encoding='utf8') as fff:
                            config = json.load(fff)
                        
                    results_dict['mrr'].append(results['metrics']['both']['realistic']['inverse_harmonic_mean_rank'])
                    results_dict['hits@1'].append(results['metrics']['both']['realistic']['hits_at_1'])
                    results_dict['hits@3'].append(results['metrics']['both']['realistic']['hits_at_3'])
                    results_dict['hits@10'].append(results['metrics']['both']['realistic']['hits_at_10'])

                    results_dict['learning-rate'].append(config['optimizer_kwargs'].split(',')[0].split(':')[1])

                    model_dims = re.findall(pattern1, config['model_kwargs'])
                    # model_dims = ['14505, 200', '237, 200']
                    results_dict['ent-dim'].append(int(model_dims[0].split(',')[1].strip()))
                    results_dict['rel-dim'].append(int(model_dims[1].split(',')[1].strip()))
                    try:
                        results_dict['type-dim'].append(int(model_dims[2].split(',')[1].strip()))
                    except:
                        results_dict['type-dim'].append('-')

                    if 'num_negs_per_pos' in config:
                        results_dict['num_negs_per_pos'].append(config['num_negs_per_pos'])
                    else:
                        results_dict['num_negs_per_pos'].append('-')

                    results_dict['batch-size'].append(config['batch_size'])
                    if 'type_smoothing' in config:
                        results_dict['type-smoothing'].append(config['type_smoothing'])
                    else:
                        results_dict['type-smoothing'].append('-')
                    if 'margin' in config['loss_kwargs']:
                        margin = re.search(r"'margin':\s*([\d\.]+)", config['loss_kwargs']).group(1)
                        results_dict['margin'].append(margin)
                    else:
                        results_dict['margin'].append('-')
                    
                    # {'training': False, '_parameters': OrderedDict(), '_buffers': OrderedDict(), '_non_persistent_buffers_set': set(), '_backward_pre_hooks': OrderedDict(), '_backward_hooks': OrderedDict(), '_is_full_backward_hook': None, '_forward_hooks': OrderedDict(), '_forward_hooks_with_kwargs': OrderedDict(), '_forward_pre_hooks': OrderedDict(), '_forward_pre_hooks_with_kwargs': OrderedDict(), '_state_dict_hooks': OrderedDict(), '_state_dict_pre_hooks': OrderedDict(), '_load_state_dict_pre_hooks': OrderedDict(), '_load_state_dict_post_hooks': OrderedDict(), '_modules': OrderedDict(), 'reduction': 'mean', '_reduction_method': <built-in method mean of type object at 0x7f2261842540>, 'inverse_softmax_temperature': 1.0, 'factor': 0.5, 'margin': 8.0}
                    if "ETC" in model_name or "NNY" in model_name:
                        if 'type_weight_temperature' in config:
                            results_dict['type_weight_temperature'].append(config['type_weight_temperature'])
                        else:
                            results_dict['type_weight_temperature'].append(str(1.0))
                    else:
                        results_dict['type_weight_temperature'].append('-')
                    if 'inverse_softmax_temperature' in config['loss_kwargs']:
                        adversarial_temperature = re.search(r"'inverse_softmax_temperature':\s*([\d\.]+)", config['loss_kwargs']).group(1)
                        results_dict['adversarial-temperature'].append(adversarial_temperature)
                    else:
                        results_dict['adversarial-temperature'].append('-')
                    if 'type_score_weight' in config:
                        results_dict['type_score_weight'].append(config['type_score_weight'])
                    else:
                        results_dict['type_score_weight'].append('-')

                    results_dict['description'].append(file_name)
                    results_dict['dataset'].append(dataset)
                    results_dict['model-size'].append(float(config['num_parameter_bytes'][:-2])*0.125*0.25)
                    results_dict['train-loop'].append(config['training_loop'])
                    results_dict['optimizer'].append(config['optimizer'])
                    if 'SLCWATrainingLoop' == config['training_loop']:
                        results_dict['negative_sampler'].append(config['negative_sampler'])
                    else:
                        results_dict['negative_sampler'].append('-')
                    if 'PreTrain' in file_name:
                        if 'pre_trained_type_name' in config:
                            results_dict['pre_trained_type_name'].append(config['pre_trained_type_name'])
                        else:
                            results_dict['pre_trained_type_name'].append('bert-base-uncased')
                    else:
                        results_dict['pre_trained_type_name'].append('-')

                    if 'loss' in config:
                        results_dict['loss'].append(config['loss'])
                        # results_dict['loss_kwargs'].append(config['loss_kwargs'])
                    else:
                        results_dict['loss'].append('see config')
                        # results_dict['loss_kwargs'].append(config['loss_kwargs'])

r = pd.DataFrame(results_dict)

r.to_csv(save_path, index=False)

        
