'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-05-23 14:24:50
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-06-29 10:00:49
FilePath: /ESETC/code/test.py
Description: 测试1-1，1-n，n-1，n-n的结果。测试不同种类关系的结果。

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
from collections import defaultdict

import pandas as pd
import pykeen
import torch
from Custom.ir_evaluation import IRRankBasedEvaluator
from pykeen.datasets import FB15k237
from pykeen.datasets import analysis as ana
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from utilities import load_dataset


def get_relation_cardinality_dict(dataset: pykeen.datasets):

    relation_cardinality_df = ana.get_relation_cardinality_types_df(dataset=dataset)
    groups = relation_cardinality_df.groupby(relation_cardinality_df.relation_id)
    relation_cardinality_dict = {'one-to-many':[],'one-to-one':[],'many-to-many':[],'many-to-one':[]}
    for i in range(len(groups)):
        """
        where there are many2many, this realtion is many2many
        if no many2many, relation type is many2one or one2many or one2one.
        """
        relation_instance = groups.get_group(i)
        
        relation_cardinality_type = 'many-to-many'
        if relation_instance.shape[0] < 4:
            if 'many-to-many' in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = 'many-to-many'
            elif 'many-to-one' in list(relation_instance.iloc[:, 1]) and 'many-to-many' not in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = 'many-to-one'
            elif 'one-to-many' in list(relation_instance.iloc[:, 1]) and 'many-to-many' not in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = 'one-to-many'
            elif 'many-to-many' not in list(relation_instance.iloc[:, 1]) and 'many-to-one' not in list(relation_instance.iloc[:, 1]) and 'one-to-many' not in list(relation_instance.iloc[:, 1]):
                relation_cardinality_type = 'one-to-one'
        
        relation_cardinality_dict[relation_cardinality_type].append(i)
    
    return relation_cardinality_dict

def relation_cardinality_type_result(model, dataset, evaluator, relation_set, head_or_tail = 'head', cardinality_type = 'one-to-one'):
    test_data = dataset.testing.new_with_restriction(relations=relation_set[cardinality_type])

    results = evaluator.evaluate(
        batch_size=8,
        model=model,
        mapped_triples=test_data.mapped_triples,
        additional_filter_triples=[dataset.training.mapped_triples,dataset.validation.mapped_triples, dataset.testing.mapped_triples],
    )

    result_dic = dict()
    result_dic['data_type'] = cardinality_type
    result_dic['target'] = head_or_tail
    results = results.to_dict()[head_or_tail]['realistic']
    result_dic['mrr'] = results['inverse_harmonic_mean_rank']
    result_dic['h@10'] = results['hits_at_10']
    result_dic['h@3'] = results['hits_at_3']
    result_dic['h@1'] = results['hits_at_1']
    
    return result_dic

def load_model(default_dataset_name, default_description, default_model_date, default_model_name):
    # Load the model
    dataset_name = input("Please input the name of the dataset:")
    if dataset_name == '':
        dataset_name = default_dataset_name
    print("The dataset name is: ", dataset_name)

    description = input("Please input the description of this model:")
    if description == '':
        description = default_description
    print("The description of this model is: ", description)

    model_name = input("Please input the name of this model:")
    if model_name == '':
        model_name = default_model_name
    print("The name of this model is: ", model_name)

    model_date = input("Please input the date of this model:")
    if model_date == '':
        model_date = default_model_date
    print("The date of this model is: ", model_date)
    
    
    model_path = '../models/' + dataset_name + '/' + description + '/' + model_name + '/' + model_date + '/' + 'trained_model.pkl'
    trained_model = torch.load(model_path) 

    return trained_model

def get_result_dict(model, evaluator, relation_set, dataset, ir_evaluator=None):
    result_dict = defaultdict(list)
    for ht in ['head', 'tail']:
        for m in ['one-to-one', 'one-to-many', 'many-to-one', 'many-to-many']:
            # print(relation_cardinality_type_result(model, dataset=dataset, evaluator=evaluator,relation_set=relation_set, head_or_tail=ht, cardinality_type=m))
            result = relation_cardinality_type_result(model, dataset=dataset, evaluator=evaluator,relation_set=relation_set, head_or_tail=ht, cardinality_type=m)
            result_dict['target'].append(ht)
            result_dict['cardinality_type'].append(m)
            result_dict['mrr'].append(result['mrr'])
            result_dict['h@10'].append(result['h@10'])
            result_dict['h@3'].append(result['h@3'])
            result_dict['h@1'].append(result['h@1'])
            if ir_evaluator:
                result = relation_cardinality_type_result(model, dataset=dataset, evaluator=ir_evaluator,relation_set=relation_set, head_or_tail=ht, cardinality_type=m)
                result_dict['ir_mrr'].append(result['mrr'])
                result_dict['ir_h@10'].append(result['h@10'])
                result_dict['ir_h@3'].append(result['h@3'])
                result_dict['ir_h@1'].append(result['h@1'])

    return result_dict



if __name__ == "__main__":
    
    dataset_name = 'CAKE-DBpedia-242'
    description = 'noDescription'
    model_name = 'CatESETCwithRotate'
    model_date = '20230427-161537'
    
    

    training_data, validation, testing = load_dataset(dataset=dataset_name)
    dataset = get_dataset(training=training_data, testing=testing, validation=validation)

    relation_set = get_relation_cardinality_dict(dataset=dataset)
    print(dataset_name)
    for m in ['one-to-one', 'one-to-many', 'many-to-one', 'many-to-many']:
        test_data = dataset.testing.new_with_restriction(relations=relation_set[m])
        print(m, ':', test_data.mapped_triples.shape[0])

    trained_model = load_model(dataset_name, description, model_date, model_name)

    # 统计每种类别的关系的数据量。


    evaluator = RankBasedEvaluator()
    ir_evaluator = IRRankBasedEvaluator()
    # r = evaluator.evaluate(
    #         model=trained_model,
    #         mapped_triples=dataset.testing.mapped_triples,
    #         additional_filter_triples=[dataset.training.mapped_triples,dataset.validation.mapped_triples, dataset.testing.mapped_triples],
    #     )
    
    result_dic = get_result_dict(trained_model, evaluator, relation_set, dataset, ir_evaluator=ir_evaluator)
    result_df = pd.DataFrame(result_dic)

    save_path = '../result/cardinality/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_df.to_csv(save_path + '%s.csv'%(dataset_name.replace('-', '')+'-'+description+'-'+model_name+'-'+model_date), index=False)