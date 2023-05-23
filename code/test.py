'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2023-05-23 14:24:50
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-05-23 15:35:00
FilePath: /code/test.py
Description: 测试1-1，1-n，n-1，n-n的结果。测试不同种类关系的结果。

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import pykeen
import torch
from pykeen.datasets import FB15k237
from pykeen.datasets import analysis as ana
from pykeen.evaluation import RankBasedEvaluator

d = FB15k237(create_inverse_triples=True)
data_analysis = ana.get_relation_cardinality_types_df(dataset=d)


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

def relation_cardinality_type_result(model, evaluator, relation_set, head_or_tail = 'head', cardinality_type = 'one-to-one'):
    test_data = d.testing.new_with_restriction(relations=relation_set)

    results = evaluator.evaluate(
        batch_size=8,
        model=model,
        mapped_triples=test_data.mapped_triples,
        additional_filter_triples=[d.training.mapped_triples,d.validation.mapped_triples, d.testing.mapped_triples],
    )

    result_dic = dict()
    result_dic['data_type'] = cardinality_type
    result_dic['target'] = head_or_tail
    results = results.to_dict()[head_or_tail]['realistic']
    result_dic['amrr'] = results['inverse_harmonic_mean_rank']
    result_dic['ah@10'] = results['hits_at_k']
    
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


if __name__ == "__main__":
    
    dataset_name = 'CAKE-FB15K237'
    description = 'noDescription'
    model_name = 'CatESETCwithRotate'
    model_date = '20230427-161537'
    
    trained_model = load_model(dataset_name, description, model_date, model_name)
