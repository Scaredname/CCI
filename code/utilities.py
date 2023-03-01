'''
Author: Ni Runyu ni-runyu@ed.tmu.ac.jp
Date: 2022-12-22 12:02:34
LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
LastEditTime: 2023-03-01 15:32:15
FilePath: /ESETC/code/utilities.py
Description: 

Copyright (c) 2023 by Ni Runyu ni-runyu@ed.tmu.ac.jp, All Rights Reserved. 
'''
import os
from collections import defaultdict

from Custom.CustomTripleFactory import TriplesTypesFactory
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory


def get_key(dict, va):
    return [k for k,v in dict.item() if v == va]

def get_white_list_relation(dataset, type_position = 0):
        """
        
        @Return: white_list_relation(relation unlike type relaiton), rel_need(relation like type relation) 
        """
        data = dataset
        # 分为两部分来统计出入度是因为我们需要判断的是和这个关系共现的某一边的所有实体，而不是两边的实体。
        rel_h_entdegree = defaultdict(dict)
        rel_t_entdegree = defaultdict(dict)
        ent_relnum = defaultdict(list)
        rel_need = list()


        for triple in data.mapped_triples:
                if int(triple[0]) not in rel_h_entdegree[int(triple[1])]:
                        rel_h_entdegree[int(triple[1])][int(triple[0])] = [1,0]
                else:
                        rel_h_entdegree[int(triple[1])][int(triple[0])][0] += 1
                if int(triple[2]) not in rel_t_entdegree[int(triple[1])]:
                        rel_t_entdegree[int(triple[1])][int(triple[2])] = [0,1]
                else:
                        rel_t_entdegree[int(triple[1])][int(triple[2])][1] += 1
                
                ent_relnum[int(triple[0])].append(int(triple[1]))
                ent_relnum[int(triple[2])].append(int(triple[1]))
        
        # 合并实体出入度
        for rel in rel_h_entdegree:
                for ent in rel_h_entdegree[rel]:
                        if ent in rel_t_entdegree[rel]:
                                rel_h_entdegree[rel][ent][1] = rel_t_entdegree[rel][ent][1]
                
                for ent in rel_t_entdegree[rel]:
                        if ent in rel_h_entdegree[rel]:
                                rel_t_entdegree[rel][ent][0] = rel_h_entdegree[rel][ent][0]

        for ent in ent_relnum:
                ent_relnum[ent] = len(set(ent_relnum[ent]))

        if type_position == 0:     
                for rel in rel_h_entdegree:
                        need = True
                        
                        for ent in rel_h_entdegree[rel]:
                                # 如果该关系所共现所有实体中存在实体的关系数目不为1，那么该关系不是type关系
                                if ent_relnum[ent] != 1:
                                        need = False
                                        break
                                # 如果该关系所共现所有实体中存在实体的出入度都不为0，那么该关系不是type关系
                                if rel_h_entdegree[rel][ent][0] != 0 and rel_h_entdegree[rel][ent][1] != 0: 
                                        need = False
                                        break
                        if need:
                                rel_need.append(data.relation_id_to_label[rel])

        elif type_position == 2:
                for rel in rel_t_entdegree:
                        need = True
                        for ent in rel_t_entdegree[rel]:
                                if ent_relnum[ent] != 1:
                                        need = False
                                        break
                                if rel_t_entdegree[rel][ent][0] != 0 and rel_t_entdegree[rel][ent][1] != 0: 
                                        need = False
                                        break             
                        if need:
                                rel_need.append(data.relation_id_to_label[rel])

        white_list_rel = list(data.relation_to_id.keys())
        for rel in rel_need:
                white_list_rel.remove(rel)

        print('type_postion:', type_position)
        print('num_type_like_relation: ', len(rel_need))

        return white_list_rel, rel_need              

def readTypeData(data_name, data_pro_func, create_inverse_triples=False, type_position=0, hasNoneType=False):
        """
        @Params: data_name, data_pro_func, create_inverse_triples, type_position
        @Return: Train, Test, Valid
        """
        if 'CAKE' in data_name:
                data_name = 'data_concept/' + data_name.replace('CAKE-', '')

        train_path = os.path.join('../data/%s/'%(data_name), 'train_type.txt')
        if hasNoneType:
                train_path = os.path.join('../data/%s/'%(data_name), 'train_type_a.txt')
        valid_path = os.path.join('../data/%s/'%(data_name), 'valid.txt') 
        test_path = os.path.join('../data/%s/'%(data_name), 'test.txt')

        training = TriplesFactory.from_path(
            train_path,
            create_inverse_triples=create_inverse_triples,)

        training_triples, training_type_triples, _, _ = data_pro_func(training, type_position=type_position)
        training_data = TriplesTypesFactory.from_labeled_triples(triples=training_triples, type_triples=training_type_triples, type_position=type_position, create_inverse_triples=create_inverse_triples)

        validation = TriplesFactory.from_path(
            valid_path, 
            entity_to_id=training_data.entity_to_id, 
            relation_to_id=training_data.relation_to_id,
            create_inverse_triples=create_inverse_triples,)
        testing = TriplesFactory.from_path(
                test_path, 
                entity_to_id=training_data.entity_to_id, 
                relation_to_id=training_data.relation_to_id,
                create_inverse_triples=create_inverse_triples,)

        return training_data, validation, testing








