
###
 # @Author: error: git config user.name && git config user.email & please set dead value or install git
 # @Date: 2023-01-02 22:11:33
 # @LastEditors: Ni Runyu ni-runyu@ed.tmu.ac.jp
 # @LastEditTime: 2023-02-17 12:13:17
 # @FilePath: /undefined/home/ni/code/ESETC/code/train.sh
 # @Description: 
 # 
 # Copyright (c) 2023 by error: git config user.name && git config user.email & please set dead value or install git, All Rights Reserved. 
###


python TrainModel.py -m 0 -reverse -lm 24 -at 1.0 -nen 400 -lr 0.0001 -b 512 -e 1000 -med 500 -mrd 500 -mtd 500 -train slcwa -eb 1 -d CAKE-DBpedia-242
python TrainModel.py -m 0 -reverse -lm 8 -at 0.5 -nen 400 -lr 0.0001 -b 256 -e 1000 -med 500 -mrd 500 -mtd 500 -train slcwa -eb 1 -d CAKE-NELL-995

python TrainModel.py -m 1 -reverse -lm 24 -at 1.0 -nen 400 -lr 0.0001 -b 512 -e 1000 -med 500 -mrd 500 -mtd 500 -train slcwa -eb 1 -d CAKE-DBpedia-242
python TrainModel.py -m 1 -reverse -lm 8 -at 0.5 -nen 400 -lr 0.0001 -b 256 -e 1000 -med 500 -mrd 500 -mtd 500 -train slcwa -eb 1 -d CAKE-NELL-995





