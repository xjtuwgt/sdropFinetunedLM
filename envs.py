import os
import logging
import sys
from os.path import join

# Add submodule path into import paths
# is there a better way to handle the sub module path append problem?
PROJECT_FOLDER = os.path.dirname(__file__)
print('Project folder = {}'.format(PROJECT_FOLDER))
sys.path.append(join(PROJECT_FOLDER))

# Define the dataset folder and model folder based on environment
# HOME_DATA_FOLDER = '/ssd/HGN/data'
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
DATASET_FOLDER = join(HOME_DATA_FOLDER, 'dataset')
MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
KNOWLEDGE_FOLDER = join(HOME_DATA_FOLDER, 'knowledge')
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'outputs')
PRETRAINED_MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models/pretrained')
print('*' * 35, ' path infor ', '*' * 35)
print('data folder = {}'.format(HOME_DATA_FOLDER))
print('hotpotqa data folder = {}'.format(DATASET_FOLDER))
print('pretrained model folder = {}'.format(MODEL_FOLDER))
print('knowledge folder = {}'.format(KNOWLEDGE_FOLDER))
print('output result folder = {}'.format(OUTPUT_FOLDER))
print('pretrained model with finetuned folder = {}'.format(PRETRAINED_MODEL_FOLDER))
print('*' * 85)
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = join(HOME_DATA_FOLDER, 'models', 'pretrained_cache')