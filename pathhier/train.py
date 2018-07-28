import os
import json

from allennlp.commands.train import train_model_from_file

from torch.cuda import device

from pathhier.nn.pathway_dataset_reader import PathwayDatasetReader
from pathhier.nn.pathway_model import PWAlignNN

config_file = '/Users/lwang/git/pathhier/config/model.json'
model_path = '/Users/lwang/git/pathhier/model/nn_model'

assert os.path.exists(config_file)

with open(config_file) as json_data:
    configuration = json.load(json_data)

cuda_device = configuration['trainer']['cuda_device']

if cuda_device >= 0:
    with device(cuda_device):
        train_model_from_file(config_file, model_path)
else:
    train_model_from_file(config_file, model_path)