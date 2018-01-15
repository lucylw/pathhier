#!~/anaconda3/envs/pathhier/bin/python
# script for aligning pathway kb ontologies to the PW

import os

from pathhier.pathway_kb_loader import PathwayKBLoader
from pathhier.pathway import PathKB
from pathhier.paths import PathhierPaths


paths = PathhierPaths()
path_kb_loader = PathwayKBLoader()

path_kb_loader.load_id_dict()
path_kb_loader.load_pw()

reactome_path = os.path.join(paths.processed_data_dir, 'kb_{}.pickle'.format('reactome'))
assert os.path.exists(reactome_path)

kb = PathKB("reactome", reactome_path)
kb = kb.load_pickle('reactome', reactome_path)
