#!python
# script for processing biomodels biopax data

import os
import sys
import glob
import tqdm
import jsonlines

from pathhier.pathway_kb_loader import PathwayKBLoader
from pathhier.pathway import PathKB
from pathhier.paths import PathhierPaths


path_kb_loader = PathwayKBLoader()
paths = PathhierPaths()

# process biomodels
kb_name = "biomodels"
kb_dir = "/Users/lwang/git/biomodels/data/"
output_file = "/Users/lwang/git/biomodels/output/biomodels.jsonlines"

kb_files = glob.glob(kb_dir + "*.owl")
pathways = []

for file_path in tqdm.tqdm(kb_files):
    file_name, file_ext = os.path.splitext(file_path.split('/')[-1])
    kb = PathKB(kb_name, file_path)
    pathways += kb.load(file_path)

with jsonlines.open(output_file, mode='w') as writer:
    for p in pathways:
        p_dict = {
            "uid": p.uid,
            "name": p.name,
            "aliases": p.aliases,
            "xrefs": p.xrefs,
            "definition": p.definition,
            "comments": p.comments,
            "entities": [ent.to_json() for ent in p.entities],
            "relations": p.relations,
            "provenance": "biomodels"
        }
        writer.write(p_dict)

