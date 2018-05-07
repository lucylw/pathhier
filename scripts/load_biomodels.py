#!python
# script for processing biomodels biopax data

import os
import sys
import jsonlines

from pathhier.pathway_kb_loader import PathwayKBLoader
from pathhier.pathway import PathKB
from pathhier.paths import PathhierPaths


path_kb_loader = PathwayKBLoader()
paths = PathhierPaths()

# process biomodels
kb_name = "biomodels"
kb_path = os.path.join(paths.other_data_dir, "BIOMD0000000015-biopax3.owl")
output_path = os.path.join("/Users/lwang/git/biomodels/output/", "BIOMD0000000015_rxs.json")

sys.stdout.write('\nLoading %s\n' % kb_name)
sys.stdout.write('\t %s\n' % kb_path)
kb = PathKB(kb_name, kb_path)
pathways = kb.load(kb_path)

with jsonlines.open(output_path, mode='w') as writer:
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

