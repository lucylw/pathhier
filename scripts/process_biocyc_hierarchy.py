#!python
# script for getting HumanCyc hierarchy

import os
import json
import pickle

from rdflib import Namespace

from pathhier.paths import PathhierPaths
from pathhier.biocyc_ontology import BiocycOntology


BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")
paths = PathhierPaths()

biocyc_classes_file = os.path.join(
    paths.other_data_dir,
    'biocyc_classes.dat'
)

biocyc_pathways_file = os.path.join(
    paths.other_data_dir,
    'biocyc_pathways.dat'
)

humancyc_data_file = os.path.join(
    paths.processed_data_dir,
    'kb_humancyc.pickle'
)

assert os.path.exists(biocyc_classes_file)
assert os.path.exists(biocyc_pathways_file)
assert os.path.exists(humancyc_data_file)

biocyc = BiocycOntology(name="BioCyc",
                        filename=biocyc_classes_file,
                        pathway_file=biocyc_pathways_file)
biocyc.load_from_file()

humancyc_pathways = pickle.load(open(humancyc_data_file, 'rb'))

for pw in humancyc_pathways:
    db, uid = pw.uid.split(':')
    if uid.startswith('Pathway'):
        uid = uid[len('Pathway'):]
    uid = 'PWY-{}'.format(uid)
    if uid in biocyc.pw_classes:
        biocyc.pw_classes[uid]['name'] = pw.name
        biocyc.pw_classes[uid]['aliases'] = pw.aliases
        biocyc.pw_classes[uid]['synonyms'] = pw.xrefs
        biocyc.pw_classes[uid]['definition'] = pw.definition

biocyc_dict = dict()

for pw, info in biocyc.pw_classes.items():
    biocyc_dict[pw] = {
        'name': pw,
        'aliases': info['aliases'],
        'synonyms': info['synonyms'],
        'definition': info['definition'],
        'subClassOf': info['subClassOf'],
        'part_of': info['part_of'],
        'instances': info['instances']
    }

output_file = os.path.join(paths.output_dir, "biocyc_ontology.json")
with open(output_file, 'w') as outf:
    json.dump(biocyc_dict, outf, indent=4, sort_keys=True)
