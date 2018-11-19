#!python
# script for getting Reactome hierarchy

import os
import tqdm
import json
import glob
from rdflib import Namespace
from rdflib.namespace import RDF

from pathhier.paths import PathhierPaths
from pathhier.reactome_ontology import ReactomeOntology


BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")
paths = PathhierPaths()

reactome_raw_data_paths = glob.glob(
    os.path.join(
        paths.raw_data_dir, "reactome", "*.owl"
    ))

if reactome_raw_data_paths:
    reactome_raw_data_path = reactome_raw_data_paths[0]

assert os.path.exists(reactome_raw_data_path)

reactome = ReactomeOntology(name="Reactome",
                            filename=reactome_raw_data_path)
reactome.load_from_file()

reactome_dict = dict()

for pw in tqdm.tqdm(reactome.graph.subjects(RDF.type, BP3['Pathway'])):
    reactome_xrefs = reactome.get_xrefs(pw)
    reactome_ids = [x for x in reactome_xrefs if x.startswith('Reactome:')]
    if reactome_ids:
        reactome_dict[reactome_ids[0]] = {
            'name': reactome.get_label(pw),
            'aliases': reactome.get_all_labels(pw),
            'synonyms': reactome.get_synonyms(pw),
            'definition': reactome.get_definition(pw),
            'subClassOf': reactome.get_subClassOf(pw),
            'part_of': reactome.get_part_of(pw),
            'instances': [pw]
        }
    else:
        print('No id for {}'.format(pw))

output_file = os.path.join(paths.processed_data_dir, "reactome_ontology.json")
with open(output_file, 'w') as outf:
    json.dump(reactome_dict, outf, indent=4, sort_keys=True)