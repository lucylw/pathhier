#!python
# script for getting Reactome hierarchy

import os
import tqdm
import json
from collections import defaultdict

from rdflib import Graph
from rdflib import Namespace
from rdflib.namespace import RDF

from pathhier.paths import PathhierPaths
from pathhier.reactome_ontology import ReactomeOntology


BP3 = Namespace("http://www.biopax.org/release/biopax-level3.owl#")
paths = PathhierPaths()

reactome_raw_data_path = os.path.join(
    paths.raw_data_dir, "reactome", "Reactome_v59_Homo_sapiens.owl"
)

assert os.path.exists(reactome_raw_data_path)

reactome = ReactomeOntology(name="Reactome",
                            filename=reactome_raw_data_path)
reactome.load_from_file()

reactome_dict = dict()

for pw in reactome.graph.subjects(RDF.type, BP3['Pathway']):
    reactome_dict[pw] = {
        'name': reactome.get_label(pw),
        'aliases': reactome.get_all_labels(pw),
        'synonyms': reactome.get_synonyms(pw),
        'definition': reactome.get_definition(pw),
        'subClassOf': reactome.get_subClassOf(pw),
        'part_of': reactome.get_part_of(pw)
    }

output_file = os.path.join(paths.output_dir, "reactome_ontology.json")
with open(output_file, 'w') as outf:
    json.dump(reactome_dict, outf, indent=4, sort_keys=True)