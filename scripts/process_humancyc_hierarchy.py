#!python
# script for getting HumanCyc hierarchy

import os
import json

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

assert os.path.exists(biocyc_classes_file)
assert os.path.exists(biocyc_pathways_file)

biocyc = BiocycOntology(name="BioCyc",
                        filename=biocyc_classes_file,
                        pathway_file=biocyc_pathways_file)
biocyc.load_from_file()

output_file = os.path.join(paths.output_dir, "biocyc_ontology.json")
with open(output_file, 'w') as outf:
    json.dump(biocyc.pw_classes, outf, indent=4, sort_keys=True)
