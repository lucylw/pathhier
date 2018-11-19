#!python
# script for getting HumanCyc hierarchy

import os
import json
import pickle
import re

from rdflib import Namespace

from pathhier.paths import PathhierPaths
from pathhier.biocyc_ontology import BiocycOntology


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def remove_chars(text):
    return text.replace('&alpha;', 'alpha').replace('&beta;', 'beta')


def process_str(text):
    return remove_chars(remove_tags(text))


def get_class_id(cls_names, classes):
    """
    Get list of UIDs matching input class names
    :param cls_names:
    :param classes:
    :return:
    """
    cls_uids = []
    for cls_name in cls_names:
        lower_name = cls_name.lower()

        matches = [
            uid for uid, info in classes.items()
            if lower_name == info['name'].lower()
            or lower_name in [a.lower() for a in info['aliases']]
        ]

        if matches:
            cls_uids.append(matches[0])
        else:
            cls_uids.append(cls_name)

    return cls_uids


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

humancyc_classes = dict()

for pw in humancyc_pathways:
    db, uid = pw.uid.split(':')

    humancyc_id = uid
    xrefs = pw.xrefs
    for xr in xrefs:
        if xr.startswith('HumanCyc'):
            humancyc_id = xr.split(':')[1]

    if humancyc_id in biocyc.pw_classes:
        humancyc_classes[humancyc_id] = biocyc.pw_classes[humancyc_id]
        if type(biocyc.pw_classes[humancyc_id]['subClassOf']) == str:
            humancyc_classes[humancyc_id]['subClassOf'] = [biocyc.pw_classes[humancyc_id]['subClassOf']]
    else:
        humancyc_classes[humancyc_id] = {}
        humancyc_classes[humancyc_id]['subClassOf'] = []
        humancyc_classes[humancyc_id]['part_of'] = []
        humancyc_classes[humancyc_id]['instances'] = []
        humancyc_classes[humancyc_id]['synonyms'] = []

    humancyc_classes[humancyc_id]['name'] = process_str(pw.name)
    humancyc_classes[humancyc_id]['aliases'] = [process_str(a) for a in pw.aliases]
    humancyc_classes[humancyc_id]['definition'] = process_str(pw.definition)


biocyc_dict = dict()

for pw, info in humancyc_classes.items():
    biocyc_dict[pw] = {
        'name': info['name'],
        'aliases': info['aliases'],
        'synonyms': info['synonyms'],
        'definition': info['definition'],
        'subClassOf': get_class_id(info['subClassOf'], humancyc_classes),
        'part_of': get_class_id(info['part_of'], humancyc_classes),
        'instances': info['instances']
    }

output_file = os.path.join(paths.processed_data_dir, "humancyc_ontology.json")
with open(output_file, 'w') as outf:
    json.dump(biocyc_dict, outf, indent=4, sort_keys=True)
