#!python

import os
import json
import csv
import random

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB
from pathhier.candidate_selector import CandidateSelector


def parse_kegg_paths(path):
    """
    Parse all KEGG pathways from file
    :param path:
    :return:
    """
    pathways = []

    with open(path, 'r') as f:
        contents = f.readlines()

    uid = None

    for line in contents:
        line = line.strip()
        if line:
            if '.' in line:
                pass
            else:
                if line.isdigit():
                    uid = 'KEGG:' + line
                else:
                    pathways.append((uid, line))
                    uid = None

    return pathways


def parse_smpdb_paths(path):
    """
    Parse all SMPDB pathways from file
    :param paths:
    :return:
    """
    pathways = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for uid, name, _, definition in reader:
            pathways.append(
                ('SMP:' + uid[3:], name, definition.replace('\n', ' '))
            )

    return pathways


def kb_pickle_to_json(kb: PathKB):
    """
    Convert PathKB to json
    :param kb:
    :return:
    """
    kb_dict = dict()

    for p in kb.pathways:
        rels = p.relations
        subclass_of = [obj.uid for prop, obj in rels if prop == 'subClassOf']
        part_of = [obj.uid for prop, obj in rels if prop == 'part_of']
        kb_dict[p.uid] = {
            'name': p.name,
            'aliases': p.aliases,
            'synonyms': [],
            'definition': p.definition,
            'subClassOf': subclass_of,
            'part_of': part_of,
            'instances': [p.uid]
        }

    return kb_dict


# initialize
paths = PathhierPaths()

xref_dict = dict()
not_found = list()
training_data = []

# parse kegg and smpdb pathway name files
kegg_file = os.path.join(paths.raw_data_dir, 'kegg_paths')
smpdb_file = os.path.join(paths.raw_data_dir, 'smpdb_paths')

kegg_paths = parse_kegg_paths(kegg_file)
smpdb_paths = parse_smpdb_paths(smpdb_file)

for uid, name in kegg_paths:
    xref_dict[uid] = (name, "")

for uid, name, definition in smpdb_paths:
    xref_dict[uid] = (name, definition)

# load KEGG and SMPDB KBs
kegg_kb_path = os.path.join(paths.processed_data_dir, 'kegg_ontology.json')
smpdb_kb_path = os.path.join(paths.processed_data_dir, 'smpdb_ontology.json')

with open(kegg_kb_path, 'r') as f:
    kegg = json.load(f)

with open(smpdb_kb_path, 'r') as f:
    smpdb = json.load(f)

# load pathway ontology
pw_file = os.path.join(paths.pathway_ontology_dir, 'pw.json')
assert os.path.exists(pw_file)

with open(pw_file, 'r') as f:
    pw = json.load(f)

# create candidate selectors
kegg_cand_sel = CandidateSelector(pw, kegg)
smpdb_cand_sel = CandidateSelector(pw, smpdb)

# iterate through PW and extract xrefs
for pw_id, pw_value in pw.items():

    pw_name = pw_value['name']
    pw_def = ''
    if pw_value['definition']:
        pw_def = pw_value['definition'][0]
    xrefs = pw_value['synonyms']

    for xref in xrefs:
        if 'KEGG:' in xref or 'SMP:' in xref:

            negatives = list()

            # get xref db and id, clean db string, format new ids, and select negatives
            xref_db, xref_id = xref.split(':')
            kb_id = ''

            if 'KEGG' in xref_db:
                xref_db = 'KEGG'
                kb_id = 'kegg:hsa' + xref_id
                negatives = kegg_cand_sel.select(pw_id)[5:]
            if 'SMP' in xref_db:
                xref_db = 'SMP'
                kb_id = 'SMP' + xref_id
                negatives = smpdb_cand_sel.select(pw_id)[5:]

            new_id = '{}:{}'.format(xref_db, xref_id)

            # sample hard negatives
            for neg_id, _ in negatives:
                if neg_id != kb_id:
                    ent = None
                    ent_name = ''
                    ent_def = ''
                    if xref_db == 'KEGG' and 'hsa' in neg_id:
                        ent = kegg[neg_id]
                        ent_name = ent['name']
                        ent_def = ent['definition']
                    elif xref_db == 'SMP':
                        ent = smpdb[neg_id]
                        ent_name = ent['name']
                        ent_def = xref_dict['SMP:' + neg_id[3:]][1]
                    if ent:
                        training_data.append(('0', pw_id, pw_name, pw_def, neg_id, ent_name, ent_def))
                        break

            # sample easy negatives
            if xref_db == 'KEGG':
                easy_neg = random.sample(kegg.keys(), 1)[0]
                ent = kegg[easy_neg]
                if easy_neg != kb_id:
                    training_data.append(
                        ('0', pw_id, pw_name, pw_def, easy_neg, ent['name'], ent['definition'])
                    )
            elif xref_db == 'SMP':
                easy_neg = random.sample(smpdb.keys(), 1)[0]
                ent = smpdb[easy_neg]
                if easy_neg != kb_id:
                    ent_def = xref_dict['SMP:' + easy_neg[3:]][1]
                    training_data.append(('0', pw_id, pw_name, pw_def, easy_neg, ent['name'], ent_def))

            # add name and def from xref dictionary if available
            if new_id in xref_dict:
                xref_name, xref_def = xref_dict[new_id]
                training_data.append(('1', pw_id, pw_name, pw_def, kb_id, xref_name, xref_def))
            else:
                training_data.append(('1', pw_id, pw_name, pw_def, kb_id, "", ""))

        # else append to not found list (PID identifiers etc)
        else:
            not_found.append(xref)

# write extracted xrefs to file
output_file = os.path.join(paths.processed_data_dir, 'training_data.tsv')

with open(output_file, 'w') as outf:
    outf.write('Match\tPW_id\tPW_name\tPW_def\txref_id\txref_name\txref_def\n')
    for training_line in training_data:
        outf.write('\t'.join(training_line) + '\n')

# write missing identifiers to file
not_found_file = os.path.join(paths.processed_data_dir, 'training_not_found.txt')

with open(not_found_file, 'w') as outf:
    for xref in not_found:
        outf.write(xref + '\n')
