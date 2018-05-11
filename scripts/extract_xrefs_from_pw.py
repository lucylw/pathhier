#!python

import os
import json
import pickle
from datetime import date

from pathhier.paths import PathhierPaths
from pathhier.candidate_selector import CandidateSelector


xref_kbs = ['kegg', 'smpdb', 'pid']
xref_dict = dict()
training_data = []

paths = PathhierPaths()

# load xref kbs
for kb_name in xref_kbs:
    kb_file = os.path.join(paths.processed_data_dir, 'kb_' + kb_name + '.pickle')
    assert os.path.exists(kb_file)
    kb = pickle.load(open(kb_file, 'rb'))

    if kb_name == 'kegg':
        for p in kb:
            _, uid = p.uid.split(':')
            if not(uid.startswith('#')):
                if uid.startswith('hsa'):
                    uid = uid[3:]
                p_def = p.definition
                if p.comments:
                    if not(p.comments[0].startswith('REPLACED')):
                        p_def = p.comments[0]
                xref_dict['KEGG:{}'.format(int(uid))] = (p.name, p_def)
    elif kb_name == 'smpdb':
        for p in kb:
            if ':' in p.uid:
                _, uid = p.uid.split(':')
            else:
                uid = p.uid
            xref_dict['SMP:{}'.format(int(uid[3:]))] = (p.name, p.definition)
    elif kb_name == 'pid':
        for p in kb:
            if p.comments:
                comment = p.comments[0]
                if comment.startswith('REPLACED'):
                    _, uid_long = comment.split('#')
                    _, uid = uid_long.split('_')
                    xref_dict['PID:{}'.format(int(uid))] = (p.name, p.definition)

# load pathway ontology
pw_file = os.path.join(paths.pathway_ontology_dir, 'pw.json')
assert os.path.exists(pw_file)

with open(pw_file, 'r') as f:
    pw = json.load(f)

for pw_id, pw_value in pw.items():
    pw_name = pw_value['name']
    pw_def = ''
    if pw_value['definition']:
        pw_def = pw_value['definition'][0]
    xrefs = pw_value['synonyms']
    for xref in xrefs:
        if xref.startswith('KEGG') or xref.startswith('SMP') or xref.startswith('PID'):
            xref_db, xref_id = xref.split(':')
            new_id = '{}:{}'.format(xref_db, int(xref_id))
            if new_id in xref_dict:
                xref_name, xref_def = xref_dict[new_id]
                training_data.append((pw_id, pw_name, pw_def, new_id, xref_name, xref_def))

# write to file
output_file = os.path.join(paths.processed_data_dir, 'training_data.tsv')

with open(output_file, 'w') as outf:
    outf.write('PW_id\tPW_name\tPW_def\txref_id\txref_name\txref_def\n')
    for training_line in training_data:
        outf.write('\t'.join(training_line) + '\n')
