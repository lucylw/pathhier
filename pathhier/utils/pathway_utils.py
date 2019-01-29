# shared utility functions for processing pathways

import csv
import tqdm
import itertools
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import pathhier.constants as constants


def clean_path_id(db_name: str, path_id: str) -> str:
    """
    Take a DB name and pathway identifier string and returns a clean id
    formatted as DB_name:path_id
    :param db_name:
    :param path_id:
    :return:
    """
    if db_name == "humancyc" or db_name == "reactome" or db_name == "pid":
        return db_name + ':' + path_id.split('#')[-1]
    elif db_name == "kegg" or db_name == "panther" or \
            db_name == "smpdb" or db_name == "biomodels":
        return db_name + ':' + path_id.split('/')[-1]
    elif db_name == "wikipathways":
        return db_name + ':' + path_id
    else:
        raise NotImplementedError("Unknown pathway KB: %s" % db_name)


def clean_subpath_id(db_name: str, subpath_id: str) -> str:
    """
    Clean the subpathway id for all DBs except Wikipathways
    :param db_name:
    :param subpath_id:
    :return:
    """
    if db_name == "reactome":
        return db_name + ':' + subpath_id.split('#')[-1]
    elif db_name == "kegg":
        return db_name + ':' + subpath_id.split('/')[-1]
    elif db_name == "smpdb":
        return db_name + ':' + subpath_id
    elif db_name == "wikipathways":
        raise NotImplementedError("Use clean_subpath_id_wp for WikiPathway ids...")
    else:
        return subpath_id


def clean_subpath_id_wp(db_name: str, subpath_id: str, pathway_list: List) -> str:
    """
    Clean subpathway ids for WikiPathway pathways
    :param db_name:
    :param subpath_id:
    :param pathway_list:
    :return:
    """
    if '{}:'.format(db_name) in subpath_id:
        return subpath_id
    else:
        referenced_pathways = [p.uid for p in pathway_list if subpath_id.split(':')[-1] == p.name]
        if len(referenced_pathways) > 0:
            return db_name + ':' + referenced_pathways[0]
        else:
            return subpath_id


def clean_subpaths(db_name: str, subpath_set: Set, pathway_list=None) -> Set:
    """
    Clean subpaths
    :param db_name:
    :param subpath_set:
    :param pathway_list:
    :return:
    """
    if db_name != "wikipathways":
        return {clean_subpath_id(db_name, subpath) for subpath in subpath_set if subpath}
    else:
        return {clean_subpath_id_wp(db_name, subpath, pathway_list) for subpath in subpath_set if subpath}


def clean_xrefs(xrefs: List, avoid_terms: List) -> List:
    """
    Clean input xref identifiers to achieve consistent spelling and capitalization
    :param xrefs:
    :param avoid_terms:
    :return:
    """
    new_xrefs = []
    for x in xrefs:
        if len(x) > 0:
            if any([t in x for t in avoid_terms]):
                continue

            parts = x.split(':')
            xref_db = parts[0]

            if xref_db.lower() == parts[1].lower():
                xref_id = ':'.join(parts[2:])
            else:
                xref_id = ':'.join(parts[1:])

            if xref_db in constants.DB_XREF_MAP:
                new_xref = "{}:{}".format(constants.DB_XREF_MAP[xref_db], xref_id)
            else:
                new_xref = "{}:{}".format(xref_db, xref_id)

            new_xrefs.append(new_xref)

    return new_xrefs


def merge_similar(map_dict: Dict) -> Dict:
    """
    Merge entries in mapping dictionary with shared values
    :param map_dict: dictionary of xref mappings
    :return:
    """
    new_map_dict = defaultdict(set)

    for uid in tqdm.tqdm(map_dict):
        for k, v in new_map_dict.items():
            if not v.isdisjoint(map_dict[uid]):
                new_map_dict[k].update(map_dict[uid])
                break
        new_map_dict[uid] = map_dict[uid]
    return new_map_dict


def form_long_pw_string_entry(pw_id: str, pw_entry: Dict, pw: Dict) -> Tuple[str, str]:
    """
    Form a string representation of the KB entry
    :param pw_id: PW id
    :param pw_entry: json entry
    :param pw: PW json object
    :return:
    """
    superclasses = ['subClassOf: {}'.format(pw[parent_id]['name'])
                    for parent_id in pw_entry['subClassOf'] if parent_id in pw]
    part_supers = ['part_of: {}'.format(pw[parent_id]['name'])
                   for parent_id in pw_entry['part_of'] if parent_id in pw]

    p_string = '; '.join(set(pw_entry['aliases']))
    if pw_entry['definition']:
        p_string += '; ' + '; '.join(pw_entry['definition'])
    if superclasses:
        p_string += '; ' + '; '.join(superclasses)
    if part_supers:
        p_string += '; ' + '; '.join(part_supers)
    p_string += '; '

    return pw_id, p_string


def form_long_kb_string_entry(kb_id: str, kb_entry: Dict, kb: Dict) -> Tuple[str, str]:
    """
    Form a string representation of the KB entry
    :param kb_id: UID in KB
    :param kb_entry: json entry
    :param kb: KB object
    :return:
    """
    superclasses = ['subClassOf: {}'.format(kb[parent_id]['name'])
                    for parent_id in kb_entry['subClassOf'] if parent_id in kb]
    part_supers = ['part_of: {}'.format(kb[parent_id]['name'])
                   for parent_id in kb_entry['part_of'] if parent_id in kb]

    kb_string = '; '.join(set(kb_entry['aliases']))
    if kb_entry['definition']:
        kb_string += '; ' + '; '.join(kb_entry['definition'])
    if superclasses:
        kb_string += '; ' + '; '.join(superclasses)
    if part_supers:
        kb_string += '; ' + '; '.join(part_supers)
    kb_string += '; '

    return kb_id, kb_string


def form_short_kb_string_entry(kb_id: str, kb_entry: List) -> Tuple[str, str]:
    """
    Form a string representation of the kb entry
    :param kb_id:
    :param kb_entry:
    :return:
    """
    kb_string = kb_entry[0] + '; ' + kb_entry[1]
    return kb_id, kb_string


def form_matching_short_entries(pos, pw_id, pw_entry, kb_id, kb_entry):
    """
    Split pair into entries
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    return {
            'label': pos,
            'pw_id': pw_id,
            'pw_names': list(set(pw_entry['aliases'])),
            'pw_def': '; '.join(pw_entry['definition']),
            'kb_id': kb_id,
            'kb_names': [kb_entry[0]],
            'kb_def': kb_entry[1]
        }


def form_matching_long_entries(pos, pw_id, pw_entry, kb_id, kb_entry):
    """
    Split pair into entries
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    return {
        'label': pos,
        'pw_id': pw_id,
        'pw_names': list(set(pw_entry['aliases'])),
        'pw_def': '; '.join(pw_entry['definition']),
        'kb_id': kb_id,
        'kb_names': list(set(kb_entry['aliases'])),
        'kb_def': '; '.join(kb_entry['definition'])
    }


def form_name_entries(pos, provenance, pw_id, pw_entry, kb_id, kb_entry):
    """
    Create name match entries
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    entries = []
    for pw_name, kb_name in itertools.product(
            set(pw_entry['aliases']), set(kb_entry['aliases'])
    ):
        entries.append({
            'label': pos,
            'provenance': provenance,
            'pw_id': pw_id,
            'pw_cls': pw_name,
            'kb_id': kb_id,
            'kb_cls': kb_name
        })
    return entries


def form_name_entries_special(pos, provenance, pw_id, pw_entry, kb_id, kb_entry):
    """
    Create name match entries
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    entries = []
    for pw_name in set(pw_entry['aliases']):
        entries.append({
            'label': pos,
            'provenance': provenance,
            'pw_id': pw_id,
            'pw_cls': pw_name,
            'kb_id': kb_id,
            'kb_cls': kb_entry[0]
        })
    return entries


def form_definition_entries(pos, provenance, pw_id, pw_entry, kb_id, kb_entry):
    """
    Create def match entries
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    entries = []
    for pw_def, kb_def in itertools.product(
            pw_entry['definition'], kb_entry['definition']
    ):
        entries.append({
            'label': pos,
            'provenance': provenance,
            'pw_id': pw_id,
            'pw_cls': pw_def,
            'kb_id': kb_id,
            'kb_cls': kb_def
        })
    return entries


def form_definition_entries_special(pos, provenance, pw_id, pw_entry, kb_id, kb_entry):
    """
    Create def match entry
    :param pos:
    :param pw_id:
    :param pw_entry:
    :param kb_id:
    :param kb_entry:
    :return:
    """
    entries = []
    for pw_def in pw_entry['definition']:
        entries.append({
            'label': pos,
            'provenance': provenance,
            'pw_id': pw_id,
            'pw_cls': pw_def,
            'kb_id': kb_id,
            'kb_cls': kb_entry[1]
        })
    return entries


def split_data(data, dev_perc, test_perc=0.0):
    """
    Split data stratified into train, dev, and test set
    :param data:
    :param dev_perc: percent of development data
    :param test_perc: percent of test data
    :return:
    """
    labels = [(i, d['label']) for i, d in enumerate(data)]
    inds = np.array([l[0] for l in labels])
    labs = np.array([l[1] for l in labels])

    if test_perc > 0.:
        ind_rest, ind_test, lab_rest, lab_test = train_test_split(inds, labs,
                                                                  stratify=labs,
                                                                  test_size=test_perc)
    else:
        ind_test = []
        ind_rest = inds
        lab_rest = labs

    ind_train, ind_dev, lab_train, lab_dev = train_test_split(ind_rest, lab_rest,
                                                              stratify=lab_rest,
                                                              test_size=dev_perc * (1-test_perc))

    train_data = [data[i] for i in ind_train]
    dev_data = [data[i] for i in ind_dev]
    test_data = [data[i] for i in ind_test]

    return train_data, dev_data, test_data


def get_corresponding_pathway(kbs: Dict, kb_id: str):
    """
    Get corresponding pathway from list of KBs
    :param kbs:
    :param kb_id:
    :return:
    """
    if ':' in kb_id:
        kb_name, identifier = kb_id.split(':')
        kb_name = kb_name.lower()
        if kb_name in kbs:
            try:
                return kbs[kb_name].get_pathway_by_uid(kb_id)
            except IndexError:
                print("ERROR: {} can't be found in given KBs.".format(kb_id))
                return None
    else:
        try:
            if kb_id.startswith('WP'):
                return kbs['wikipathways'].get_pathway_by_uid(kb_id)
            if kb_id.startswith('SMP'):
                return kbs['smpdb'].get_pathway_by_uid(kb_id)
        except IndexError:
            print("ERROR: {} can't be found in given KBs.".format(kb_id))
            return None


def load_pathway_pairs(pair_file):
    """
    Load PW clustering outputs (pathways to align)
    :param pair_file:
    :return:
    """
    all_pairs = []

    with open(pair_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        try:
            while reader:
                sim_score, overlap, pw_id, kb1_id, kb2_id = next(reader)
                _, _, _, kb1_name, kb2_name = next(reader)
                all_pairs.append([float(sim_score), float(overlap), pw_id, kb1_id, kb2_id])
                next(reader)
        except StopIteration:
            pass

    return all_pairs


def load_pairs_quick(tsv_file):
    """
    Load PW clustering outputs from tsv file
    :param tsv_file:
    :return:
    """
    all_pairs = []

    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for sim_score, overlap, pw_id, kb1_id, kb2_id in reader:
            all_pairs.append([float(sim_score), float(overlap), pw_id, kb1_id, kb2_id])

    return all_pairs


def generate_gmt_file(output_file: str, gene_sets: List) -> str:
    """
    Generate a GSEA gmt file
    :param output_file: name of output file
    :param gene_sets: list of gene sets; each entry is (gene_set_name, gene_set_origin, list of gene symbols)
    :return:
    """
    with open(output_file, 'w') as f:
        for gs_name, gs_origin, symbols in gene_sets:
            f.write('{}\t{}\t{}\n'.format(
                gs_name,
                gs_origin,
                '\t'.join(symbols)
            ))
    return output_file


def get_pw_synonyms(synonyms: List) -> List:
    """
    Get pathway synonyms from PW class
    :param syns:
    :return:
    """
    pathway_instances = []

    for syn in synonyms:
        if syn.lower().startswith('kegg'):
            db, id = syn.lower().split(':')
            pathway_instances.append('{}:hsa{}'.format(db, id))
        elif syn.startswith('SMP'):
            pathway_instances.append(syn.replace(':', ''))
        elif syn.lower().startswith('pid'):
            pathway_instances.append(syn.lower())

    return pathway_instances


def get_pathway_kb(uid):
    """
    Return pathway name from uid
    :param uid:
    :return:
    """
    uid_lower = uid.lower()
    if uid_lower.startswith('humancyc'):
        return 'humancyc'
    if uid_lower.startswith('kegg'):
        return 'kegg'
    if uid_lower.startswith('panther'):
        return 'panther'
    if uid_lower.startswith('pid'):
        return 'pid'
    if uid_lower.startswith('reactome') or uid_lower.startswith('r-hsa'):
        return 'reactome'
    if uid_lower.startswith('smp'):
        return 'smpdb'
    if uid_lower.startswith('wp') or uid_lower.startswith('wikipath'):
        return 'wikipathways'
    raise Exception('Unknown pathway database')


