# shared utility functions for processing pathways

import tqdm
import itertools
from collections import defaultdict
import pathhier.constants as constants


def flatten(l):
    """
    Flatten a list of lists
    :param l:
    :return:
    """
    return [item for sublist in l for item in sublist]


def clean_path_id(db_name, path_id):
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


def clean_subpath_id(db_name, subpath_id):
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


def clean_subpath_id_wp(db_name, subpath_id, pathway_list):
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


def clean_subpaths(db_name, subpath_set, pathway_list=None):
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


def clean_xrefs(xrefs):
    """
    Clean input xref identifiers to achieve consistent spelling and capitalization
    :param xrefs:
    :return:
    """
    new_xrefs = []
    for x in xrefs:
        if len(x) > 0:
            parts = x.split(':')
            xref_db = parts[0]
            xref_id = ':'.join(parts[1:])
            if xref_db in constants.DB_XREF_MAP:
                new_xrefs.append("{}:{}".format(constants.DB_XREF_MAP[xref_db], xref_id))
            else:
                new_xrefs.append("{}:{}".format(xref_db, xref_id))
    return new_xrefs


def merge_similar(map_dict):
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


def form_long_pw_string_entry(pw_id, pw_entry, pw):
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


def form_long_kb_string_entry(kb_id, kb_entry, kb):
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


def form_short_kb_string_entry(kb_id, kb_entry):
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


def form_name_entries(pos, pw_id, pw_entry, kb_id, kb_entry):
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
            'pw_id': pw_id,
            'pw_cls': pw_name,
            'kb_id': kb_id,
            'kb_cls': kb_name
        })
    return entries


def form_name_entries_special(pos, pw_id, pw_entry, kb_id, kb_entry):
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
            'pw_id': pw_id,
            'pw_cls': pw_name,
            'kb_id': kb_id,
            'kb_cls': kb_entry[0]
        })
    return entries






