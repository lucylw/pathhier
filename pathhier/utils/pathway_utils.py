# shared utility functions for processing pathways

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
    elif db_name == "kegg" or db_name == "panther" or db_name == "smpdb":
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
        return db_name + ':' + 'PW'+'0'*(6-len(subpath_id))+subpath_id
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
        return {clean_subpath_id(db_name, subpath) for subpath in subpath_set}
    else:
        return {clean_subpath_id_wp(db_name, subpath, pathway_list) for subpath in subpath_set}


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
            xref_id = parts[-1]
            if xref_db in constants.KEEP_XREF_DBS:
                new_xrefs.append("{}:{}".format(xref_db, xref_id))
            elif xref_db in constants.DB_XREF_MAP:
                new_xrefs.append("{}:{}".format(constants.DB_XREF_MAP[xref_db], xref_id))
    return new_xrefs
