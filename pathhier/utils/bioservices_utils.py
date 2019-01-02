# shared utility functions for processing bioservices outputs


def get_uniprot_secondaries(uniprot_result):
    """
    Get secondary accession identifiers from UniProt text result
    :param uniprot_result:
    :return:
    """
    accession_lines = [l for l in uniprot_result.split('\n') if l.startswith('AC')]
    secondaries = []
    for ac_line in accession_lines:
        secondaries += ['UniProt:{}'.format(uid[:-1]) for uid in ac_line.split()[1:]]
    return secondaries


def get_uniprot_id(uniprot_result):
    """
    Get names from UniProt text result
    :param uniprot_result:
    :return:
    """
    id_lines = [l for l in uniprot_result.split('\n') if l.startswith('ID')]

    for id_line in id_lines:
        return id_line.split()[1]

    return None


def get_uniprot_names(uniprot_result):
    """
    Get names from UniProt text result
    :param uniprot_result:
    :return:
    """
    name_lines = [l for l in uniprot_result.split('\n') if l.startswith('DE')]

    names = []

    for nm_line in name_lines:
        if 'Full=' in nm_line:
            names.append(nm_line.split('Full=')[-1][:-1])
        elif 'Short=' in nm_line:
            names.append(nm_line.split('Short=')[-1][:-1])

    return names


def get_uniprot_gene_info(uniprot_result):
    """
    Get gene info from UniProt text result
    :param uniprot_result:
    :return:
    """
    gene_lines = [l for l in uniprot_result.split('\n') if l.startswith('GN')]

    gene_names = []

    for gn_line in gene_lines:
        parts = gn_line[2:].split(';')
        for p in parts:
            p = p.strip()
            if p.startswith('Name='):
                gene_names.append(p[5:])
            elif p.startswith('Synonyms='):
                gene_names += [s.strip() for s in p[9:].split(',')]

    return gene_names


def get_chebi_synonyms(chebi_ent):
    """
    Get synonyms from ChEBI entity
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'Synonyms'):
        return [entry.data for entry in chebi_ent.Synonyms]
    else:
        return []


def get_chebi_secondaries(chebi_ent):
    """
    Get secondary accession identifiers from ChEBI result
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'SecondaryChEBIIds'):
        return chebi_ent.SecondaryChEBIIds
    else:
        return []


def get_chebi_parents(chebi_ent):
    """
    Get parents of ChEBI entity
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'OntologyParents'):
        return [ent.chebiId for ent in chebi_ent.OntologyParents if
                (ent.type == 'is a')]
    else:
        return []


def get_tautomers_of(chebi_ent):
    """
    Get tautomers of ChEBI entity
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'OntologyParents'):
        return [ent.chebiId for ent in chebi_ent.OntologyParents if
                (ent.type == "is tautomer of")]
    else:
        return []


def get_conjugate_acids_of(chebi_ent):
    """
    Get conjugate acids of ChEBI entities
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'OntologyParents'):
        return [ent.chebiId for ent in chebi_ent.OntologyParents if
                (ent.type == "is conjugate acid of")]
    else:
        return []


def get_conjugate_bases_of(chebi_ent):
    """
    Get conjugate bases of ChEBI entities
    :param chebi_ent:
    :return:
    """
    if hasattr(chebi_ent, 'OntologyParents'):
        return [ent.chebiId for ent in chebi_ent.OntologyParents if
                (ent.type == "is conjugate base of")]
    else:
        return []