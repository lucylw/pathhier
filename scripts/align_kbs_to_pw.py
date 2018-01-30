#!python

import os
import json
import numpy as np
import itertools
from collections import defaultdict
from typing import List, Dict

from pathhier.paths import PathhierPaths
import pathhier.constants as constants
from pathhier.candidate_selector import CandidateSelector


def get_idf_dict(kb, i_to_w):
    """
    Construct idf mapping dict between word ind and idf score in kb
    :param kb:
    :param i_to_w:
    :return:
    """
    idf_dict = defaultdict(float)
    n = len(kb)

    for ind in i_to_w:
        names_with_word = [k for k, v in kb.items() if ind in v['name_tokens']]
        if names_with_word:
            idf = np.log(n / (len(names_with_word) + 1))
        else:
            idf = np.log(n / 1)
        idf_dict[ind] = idf

    return idf_dict


def compute_weighted_token_jaccard(
        s_toks: List,
        t_toks: List,
        s_idfs: Dict,
        t_idfs: Dict
):
    """
    Compute weighted token jaccard index between the set of tokens from source entity and target entity
    :param s_toks: tokens from source entity
    :param t_toks: tokens from target entity
    :param s_idfs: idf scores of tokens from source kb
    :param t_idfs: idf scores of tokens from target kb
    :return:
    """
    tok_intersect = set(s_toks).intersection(set(t_toks))
    tok_union = set(s_toks).union(set(t_toks))

    weighted_numerator = 0.
    for itok in tok_intersect:
        weighted_numerator += 0.5*(s_idfs[itok] + t_idfs[itok])

    weighted_denominator = 0.
    for utok in tok_union:
        weighted_denominator += 0.5*(s_idfs[utok] + t_idfs[utok])

    return weighted_numerator / weighted_denominator


def compute_unweighted_jaccard(
        s_ngms: List,
        t_ngms: List
):
    """
    Compute jaccard index between the set of ngrams from source entity and target entity
    :param s_ngms: ngrams from source entity
    :param t_ngms: ngrams from target entity
    :return:
    """
    ngm_intersect = set(s_ngms).intersection(set(t_ngms))
    ngm_union = set(s_ngms).union(set(t_ngms))
    return len(ngm_intersect) / len(ngm_union)


paths = PathhierPaths()

source_names = ['biocyc', 'reactome']

for source_kb_name in source_names:
    source_json = source_kb_name + '_ontology.json'

    source_file = os.path.join(paths.output_dir, source_json)
    target_file = os.path.join(paths.output_dir, 'pw.json')

    with open(source_file, 'r') as f:
        s_ont = json.load(f)

    with open(target_file, 'r') as f:
        t_ont = json.load(f)

    selector = CandidateSelector(s_ont, t_ont)
    jacc_list = []
    done_pairs = set([])
    instance_matches = []

    for s_pw, val in s_ont.items():
        insts = s_ont[s_pw]['instances']
        t_matches = []
        for t_pw in selector.select(s_pw)[:constants.KEEP_TOP_N_CANDIDATES]:
            s_tokens = val['name_tokens']
            t_tokens = t_ont[t_pw]['name_tokens']
            jacc = compute_weighted_token_jaccard(s_tokens, t_tokens, selector.s_token_to_idf, selector.t_token_to_idf)

            s_ngrams = val['name_ngrams']
            t_ngrams = t_ont[t_pw]['name_ngrams']
            ngram_jacc = compute_unweighted_jaccard(s_ngrams, t_ngrams)

            t_matches.append((t_pw, jacc, ngram_jacc))

        if t_matches:
            t_matches.sort(key=lambda x: x[1] + x[2], reverse=True)
            best_t, best_jacc = t_matches[0]
            if (s_pw, best_t) not in done_pairs:
                jacc_list.append((s_pw, best_t, best_jacc))
                for i in insts:
                    instance_matches.append((i, best_t, best_jacc))
                done_pairs.add((s_pw, best_t))

    instance_matches.sort(key=lambda x: x[2], reverse=True)
    jacc_list.sort(key=lambda x: x[2], reverse=True)

    output_file = os.path.join(paths.output_dir, '{}_pw_alignment.tsv'.format(source_kb_name))
    instance_file = os.path.join(paths.output_dir, '{}_pw_instance_alignment.tsv'.format(source_kb_name))

    with open(output_file, 'w') as outf:
        outf.write('score\t{}_id\t{}_name\tpw_id\tpw_name\n'.format(
            source_kb_name, source_kb_name
        ))
        for s_id, t_id, score in jacc_list:
            outf.write('%f\t%s\t%s\t%s\t%s\n' % (
                score, s_id, s_ont[s_id]['name'], t_id, t_ont[t_id]['name']
            ))

    with open(instance_file, 'w') as outf:
        outf.write('score\t{}_instance_id\tpw_id\tpw_name\n'.format(
            source_kb_name
        ))
        for s_inst, t_id, score in instance_matches:
            outf.write('%f\t%s\t%s\t%s\n' % (
                score, s_inst, t_id, t_ont[t_id]['name']
            ))













