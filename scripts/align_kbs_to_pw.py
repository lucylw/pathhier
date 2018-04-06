#!python

import os
import sys
import json
import numpy as np
import itertools
from collections import defaultdict
from typing import List, Dict

from pathhier.paths import PathhierPaths
import pathhier.constants as constants
from pathhier.candidate_selector import CandidateSelector


# class for aligning a pathway KB to the Pathway Ontology
class KBAligner:
    def __init__(self, kb_name):
        """
        Initialize and load all files
        :param kb_name:
        """
        self.kb_name = kb_name

        paths = PathhierPaths()
        kb_file = os.path.join(paths.output_dir, kb_name + '_ontology.json')
        pw_file = os.path.join(paths.output_dir, 'pw.json')

        assert os.path.exists(kb_file)
        assert os.path.exists(pw_file)

        # load KB to align to PW
        with open(kb_file, 'r') as f:
            self.kb = json.load(f)

        # load pathway ontology
        with open(pw_file, 'r') as f:
            self.pw = json.load(f)

        self.cand_sel = CandidateSelector(self.kb, self.pw)
        self.score_list = []
        self.inst_list = []

        self.output_file = os.path.join(paths.output_dir, '{}_pw_alignment.tsv'.format(kb_name))
        self.instance_file = os.path.join(paths.output_dir, '{}_pw_instance_alignment.tsv'.format(kb_name))

    def compute_weighted_jaccard(
            self,
            s_toks: List,
            t_toks: List
    ):
        """
        Compute weighted token jaccard index between the set of tokens from source entity and target entity
        :param s_toks: tokens from source entity
        :param t_toks: tokens from target entity
        :return:
        """
        tok_intersect = set(s_toks).intersection(set(t_toks))
        tok_union = set(s_toks).union(set(t_toks))

        weighted_numerator = 0.
        for itok in tok_intersect:
            weighted_numerator += 0.5*(self.cand_sel.s_token_to_idf[itok] + self.cand_sel.t_token_to_idf[itok])

        weighted_denominator = sys.float_info.epsilon
        for utok in tok_union:
            weighted_denominator += 0.5*(self.cand_sel.s_token_to_idf[utok] + self.cand_sel.t_token_to_idf[utok])

        return weighted_numerator / weighted_denominator

    @staticmethod
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

    def align(self):
        """
        Align kb with PW
        :return:
        """
        done_pairs = set([])
        self.score_list = []
        self.inst_list = []

        for p_id, p_info in self.kb.items():

            # list of matches in PW
            matches = []

            for pw_class in self.cand_sel.select(p_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                scores = dict()

                scores['name_token_jaccard'] = self.compute_weighted_jaccard(
                    p_info['name_tokens'],
                    self.pw[pw_class]['name_tokens']
                )

                scores['name_ngram_jaccard'] = self.compute_unweighted_jaccard(
                    p_info['name_ngrams'],
                    self.pw[pw_class]['name_ngrams']
                )

                # get max alias token jaccard
                scores['alias_token_jaccard'] = max(map(
                    lambda x: self.compute_unweighted_jaccard(x[0], x[1]),
                    itertools.product(
                        p_info['alias_tokens'],
                        self.pw[pw_class]['alias_tokens']
                    )
                ))

                scores['def_token_jaccard'] = self.compute_weighted_jaccard(
                    p_info['def_tokens'],
                    self.pw[pw_class]['def_tokens']
                )

                scores['all_token_jaccard'] = self.compute_weighted_jaccard(
                    p_info['all_tokens'],
                    self.pw[pw_class]['all_tokens']
                )

                max_score = max(scores.values())
                mean_score = np.mean(list(scores.values()))

                matches.append((pw_class, max_score, mean_score, scores))

            if matches:
                # sort matches by best similarity score
                matches.sort(key=lambda x: x[1], reverse=True)

                # best match class
                best_pw, best_simscore, mean_simscore = matches[0][:3]
                self.score_list.append(('best', p_id, best_pw, best_simscore, mean_simscore))
                for inst in p_info['instances']:
                    self.inst_list.append(('best', inst, best_pw, best_simscore, mean_simscore))

                for m in matches[1:]:
                    best_pw, best_simscore, mean_simscore = m[:3]
                    if best_simscore >= constants.SIMSCORE_THRESHOLD or mean_simscore >= constants.SIMSCORE_THRESHOLD \
                            and (p_id, best_pw) not in done_pairs:
                        self.score_list.append(('other', p_id, best_pw, best_simscore, mean_simscore))
                        for inst in p_info['instances']:
                            self.inst_list.append(('other', inst, best_pw, best_simscore, mean_simscore))
                        done_pairs.add((p_id, best_pw))

        self.inst_list.sort(key=lambda x: x[3], reverse=True)
        self.score_list.sort(key=lambda x: (x[1], 1-x[3]))

        self.write_to_file()

        return

    def write_to_file(self):
        """
        Write score_list to output_file and inst_list to instance_file
        :return:
        """
        with open(self.output_file, 'w') as outf:
            outf.write('match_type\tmax_score\tmean_score\t{}_id\t{}_name\tpw_id\tpw_name\n'.format(
                self.kb_name, self.kb_name
            ))
            for match_type, kb_id, pw_id, max_score, mean_score in self.score_list:

                # adjust pathway ID for BioCyc pathways
                kb_id_use = kb_id
                if self.kb_name == 'biocyc':
                    kb_id_use = 'BioCyc:' + kb_id

                outf.write('%s\t%.2f\t%.2f\t%s\t%s\t%s\t%s\n' % (
                    match_type, max_score, mean_score, kb_id_use,
                    self.kb[kb_id]['name'], pw_id, self.pw[pw_id]['name']
                ))

        with open(self.instance_file, 'w') as outf:
            outf.write('match_type\tmax_score\tmean_score\t{}_instance_id\tpw_id\tpw_name\n'.format(
                self.kb_name
            ))
            for match_type, kb_inst, pw_id, max_score, mean_score in self.inst_list:
                outf.write('%s\t%.2f\t%.2f\t%s\t%s\t%s\n' % (
                    match_type, max_score, mean_score, kb_inst, pw_id, self.pw[pw_id]['name']
                ))


# list of KBs to align
kb_names = ['biocyc', 'reactome']

# align each KB in list
for name in kb_names:
    aligner = KBAligner(name)
    aligner.align()















