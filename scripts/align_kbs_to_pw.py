#!python

import os
import sys
import json
import numpy as np
import itertools
from collections import defaultdict
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

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
        pw_file = os.path.join(paths.pathway_ontology_dir, 'pw.json')

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

        self.output_file = os.path.join(
            paths.output_dir,
            '{}_pw_alignment_{}.tsv'.format(kb_name, date(2002, 12, 25).strftime('%Y%m%d'))
        )

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
        if len(ngm_union) == 0:
            ngm_union = {0}
        return len(ngm_intersect) / len(ngm_union)

    def align(self):
        """
        Align kb with PW
        :return:
        """
        done_pairs = set([])
        self.score_list = []
        self.inst_list = []

        for s_id, s_info in self.kb.items():
            s_vocab = self.cand_sel.s_mat[s_id]

            # list of matches in PW
            matches = []

            for pw_class, pw_vocab in self.cand_sel.select(s_id).items():
                score = cosine_similarity(s_vocab, pw_vocab)[0][0]
                matches.append((pw_class, score))

            if matches:
                # sort matches by best similarity score
                matches.sort(key=lambda x: x[1], reverse=True)

                for m, s in matches:
                    if s >= constants.SIMSCORE_THRESHOLD and (s_id, m) not in done_pairs:
                        self.score_list.append(('match', s_id, m, s))
                        done_pairs.add((s_id, m))

        self.score_list.sort(key=lambda x: (x[1], 1-x[3]))
        self.write_to_file()
        return

    def write_to_file(self):
        """
        Write score_list to output_file and inst_list to instance_file
        :return:
        """
        with open(self.output_file, 'w') as outf:
            outf.write('match_type\tcosine_similarity\t{}_id\t{}_name\tpw_id\tpw_name\n'.format(
                self.kb_name, self.kb_name
            ))
            for match_type, kb_id, pw_id, score in self.score_list:

                # adjust pathway ID for BioCyc pathways
                kb_id_use = kb_id
                if self.kb_name == 'biocyc':
                    kb_id_use = 'BioCyc:' + kb_id

                outf.write('%s\t%.2f\t%s\t%s\t%s\t%s\n' % (
                    match_type, score, kb_id_use,
                    self.kb[kb_id]['name'], pw_id, self.pw[pw_id]['name']
                ))


# list of KBs to align
kb_names = ['reactome']

# align each KB in list
for name in kb_names:
    aligner = KBAligner(name)
    aligner.align()















