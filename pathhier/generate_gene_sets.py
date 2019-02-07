import os
import csv
import json
import glob
import tqdm
import pickle
import itertools

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from sklearn.cluster import SpectralClustering, DBSCAN
from pybiomart import Server
from bioservices import *
import requests

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB
from pathhier.cluster_model import PathwayClusterer
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.base_utils as base_utils
import pathhier.constants as constants


class GeneSetGenerator:
    def __init__(self):
        """
        Initialize class
        """
        paths = PathhierPaths()

        # load PW
        pw_file = os.path.join(paths.pathway_ontology_dir, 'pw.json')
        self.pw = json.load(open(pw_file, 'r'))

        # load KBs
        self.kb_names = dict()

        for kb_name in paths.all_kb_paths:
            print("Loading {}...".format(kb_name))
            kb_file_path = os.path.join(paths.processed_data_dir, 'kb_{}.pickle'.format(kb_name))
            kb = PathKB(kb_name)
            kb = kb.load_pickle(kb_name, kb_file_path)
            for p in kb.pathways:
                self.kb_names[p.uid] = p.name

        # load cluster file and pairs associated with each PW class
        path_pairs_file = os.path.join(paths.output_dir, 'model_output', 'clustered_groups.tsv')
        self.path_pairs = pathway_utils.load_pathway_pairs(path_pairs_file)

        # load pairs parsed from pickle
        keep_pairs_file = os.path.join(paths.output_dir, 'model_output', 'pairs_keep.pickle')
        self.keep_pairs = pickle.load(open(keep_pairs_file, 'rb'))

        # load pathways from temp files
        temp_dir = os.path.join(paths.base_dir, 'temp')
        self.pathways = self._read_pathways(self.keep_pairs, temp_dir)

        # load all PW mappings from disk
        self.best_pw_matches = self._construct_pw_mapping_dict()

        # load alignments
        alignment_dir = os.path.join(paths.output_dir, 'alignments')
        alignment_files = glob.glob(os.path.join(alignment_dir, 'alignment*.pickle'))
        self.al_scores, self.al_mappings = self._read_alignments(alignment_files)

        # ensembl lookup dict
        self.ensembl_dict = self.get_ensembl_dict()

        # set output gene set file
        self.output_pickle = os.path.join(paths.output_dir, 'pw_gene_sets_v0.1.pickle')
        self.output_file = os.path.join(paths.output_dir, 'pw_gene_sets_v0.1.gmt')

    @staticmethod
    def get_ensembl_dict():
        """
        Create BioMart server and get Ensembl id mapping dict
        :return:
        """
        # map ensemble identifiers to gene names
        server = Server(host='http://www.ensembl.org')

        ensembl_dataset = (
            server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
        )

        results = ensembl_dataset.query(
            attributes=['ensembl_gene_id', 'external_gene_name']
        )

        ens_dict = dict(zip(results['Gene stable ID'], results['Gene name']))

        return ens_dict

    @staticmethod
    def _read_pathways(path_pairs: List, temp_dir: str) -> Dict:
        """
        Read pathways from temp dir
        :param path_pairs:
        :param temp_dir:
        :return:
        """
        pathway_ids = list(set([x[0] for x in path_pairs] + [x[1] for x in path_pairs]))
        pathway_ids.sort()

        pathways = dict()

        for i, pathway_id in tqdm.tqdm(enumerate(pathway_ids)):
            path_file = os.path.join(temp_dir, 'pathway{}.pickle'.format(i))
            if os.path.exists(path_file):
                pathways[pathway_id] = pickle.load(open(path_file, 'rb'))

        return pathways

    @staticmethod
    def _read_alignments(al_files: List) -> Tuple[Dict, Dict]:
        """
        Read all alignments from disk
        :return:
        """
        scores = dict()
        al_dict = dict()

        print('Loading alignments from file...')

        for al_file in tqdm.tqdm(al_files):
            p1_id, p2_id, al_score, mapping = pickle.load(open(al_file, 'rb'))
            scores[(p1_id, p2_id)] = al_score
            al_dict[(p1_id, p2_id)] = mapping

        return scores, al_dict

    def _construct_pw_mapping_dict(self):
        """
        Read all PW matches from output files
        :return:
        """
        clusterer = PathwayClusterer(load=False)
        pw_mapping_dict = clusterer.load_all_pw_alignment_outputs(defaultdict(list))

        kb_to_pw = defaultdict(list)

        for pw_id, kb_vals in pw_mapping_dict.items():

            if pw_id in constants.PW_ROOT_CLASSES:
                continue

            pw_id_short = pw_id.split('/')[-1]

            for score, kb_name, kb_id in kb_vals:
                if kb_name == 'humancyc':
                    uid = '{}:{}'.format('HumanCyc', kb_id)
                elif kb_name == 'kegg':
                    uid = '{}:{}'.format(kb_name, kb_id)
                elif kb_name == 'panther':
                    uid = '{}:{}'.format(kb_name, kb_id)
                elif kb_name == 'pid':
                    if kb_id in clusterer.pid_mapping_dict:
                        uid = clusterer.pid_mapping_dict[kb_id]
                    else:
                        uid = kb_id
                elif kb_name == 'reactome':
                    uid = '{}:{}'.format('Reactome', kb_id)
                elif kb_name == 'smpdb':
                    uid = 'SMP{}'.format(kb_id)
                elif kb_name == 'wikipathways':
                    uid = kb_id
                else:
                    uid = '{}:{}'.format(kb_name, kb_id)

                kb_to_pw[uid].append((score, pw_id_short))

        for val in kb_to_pw.values():
            val.sort(key=lambda x: x[0], reverse=True)

        return kb_to_pw

    def _organize_by_pw_class(self) -> Tuple[Dict, List]:
        """
        Add alignments by PW class
        :return:
        """
        group_by_pw_cls = defaultdict(set)
        feature_list = []

        for sim_score, overlap, pw_id, kb1_id, kb2_id in self.path_pairs:
            group_by_pw_cls[pw_id].add(kb1_id)
            group_by_pw_cls[pw_id].add(kb2_id)
            if (kb1_id, kb2_id) in self.al_scores:
                al_score = self.al_scores[(kb1_id, kb2_id)]
            else:
                al_score = 0.0
            feature_list.append((pw_id, kb1_id, kb2_id, sim_score, overlap, al_score))

        return group_by_pw_cls, feature_list

    @staticmethod
    def _convert_labels_to_groups(cluster_labels):
        """
        Convert cluster label array to list of groups
        :param cluster_labels:
        :return:
        """
        groups = []

        for l in range(max(cluster_labels) + 1):
            inds = np.where(cluster_labels == l)
            groups.append(list(inds[0]))

        return groups

    def _check_constraints(self, cluster_labels, groups, pathway_ids, synonyms):
        """
        Check what degree constraints are satisfied
        :param cluster_labels:
        :param groups:
        :param pathway_ids: pathway ids
        :param synonyms: dictionary of pathways that should link together
        :return:
        """
        num_incorrect = 0

        for syns in synonyms.values():
            inds = [pathway_ids.index(s) for s in syns]
            if len(inds) > 1:
                labels = [cluster_labels[i] for i in inds]
                if len(set(labels)) > 1:
                    num_incorrect += 1

        correctness = (len(synonyms) - num_incorrect) / len(synonyms)

        num_penalty = 0
        num_total = 0

        for inds in groups:

            path_ids = [pathway_ids[i] for i in inds]
            path_kbs = [pathway_utils.get_pathway_kb(pid) for pid in path_ids]
            num_overlap = len(inds) - len(set(path_kbs))

            num_penalty += num_overlap
            num_total += len(inds)

        penalty = num_penalty / num_total

        return correctness, penalty

    def _remove_constraint_breaking(self, s_list, cl):
        """
        Remove scores from score list that break constraints of clusters
        :param s_list:
        :param cl:
        :return:
        """
        new_list = []

        for pw_id, pid1, pid2, score in s_list:

            skip = False

            # continue if pathways already in the appropriate cluster
            if any([{pid1, pid2}.issubset(set(c)) for c in cl[pw_id]]):
                skip = True
                continue
            else:
                # get databases of pathways
                kb1 = pathway_utils.get_pathway_kb(pid1)
                kb2 = pathway_utils.get_pathway_kb(pid2)

                # continue if one pathway in cluster and other violates kb rule
                for c in cl[pw_id]:
                    kbs = [pathway_utils.get_pathway_kb(p) for p in c]

                    # check if pathways violate rule
                    if pid1 in c and pid2 not in c:
                        if kb2 in kbs:
                            skip = True
                            continue
                    elif pid2 in c and pid1 not in c:
                        if kb1 in kbs:
                            skip = True
                            continue
                    else:
                        if kb1 in kbs or kb2 in kbs:
                            skip = True
                            continue

            # append if not need to continue
            if not skip:
                new_list.append((pw_id, pid1, pid2, score))

        return new_list

    def _select_best_alignments(self, scores: List, top_k=10, threshold=0.25) -> Dict:
        """
        Select best alignments for each PW class grouping
        :param scores:
        :param top_k:
        :param threshold:
        :return:
        """
        clusters = defaultdict(list)

        for pw_id, pw_cls in self.pw.items():
            syn_pathways = pathway_utils.get_pw_synonyms(pw_cls['synonyms'])
            if syn_pathways:
                clusters[pw_id].append(set(syn_pathways))

        restricted = [
            'http://purl.obolibrary.org/obo/PW_0000002',    # metabolic
            'http://purl.obolibrary.org/obo/PW_0000013',    # disease
            'http://purl.obolibrary.org/obo/PW_0000754',    # drug
            'http://purl.obolibrary.org/obo/PW_0000004',    # regulatory
            'http://purl.obolibrary.org/obo/PW_0000003'     # signaling
        ]

        scores = [
            (pw_id, pid1, pid2, np.mean([np.mean([s_score, o_score]), a_score]))
            for pw_id, pid1, pid2, s_score, o_score, a_score in scores if pw_id not in restricted
        ]
        scores.sort(key=lambda x: x[3], reverse=True)
        iter = 0

        print('Clustering results...')

        while scores:
            scores = self._remove_constraint_breaking(scores, clusters)
            print('\tIter: {}, num pairs: {}'.format(iter, len(scores)))

            if scores[0][3] < threshold:
                break

            for pw_id, pid1, pid2, score in scores[:top_k]:
                pid1_in = [pid1 in cl for cl in clusters[pw_id]]
                pid2_in = [pid2 in cl for cl in clusters[pw_id]]

                pid1_in = list(itertools.compress(range(len(pid1_in)), pid1_in))
                pid2_in = list(itertools.compress(range(len(pid2_in)), pid2_in))

                if pid1_in and not pid2_in:
                    clusters[pw_id][pid1_in[0]].add(pid2)
                elif pid2_in and not pid1_in:
                    clusters[pw_id][pid2_in[0]].add(pid1)
                elif pid1_in and pid2_in:
                    cl_keep = pid1_in[0]
                    cl_merge = pid2_in[0]
                    if cl_keep == cl_merge:
                        continue
                    else:
                        clusters[pw_id][cl_keep].update(clusters[pw_id][cl_merge])
                        del clusters[pw_id][cl_merge]
                else:
                    clusters[pw_id].append({pid1, pid2})

            iter += 1

        print('Adding singleton pathways...')
        done_paths = set([])
        for clust_sets in clusters.values():
            for cl_set in clust_sets:
                done_paths.update(cl_set)

        num_singleton = 0
        for pid in self.pathways:
            if pid not in done_paths and len(self.pathways[pid][1]) >= 15:
                best_matches = self.best_pw_matches[pid]
                if best_matches:
                    pw_id_match = best_matches[0][1]
                    singleton_name = '{}_{}_{}'.format(
                        pw_id_match.upper(),
                        pathway_utils.get_pathway_kb(pid).upper(),
                        self.kb_names[pid].upper().replace(' ', '_')
                    )
                    clusters[singleton_name].append({pid})
                else:
                    singleton_name = '{}_{}'.format(
                        pathway_utils.get_pathway_kb(pid).upper(),
                        self.kb_names[pid].upper().replace(' ', '_')
                    )
                    clusters[singleton_name].append({pid})
                num_singleton += 1

        paths = PathhierPaths()
        outfile = os.path.join(paths.output_dir, 'gene_set_clusters.pickle')
        pickle.dump(clusters, open(outfile, 'wb'))

        print('\tNumber singleton gene sets added: {}'.format(num_singleton))

        print('Forming gene sets...')
        gene_sets = dict()

        for gs_name, cl_sets in clusters.items():
            if gs_name in self.pw:
                pw_name = self.pw[gs_name]['name']
                pw_short = gs_name.split('/')[-1]
                gs_name_header = '{}_{}'.format(pw_short, pw_name.replace(' ', '_').upper())
            else:
                gs_name_header = gs_name

            cl_keep = [cs for cs in cl_sets if len(cs) > 0]

            for i, clust in enumerate(cl_keep):
                if len(cl_keep) > 1:
                    gene_sets['{}_{}'.format(gs_name_header, i + 1)] = list(clust)
                else:
                    gene_sets[gs_name_header] = list(clust)

        return gene_sets

    def _map_xrefs_to_gene_symbols(self, xref_list):
        """
        Map each set of xrefs to gene symbol
        :param xref_list:
        :return:
        """
        gene_symbols = set([])

        for xrefs in xref_list:

            # add gene symbols corresponding to ensembl ids
            ensembl_ids = [x for x in xrefs if x.startswith('Ens') or x.startswith('ENS')]
            for ens_id in ensembl_ids:
                if ':' in ens_id:
                    _, ens_id = ens_id.split(':')
                if ens_id in self.ensembl_dict:
                    gene_symbols.add(self.ensembl_dict[ens_id])

            # add gene symbols corresponding to uniprot ids if
            for uniprot_id in [x for x in xrefs if x.startswith('Uni')]:
                uni_db, uni_id = uniprot_id.split(':')
                r = requests.get('http://webservice.bridgedb.org/Human/xrefs/{}/{}?dataSource={}'.format(
                    constants.BRIDGEDB_KEYS[uni_db],
                    uni_id,
                    constants.BRIDGEDB_KEYS['Ensembl']
                ))

                mapped_ids = r.text.split('\n')[:-1]

                for ens_id in [m.split('\t')[0] for m in mapped_ids]:
                    if ens_id in self.ensembl_dict:
                        gene_symbols.add(self.ensembl_dict[ens_id])

        return list(gene_symbols)

    def _make_gene_set(self, p_list: List) -> List:
        """
        Make a gene set based on pathways in pair list
        :param pair_list:
        :return:
        """
        all_ents = [self.pathways[pid][1] for pid in p_list if pid in self.pathways]
        keep_ents = []
        gene_set = []

        for ent_dict in all_ents:
            for ent_id, ent_vals in ent_dict.items():
                if ent_vals['obj_type'] == 'BiochemicalReaction' \
                        or ent_vals['obj_type'] == 'Protein' \
                        or ent_vals['obj_type'] == 'Complex' \
                        or ent_vals['obj_type'] == 'Group':
                    all_xrefs = ent_vals['xrefs'].union(ent_vals['secondary_xrefs']).union(ent_vals['bridgedb_xrefs'])
                    keep_ents.append(all_xrefs)

        if keep_ents:
            gene_set = self._map_xrefs_to_gene_symbols(keep_ents)
            gene_set = list(gene_set)
            gene_set.sort()

        return gene_set

    def _output_gene_sets(self, gs: Dict) -> None:
        """
        Output gene sets to gmt file
        :param gs:
        :return:
        """
        gene_sets = []

        sorted_gs = [(k, v) for k, v in gs.items()]
        sorted_gs.sort(key=lambda x: x[0])

        for k, v in sorted_gs:
            gene_sets.append((k.replace('/', '_'), 'https://github.com/lucylw/pathhier', v))

        pickle.dump(gene_sets, open(self.output_pickle, 'wb'))
        pathway_utils.generate_gmt_file(self.output_file, gene_sets)

    def generate_gene_sets(self):
        """
        Generate gene sets based on alignment scores and mappings
        :return:
        """
        pw_cls_dict, score_list = self._organize_by_pw_class()
        groups_to_merge = self._select_best_alignments(score_list)

        pw_gene_sets = dict()

        for k, path_list in tqdm.tqdm(groups_to_merge.items()):
            gs = self._make_gene_set(path_list)
            if gs:
                if (len(path_list) == 1 and len(gs) >= constants.GENE_SET_MINIMUM_SIZE) or (len(path_list) > 1):
                    pw_gene_sets[k] = gs

        print('Total gene sets: {}'.format(len(pw_gene_sets)))
        gs_lengths = [len(gs) for gs in pw_gene_sets.values()]
        print('Average gene set size: {}'.format(np.mean(gs_lengths)))
        print('Max gene set size: {}'.format(np.max(gs_lengths)))
        print('Min gene set size: {}'.format(np.min(gs_lengths)))

        self._output_gene_sets(pw_gene_sets)


if __name__ == '__main__':
    gs_generator = GeneSetGenerator()
    gs_generator.generate_gene_sets()




