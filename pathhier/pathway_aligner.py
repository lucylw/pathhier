import os
import sys
import csv
import json
import tqdm
import pickle
import itertools
import requests
from typing import List, Dict, Tuple
import numpy as np
from copy import copy
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from bioservices.chebi import ChEBI
from bioservices.uniprot import UniProt
from bioservices.kegg import KEGG

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB, Pathway, Entity
import pathhier.constants as constants
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.bioservices_utils as bioservices_utils
import pathhier.utils.string_utils as string_utils
import pathhier.utils.base_utils as base_utils


# class for clustering pathways based on the output of the PW alignment algorithm
class PathAligner:
    def __init__(self, pathway_pair_file):
        """
        Initialize class
        """
        paths = PathhierPaths()
        self.pathway_pairs = self._load_pathway_pairs(pathway_pair_file)

        # load KBs
        self.kbs = dict()
        for kb_name in paths.all_kb_paths:
            print("Loading {}...".format(kb_name))
            kb_file_path = os.path.join(paths.processed_data_dir, 'kb_{}.pickle'.format(kb_name))
            assert (os.path.exists(kb_file_path))
            kb = PathKB(kb_name)
            kb = kb.load_pickle(kb_name, kb_file_path)
            self.kbs[kb_name] = kb

        # load chebi/uniprot lookup dicts
        print("Loading ChEBI and UniProt lookup dicts...")
        self.chebi_lookup = pickle.load(open(os.path.join(paths.processed_data_dir, 'chebi_lookup.pickle'), 'rb'))
        self.uniprot_lookup = pickle.load(open(os.path.join(paths.processed_data_dir, 'uniprot_lookup.pickle'), 'rb'))

        # create bioservices services
        self.chebi_db = ChEBI()
        self.uniprot_db = UniProt()

        # other
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.STOP = set([w for w in stopwords.words('english')])

    @staticmethod
    def _load_pathway_pairs(pair_file):
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

        print('{} pairs loaded.'.format(len(all_pairs)))
        return all_pairs

    @staticmethod
    def _convert_ent_to_dict(ent: Entity) -> Dict:
        """
        Convert entity to dictionary representation
        :param ent:
        :return:
        """
        components = []

        if ent.obj_type == 'Group':
            aliases = []
            definition = []
            xrefs = ent.xrefs
            for mem in ent.members:
                if type(mem) == str:
                    components.append(mem)
                else:
                    components.append(mem.uid)
        elif ent.obj_type == 'Complex':
            aliases = ent.aliases
            definition = ent.definition
            xrefs = ent.xrefs
            for mem in ent.components:
                if type(mem) == str:
                    components.append(mem)
                else:
                    components.append(mem.uid)
        elif ent.obj_type == 'BiochemicalReaction':
            aliases = ent.aliases
            definition = ent.definition
            xrefs = ent.xrefs
            components = [left.uid for left in ent.left] \
                         + [right.uid for right in ent.right] \
                         + [controller.uid for controller in ent.controllers] \
                         + [other.uid for other in ent.other]
        else:
            aliases = ent.aliases
            definition = ent.definition
            xrefs = set(pathway_utils.clean_xrefs(
                ent.xrefs, constants.ENTITY_XREF_AVOID_TERMS)
            )

        return {
            'name': ent.name,
            'aliases': aliases,
            'definition': definition,
            'obj_type': ent.obj_type,
            'xrefs': xrefs,
            'secondary_xrefs': [],
            'bridgedb_xrefs': [],
            'parent_xrefs': [],
            'synonym_xrefs': [],
            'related_terms': [],
            'components': components
        }

    def _get_bridgedb_synonym_identifiers(self, xref):
        """
        Get synonym identifiers from Bridge DB
        :param xref:
        :return:
        """
        xref_db, xref_id = xref.split(':')
        syn_ids = []

        if xref_db in constants.BRIDGEDB_MAP:
            keep_dbs = constants.BRIDGEDB_MAP[xref_db]
            for db in keep_dbs:
                r = requests.get('http://webservice.bridgedb.org/Human/xrefs/{}/{}?dataSource={}'.format(
                        constants.BRIDGEDB_KEYS[xref_db],
                        xref_id,
                        constants.BRIDGEDB_KEYS[db]
                ))

                mapped_ids = r.text.split('\n')[:-1]

                if len(mapped_ids) > 0:
                    syn_ids += ['{}:{}'.format(db, m.split('\t')[0]) for m in mapped_ids]

        return syn_ids

    def _enrich_entity(self, ent: Dict):
        """
        Enrich entities with more info from xref databases
        :param ent:
        :return:
        """
        db_name = []
        definition = []
        synonyms = []
        secondary_ids = []
        parents = []
        conjugate_acids = []
        conjugate_bases = []
        tautomers = []
        gene_names = []
        bridgedb_ids = []

        for xref in ent['xrefs']:
            if 'chebi' in xref.lower():
                db_name.append(self.chebi_lookup[xref]['name'])
                definition.append(self.chebi_lookup[xref]['definition'])
                synonyms += self.chebi_lookup[xref]['synonyms']
                secondary_ids += self.chebi_lookup[xref]['secondary_ids']
                parents += self.chebi_lookup[xref]['parents']
                conjugate_acids += self.chebi_lookup[xref]['conjugate_acids']
                conjugate_bases += self.chebi_lookup[xref]['conjugate_bases']
                tautomers += self.chebi_lookup[xref]['tautomers']
            elif 'uniprot' in xref.lower():
                db_name.append(self.uniprot_lookup[xref]['name'])
                synonyms += self.uniprot_lookup[xref]['synonyms']
                secondary_ids += self.uniprot_lookup[xref]['secondary_ids']
                gene_names += self.uniprot_lookup[xref]['gene_names']

            bridgedb_ids += self._get_bridgedb_synonym_identifiers(xref)

        ent['secondary_xrefs'] = set(secondary_ids).difference(ent['xrefs'])
        ent['bridgedb_xrefs'] = set(bridgedb_ids).difference(ent['xrefs']).difference(ent['secondary_xrefs'])
        ent['parent_xrefs'] = set(parents).difference(ent['xrefs'])
        ent['synonym_xrefs'] = set(conjugate_acids + conjugate_bases + tautomers).difference(ent['xrefs'])
        ent['related_terms'] = set(gene_names)

        return ent

    def _enrich_pathway(self, pathway: Pathway) -> Tuple(List, Dict):
        """
        Enrich all entities in pathway
        :param pathway:
        :return:
        """
        ent_ids = []
        enriched_ents = dict()

        for ent in pathway.entities:
            ent_ids.append(ent.uid)
            ent_dict = self._convert_ent_to_dict(ent)
            if ent.obj_type in constants.ENRICH_ENTITY_TYPES:
                enriched_ents[ent.uid] = self._enrich_entity(ent_dict)
            else:
                enriched_ents[ent.uid] = ent_dict

        return ent_ids, enriched_ents

    def _get_attributes(self, ents: Dict):
        """
        Get node and edge for two pathways
        :param ents:
        :return:
        """
        node_sim_measure = [
            'equivalence',
            'set_overlap',
            'jaccard',
            'jaccard',
            'jaccard',
            'set_overlap',
            'set_overlap',
            'set_overlap',
            'set_overlap',
            'set_overlap',
            'set_overlap',
            'equivalence'
        ]

        node_attrib = []

        for uid, ent in ents.items():
            name_lower = ent['name'].lower()
            aliases_lower = [a.lower() for a in ent['aliases']]
            node_attrib.append((
                name_lower,
                set(aliases_lower),
                set(string_utils.tokenize_string(name_lower, self.tokenizer, {})),
                set(base_utils.flatten([string_utils.tokenize_string(a, self.tokenizer, {}) for a in aliases_lower])),
                set(base_utils.flatten([string_utils.tokenize_string(a, self.tokenizer, self.STOP) for a in ent['definition']])),
                set(ent['xrefs']),
                set(ent['secondary_xrefs']),
                set(ent['bridgedb_xrefs']),
                set(ent['parent_xrefs']),
                set(ent['synonym_xrefs']),
                set(ent['related_terms']),
                ent['obj_type']
            ))

        return node_sim_measure, node_attrib

    def _compute_graph_representation(self, pathway: Pathway):
        """
        Convert pathway to graph representation
        :param pathway:
        :return:
        """
        num_ents = len(pathway.entities)
        ent_list = [ent.uid for ent in pathway.entities]

        adj_matrix = np.zeros([num_ents, num_ents])
        edge_attrib = np.zeros([num_ents, num_ents])

        for ent1, prop, ent2 in pathway.relations:
            ent1_ind = ent_list.index(ent1)
            ent2_ind = ent_list.index(ent2)
            if ent1_ind and ent2_ind:
                adj_matrix[ent1_ind][ent2_ind] = 1
                adj_matrix[ent2_ind][ent1_ind] = 1
            edge_attrib[ent1_ind][ent2_ind] = constants.EDGE_TYPE_ATTRIB[prop]
            edge_attrib[ent2_ind][ent1_ind] = constants.EDGE_TYPE_ATTRIB[prop]

        edge_attrib = np.stack(edge_attrib)

        return adj_matrix, edge_attrib

    @staticmethod
    def _compute_node_similarities(n1, n2, sim_funcs, attrib1, attrib2):
        """
        Compute cosine cross-similarity using similarity functions provided
        :param n1:
        :param n2:
        :param sim_funcs:
        :param attrib1:
        :param attrib2:
        :return:
        """
        sim = np.zeros((n1 * n2, n1 * n2))

        for a, entry1 in enumerate(attrib1):
            for x, entry2 in enumerate(attrib2):
                sim_vec = [fx((val1, val2)) for fx, val1, val2 in zip(sim_funcs, entry1, entry2)]
                sim[n1 * a + x][n1 * a + x] = sum(sim_vec)

        return sim

    @staticmethod
    def _compute_edge_similarities(n1, n2, attrib1, attrib2):
        """
        Compute tensor product of edge attributes
        :param n1:
        :param n2:
        :param attrib1:
        :param attrib2:
        :return:
        """
        sim = np.zeros((n1 * n2, n1 * n2))

        for a, row1 in enumerate(attrib1):
            for b, val1 in enumerate(row1):
                for x, row2 in enumerate(attrib2):
                    for y, val2 in enumerate(row2):
                        if val1 > 0 and val2 > 0:
                            if val1 == val2:
                                sim[n1 * a + x][n2 * b + y] = 1.
                            else:
                                sim[n1 * a + x][n2 * b + y] = 0.5
        return sim

    def _compute_degree_matrix(self, nodes, edges, adj1, adj2):
        """
        Compute degree matrix
        :param nodes:
        :param edges:
        :param adj1:
        :param adj2:
        :return:
        """
        adj_tensor_prod = np.tensordot(adj1, adj2)
        degree = np.multiply(np.multiply(nodes, edges), adj_tensor_prod)
        return degree

    def align_pair(self, path1: Pathway, path2: Pathway):
        """
        Align a pair of pathways
        :param path1:
        :param path2:
        :return:
        """
        p1_ents, p1_enriched = self._enrich_pathway(path1)
        p2_ents, p2_enriched = self._enrich_pathway(path2)

        p1_edgelist, p1_bow_ent_embeddings = self._process_pathway_graph(p1_ents, path1)
        p2_edgelist, p2_bow_ent_embeddings = self._process_pathway_graph(p2_ents, path2)

        p1_s2v_embeddings = self._get_struc2vec_embeddings(p1_edgelist)
        p2_s2v_embeddings = self._get_struc2vec_embeddings(p2_edgelist)

        xref_alignments = self._get_prelim_alignments(p1_ents, p2_ents, path1, path2)

        results = self._run_graph_aligner(
            xref_alignments,
            p1_bow_ent_embeddings,
            p2_bow_ent_embeddings,
            p1_s2v_embeddings,
            p2_s2v_embeddings
        )

        return results

    def align_pathways(self, save_path=None):
        """
        Align all pathway pairs
        :return:
        """
        all_alignments = []

        for sim_score, overlap, pw_id, kb1_id, kb2_id in self.pathway_pairs:
            pathway1 = pathway_utils.get_corresponding_pathway(self.kbs, kb1_id)
            pathway2 = pathway_utils.get_corresponding_pathway(self.kbs, kb2_id)

            if pathway1 and pathway2 and pathway1.entities and pathway2.entities:
                alignment = self.align_pair(pathway1, pathway2)

                if alignment:
                    align_score, mapping = alignment
                    all_alignments.append([align_score, kb1_id, kb2_id, mapping])

        if save_path:
            pickle.dump(all_alignments, open(save_path, 'wb'))

        return all_alignments

    def kb_stats(self):
        """
        Compute KB stats
        :return:
        """
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        max_ents = 200

        for i, kb_name in enumerate(self.kbs):
            row_num = int(np.floor(i / 4))
            col_num = i % 4

            num_paths = len(self.kbs[kb_name].pathways)
            num_ents = []
            for pathway in self.kbs[kb_name].pathways:
                num_ents.append(len(pathway.entities))

            counts = Counter(num_ents)
            x = [k for k in counts.keys() if k != 0]
            y = [counts[num] for num in x]

            axes[row_num][col_num].bar(x, y, width=1)
            axes[row_num][col_num].set_xlim([0, max_ents])
            axes[row_num][col_num].set_title('{} ({})'.format(kb_name.upper(), num_paths))

            if max(y) < 10:
                yint = range(min(y), int(np.ceil(max(y)) + 1))
                axes[row_num][col_num].set_yticks(yint)

        axes[1][3].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.show()


if __name__ == '__main__':
    path_pair_file = '/Users/lwang/git/pathhier/output/model_output/clustered_groups.tsv'
    aligner = PathAligner(path_pair_file)
    # aligner.kb_stats()
    aligner.align_pathways()
