import os
import sys
import csv
import gzip
import tqdm
import json
import string
import random
import pickle
import requests
from copy import copy
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from subprocess import call

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process

from bioservices.chebi import ChEBI
from bioservices.uniprot import UniProt

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB, Pathway, Entity
import pathhier.constants as constants
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.string_utils as string_utils
import pathhier.utils.base_utils as base_utils


# class for clustering pathways based on the output of the PW alignment algorithm
class PathAligner:
    def __init__(
            self,
            pathway_pair_file,
            s2v_path,
            start_ind=0,
            end_ind=None,
            w2v_file=None,
            ft_file=None
    ):
        """
        Initialize class
        """
        paths = PathhierPaths()

        # load pathway pairs
        print('Loading pathway pairs...')
        self.pathway_pairs = self._load_pathway_pairs(pathway_pair_file)
        print('{} pairs to align.'.format(len(self.pathway_pairs)))

        self.start_ind = start_ind
        if not end_ind:
            self.end_ind = len(self.pathway_pairs)
        else:
            self.end_ind = end_ind

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

        # tokenizers and stop words
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.STOP = set([w for w in stopwords.words('english')])

        # load word vectors
        print("Loading word2vec vectors...")
        self.w2v = dict()
        if w2v_file and os.path.exists(w2v_file):
            with gzip.open(w2v_file, 'rb') as f:
                content = f.read().split(b'\n')
            for l in content:
                vec = l.split()
                if vec:
                    self.w2v[vec[0].decode('-utf-8')] = [float(val) for val in vec[1:]]

        # load fasttext vectors
        print("Loading fasttext vectors...")
        self.fasttext = dict()
        if ft_file and os.path.exists(ft_file):
            with gzip.open(ft_file, 'rb') as f:
                content = f.read().split(b'\n')
            for l in content:
                vec = l.split()
                if vec:
                    self.fasttext[vec[0].decode('-utf-8')] = [float(val) for val in vec[1:]]

        # struc2vec path
        self.s2v_path = s2v_path
        assert os.path.exists(self.s2v_path)

        # temp directory
        self.temp_dir = os.path.join(paths.base_dir, 'temp')
        if not(os.path.exists(self.temp_dir)):
            os.mkdir(self.temp_dir)

        # alignment directory
        self.alignment_dir = os.path.join(paths.output_dir, 'alignments')
        if not(os.path.exists(self.alignment_dir)):
            os.mkdir(self.alignment_dir)

        # load enrich dict
        all_pathway_ids = list(set([x[3] for x in self.pathway_pairs] + [x[4] for x in self.pathway_pairs]))
        all_pathway_ids.sort()
        self.pathway_ind_mapping = {pathway_id: i for i, pathway_id in enumerate(all_pathway_ids)}

        self.enrich_file_path = os.path.join(paths.output_dir, 'enriched_entities.txt')
        self.enrich_dict = dict()

        if os.path.exists(self.enrich_file_path):
            with open(self.enrich_file_path, 'r') as f:
                contents = f.read().split('\n')
            for l in contents[:-1]:
                pathway_uid, pathway_enrichment_file = l.split()
                fname_parts = os.path.split(pathway_enrichment_file)
                self.enrich_dict[pathway_uid] = os.path.join(self.temp_dir, fname_parts[-1])

        # load alignment dict
        self.alignment_ind_mapping = {(pair_info[3], pair_info[4]): i for i, pair_info in enumerate(self.pathway_pairs)}
        self.alignment_file_path = os.path.join(paths.output_dir, 'alignment_files.txt')
        self.alignment_dict = dict()

        if os.path.exists(self.alignment_file_path):
            with open(self.alignment_file_path, 'r') as f:
                contents = f.read().split('\n')
            for l in contents[:-1]:
                p1_uid, p2_uid, alignment_file = l.split()
                fname_parts = os.path.split(alignment_file)
                self.alignment_dict[(p1_uid, p2_uid)] = os.path.join(self.alignment_dir, fname_parts[-1])

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
            'xrefs': set(xrefs),
            'secondary_xrefs': set([]),
            'bridgedb_xrefs': set([]),
            'equal_xrefs': set(xrefs),
            'parent_xrefs': set([]),
            'synonym_xrefs': set([]),
            'related_terms': set([]),
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
            if 'chebi' in xref.lower() and xref in self.chebi_lookup:
                db_name.append(self.chebi_lookup[xref]['name'])
                definition.append(self.chebi_lookup[xref]['definition'])
                synonyms += self.chebi_lookup[xref]['synonyms']
                secondary_ids += self.chebi_lookup[xref]['secondary_ids']
                parents += self.chebi_lookup[xref]['parents']
                conjugate_acids += self.chebi_lookup[xref]['conjugate_acids']
                conjugate_bases += self.chebi_lookup[xref]['conjugate_bases']
                tautomers += self.chebi_lookup[xref]['tautomers']
            elif 'uniprot' in xref.lower() and xref in self.uniprot_lookup:
                db_name.append(self.uniprot_lookup[xref]['name'])
                synonyms += self.uniprot_lookup[xref]['synonyms']
                secondary_ids += self.uniprot_lookup[xref]['secondary_ids']
                gene_names += self.uniprot_lookup[xref]['gene_names']
            try:
                bridgedb_ids += self._get_bridgedb_synonym_identifiers(xref)
            except Exception:
                continue

        ent['secondary_xrefs'] = set(secondary_ids)
        ent['bridgedb_xrefs'] = set(bridgedb_ids)
        ent['equal_xrefs'] = ent['xrefs'].union(ent['secondary_xrefs']).union(ent['bridgedb_xrefs'])

        ent['parent_xrefs'] = set(parents)
        ent['synonym_xrefs'] = set(conjugate_acids + conjugate_bases + tautomers)

        ent['related_terms'] = set(gene_names)
        return ent

    def _enrich_pathway(self, pathway: Pathway) -> Tuple[List, Dict]:
        """
        Enrich all entities in pathway
        :param pathway:
        :return:
        """
        ent_ids = [ent.uid for ent in pathway.entities]

        if pathway.uid in self.enrich_dict:
            enriched_ents = pickle.load(open(self.enrich_dict[pathway.uid], 'rb'))
        else:
            enriched_ents = dict()

            for ent in pathway.entities:
                ent_dict = self._convert_ent_to_dict(ent)
                if ent.obj_type in constants.ENRICH_ENTITY_TYPES:
                    enriched_ents[ent.uid] = self._enrich_entity(ent_dict)
                else:
                    enriched_ents[ent.uid] = ent_dict

            pathway_index = self.pathway_ind_mapping[pathway.uid]
            enrich_file = os.path.join(self.temp_dir, 'pathway{}.pickle'.format(pathway_index))
            pickle.dump(enriched_ents, open(enrich_file, 'wb'))

            # add enriched pathway to processed dictionary
            self.enrich_dict[pathway.uid] = enrich_file
            with open(self.enrich_file_path, 'a') as outf:
                outf.write('{} {}\n'.format(pathway.uid, enrich_file))

        return ent_ids, enriched_ents

    @staticmethod
    def _process_pathway_graph(ent_ids: List, pathway: Pathway):
        """
        Generate pathway edgelist
        :param entities:
        :param pathway:
        :return:
        """
        edgelist = []

        for ent1, prop, ent2 in pathway.relations:
            if ent1 in ent_ids and ent2 in ent_ids:
                ent1_ind = ent_ids.index(ent1)
                ent2_ind = ent_ids.index(ent2)
                if ent1_ind and ent2_ind:
                    edgelist.append((ent1_ind, ent2_ind))

        return edgelist

    def _get_bow_embeddings(self, ent_ids: List, entities: Dict):
        """
        Construct BOW embeddings for each entity
        :param ent_ids:
        :param entities:
        :return:
        """
        embeddings = []

        for ent_id in ent_ids:
            alias_tokens = set(base_utils.flatten([
                string_utils.tokenize_string(a.lower(), self.tokenizer, self.STOP)
                for a in entities[ent_id]['aliases'] + [entities[ent_id]['name']]
            ]))

            all_tokens = []

            # get embeddings for each token
            for tok in alias_tokens:
                tok_embedding = []

                if tok in self.w2v:
                    tok_embedding += self.w2v[tok]
                else:
                    tok_embedding += [random.uniform(-1, 1) for _ in range(100)]

                if tok in self.fasttext:
                    tok_embedding += self.fasttext[tok]
                else:
                    tok_embedding += [random.uniform(-1, 1) for _ in range(100)]

                all_tokens.append(tok_embedding)

            # stack and average all token embeddings
            if all_tokens:
                all_tokens = np.stack(all_tokens)
                embeddings.append(np.mean(all_tokens, axis=0))
            else:
                embeddings.append(np.array([random.uniform(-1, 1) for _ in range(200)]))

        return embeddings

    def _get_struc2vec_embeddings(self, ent_ids: List, edgelist: List, edgelist_file: str, output_file: str):
        """
        Compute s2v embeddings
        :param ent_ids:
        :param edgelist:
        :param output_file:
        :return:
        """
        all_ent_inds = list(range(len(ent_ids)))
        connected_inds = set([])

        with open(edgelist_file, 'w') as outf:
            for node1, node2 in edgelist:
                outf.write('{} {}\n'.format(node1, node2))
                connected_inds.add(node1)
                connected_inds.add(node2)
            for node in set(all_ent_inds).difference(connected_inds):
                outf.write('{}\n'.format(node))

        call([self.s2v_path, '--input', edgelist_file, '--output', output_file])

        embeddings = dict()

        with open(output_file, 'r') as f:
            contents = f.read().split('\n')

        for l in contents[1:]:
            parts = l.split()
            if len(parts) > 1:
                embeddings[int(parts[0])] = np.array([float(val) for val in parts[1:]])

        return embeddings

    def _get_prelim_alignments(self, p1_entities, p2_entities, p1_enriched, p2_enriched):
        """
        Construct preliminary alignments using xref and name overlap
        :param p1_entities:
        :param p2_entities:
        :param p1_enriched:
        :param p2_enriched:
        :return:
        """
        prelim_alignments = []
        type_restrictions = []

        for i, ent_id1 in enumerate(p1_entities):

            p1_ent = p1_enriched[ent_id1]
            p1_aliases = set([a.lower() for a in p1_ent['aliases']])
            p1_is_rx = p1_ent['obj_type'] in constants.BIOPAX_RX_TYPES

            for j, ent_id2 in enumerate(p2_entities):
                p2_ent = p2_enriched[ent_id2]

                if p1_ent['equal_xrefs'].intersection(p2_ent['equal_xrefs']) \
                        or p1_ent['synonym_xrefs'].intersection(p2_ent['synonym_xrefs']) \
                        or p1_aliases.intersection(set([a.lower() for a in p2_ent['aliases']])):
                    prelim_alignments.append((i, j))

                p2_is_rx = p2_ent['obj_type'] in constants.BIOPAX_RX_TYPES

                if p1_is_rx != p2_is_rx:
                    type_restrictions.append((i, j))

        return prelim_alignments, type_restrictions

    def _run_graph_aligner(
            self,
            starting_alignment,
            type_restrictions,
            p1_entities,
            p2_entities,
            p1_ent_embeddings,
            p2_ent_embeddings,
            p1_top_embeddings,
            p2_top_embeddings
        ):
        """
        Combine embeddings and use REGAL alignment method
        :param starting_alignment:
        :param type_restrictions:
        :param p1_entities:
        :param p2_entities:
        :param p1_ent_embeddings:
        :param p2_ent_embeddings:
        :param p1_top_embeddings:
        :param p2_top_embeddings:
        :return:
        """
        embedding1 = []
        embedding2 = []

        for i, p1_ent_id in enumerate(p1_entities):
            embedding1.append(np.concatenate([p1_ent_embeddings[i], p1_top_embeddings[i]]))

        for i, p2_ent_id in enumerate(p2_entities):
            embedding2.append(np.concatenate([p2_ent_embeddings[i], p2_top_embeddings[i]]))

        embedding1 = np.stack(embedding1)
        embedding2 = np.stack(embedding2)

        similarity_matrix = cosine_similarity(embedding1, embedding2)

        # normalize similarity matrix to between 0 and 1
        similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / np.ptp(similarity_matrix)

        # set xref matches to 1.
        for x_ind, y_ind in starting_alignment:
            similarity_matrix[x_ind][y_ind] = 1.

        # set type-restricted matches to 0.
        for x_ind, y_ind in type_restrictions:
            similarity_matrix[x_ind][y_ind] = 0.

        return similarity_matrix

    @staticmethod
    def _greedy_align(sim_scores):
        """
        Greedily select maximal alignment
        :param sim_scores:
        :return:
        """
        n1, n2 = sim_scores.shape
        flip = (n1 < n2)

        alignment = copy(sim_scores)

        # transpose if first dimension is smaller
        if flip:
            alignment = alignment.T

        # get locations where alignment is 1.
        pos_alignments = np.transpose(np.nonzero(alignment == 1.))

        matching_inds = []

        for x_ind, y_ind in pos_alignments:
            alignment[x_ind][:] = 0.
            alignment[:][y_ind] = 0.
            matching_inds.append((x_ind, y_ind, 1.))

        cont = True

        # greedily select maximum and set rows and cols to zero
        while cont:
            max_val = np.max(alignment)
            if max_val >= constants.MIN_ALIGNMENT_THRESHOLD:
                x_ind, y_ind = np.unravel_index(alignment.argmax(), alignment.shape)
                matching_inds.append((x_ind, y_ind, max_val))
                alignment[x_ind][y_ind] = 0.

                # append other indices that fall within the epsilon range
                for col_ind, row_val in enumerate(alignment[x_ind][:]):
                    if row_val >= max_val + constants.ALIGNMENT_SCORE_EPSILON:
                        matching_inds.append((x_ind, col_ind, row_val))

                for row_ind, col_val in enumerate(alignment[:][y_ind]):
                    if col_val >= max_val + constants.ALIGNMENT_SCORE_EPSILON:
                        matching_inds.append((row_ind, y_ind, col_val))

                alignment[x_ind][:] = 0.
                alignment[:][y_ind] = 0.
            else:
                alignment = np.zeros(alignment.shape)
                cont = False

        for x_ind, y_ind, score in matching_inds:
            alignment[x_ind][y_ind] = score

        if flip:
            matching_inds = [(y_ind, x_ind, val) for x_ind, y_ind, val in matching_inds]

        return matching_inds

    def align_pair(self, path1: Pathway, path2: Pathway):
        """
        Align a pair of pathways
        :param path1:
        :param path2:
        :return:
        """
        # Process nodes
        p1_ents = [ent.uid for ent in path1.entities]
        p2_ents = [ent.uid for ent in path2.entities]

        p1_enriched = pickle.load(open(self.enrich_dict[path1.uid], 'rb'))
        p2_enriched = pickle.load(open(self.enrich_dict[path2.uid], 'rb'))

        xref_alignments, type_restrictions = self._get_prelim_alignments(p1_ents, p2_ents, p1_enriched, p2_enriched)

        p1_bow_ent_embeddings = self._get_bow_embeddings(p1_ents, p1_enriched)
        p2_bow_ent_embeddings = self._get_bow_embeddings(p2_ents, p2_enriched)

        # Process edges
        p1_edgelist = self._process_pathway_graph(p1_ents, path1)
        p2_edgelist = self._process_pathway_graph(p2_ents, path2)

        # generate random string
        rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # s2v files
        p1_s2v_file = os.path.join(self.temp_dir, 'p1_{}.emb'.format(rand_str))
        p2_s2v_file = os.path.join(self.temp_dir, 'p2_{}.emb'.format(rand_str))
        temp_edgelist_file = os.path.join(self.temp_dir, 'pathway_{}.edgelist'.format(rand_str))

        p1_s2v_embeddings = self._get_struc2vec_embeddings(p1_ents, p1_edgelist, temp_edgelist_file, p1_s2v_file)
        p2_s2v_embeddings = self._get_struc2vec_embeddings(p2_ents, p2_edgelist, temp_edgelist_file, p2_s2v_file)

        # Align based on computed embeddings
        sim_scores = self._run_graph_aligner(
            xref_alignments,
            type_restrictions,
            p1_ents,
            p2_ents,
            p1_bow_ent_embeddings,
            p2_bow_ent_embeddings,
            p1_s2v_embeddings,
            p2_s2v_embeddings
        )

        os.remove(p1_s2v_file)
        os.remove(p2_s2v_file)
        os.remove(temp_edgelist_file)

        # Greedily select alignments from similarity scores
        results = self._greedy_align(sim_scores)

        matches = []
        for p1_ind, p2_ind, score in results:
            p1_ent_id = p1_ents[p1_ind]
            p2_ent_id = p2_ents[p2_ind]
            p1_name = p1_enriched[p1_ent_id]['name']
            p2_name = p2_enriched[p2_ent_id]['name']
            matches.append((score, p1_ent_id, p2_ent_id, p1_name, p2_name))

        matches.sort(key=lambda x: x[0], reverse=True)

        if matches:
            match_score = np.mean([m[0] for m in matches]) * len(matches) / (0.5 * (len(p1_ents) + len(p2_ents)))
        else:
            match_score = 0.

        return match_score, matches

    def align_pathway_process(self, pathway_pairs_split, offset=0, verbose=False):
        """
        Process to run for pathway alignment
        :param pathway_pairs_split:
        :param verbose:
        :return:
        """
        print('Starting process parsing {} pathway pairs...'.format(len(pathway_pairs_split)))
        skipped = []

        for sim_score, overlap, pw_id, kb1_id, kb2_id in tqdm.tqdm(pathway_pairs_split):
            pathway1 = pathway_utils.get_corresponding_pathway(self.kbs, kb1_id)
            pathway2 = pathway_utils.get_corresponding_pathway(self.kbs, kb2_id)

            # already processed, skip
            if (pathway1.uid, pathway2.uid) in self.alignment_dict:
                continue

            # skip if no entities
            if not pathway1.entities:
                print('SKIPPING: {} has no entities.'.format(kb1_id))
                continue

            if not pathway2.entities:
                print('SKIPPING: {} has no entities.'.format(kb2_id))
                continue

            # print pathway info
            if verbose:
                print()
                print('{}: {}'.format(kb1_id, pathway1.name))
                print('{}: {}'.format(kb2_id, pathway2.name))

            try:
                # process new pathway pair
                align_score, mapping = self.align_pair(pathway1, pathway2)

                if align_score > 0.:
                    alignment_file_name = os.path.join(self.alignment_dir, 'alignment{}.pickle'.format(
                        self.alignment_ind_mapping[(pathway1.uid, pathway2.uid)]
                    ))

                    pickle.dump([align_score, mapping], open(alignment_file_name, 'wb'))

                    self.alignment_dict[(pathway1.uid, pathway2.uid)] = alignment_file_name

                    with open(self.alignment_file_path, 'a') as outf:
                        outf.write('{} {} {}\n'.format(pathway1.uid, pathway2.uid, alignment_file_name))

                    if verbose:
                        print('Alignment score: {:.2f}'.format(align_score))
                        print('---Alignment---')
                        for score, p1_id, p2_id, p1_name, p2_name in mapping:
                            print('{:.2f}\t{}\t{}'.format(score, p1_name, p2_name))
                else:
                    print('Alignment score = 0. No alignment.')

            except Exception:
                print('ERROR occurred: skipping {} and {}'.format(kb1_id, kb2_id))
                skipped.append((kb1_id, kb2_id))
                continue

        skipped_file = os.path.join(self.temp_dir, 'skipped_pathway_pairs_{}.pickle'.format(offset))
        pickle.dump(skipped, open(skipped_file, 'wb'))

    def align_pathways(self):
        """
        Align all pathway pairs
        :return:
        """
        self.align_pathway_process(
            self.pathway_pairs[self.start_ind:self.end_ind],
            offset=self.start_ind,
            verbose=False
        )

    def enrich_only(self):
        """
        Enrich all pathways first
        :return:
        """
        skip_file = os.path.join(self.temp_dir, 'skipped_for_enrichment.pickle')

        if os.path.exists(skip_file):
            pathway_list = pickle.load(open(skip_file, 'rb'))
        else:
            pathway_list = list(self.pathway_ind_mapping.keys())

        skipped = []

        for pathway_id in tqdm.tqdm(pathway_list):
            pathway = pathway_utils.get_corresponding_pathway(self.kbs, pathway_id)
            if pathway:
                try:
                    self._enrich_pathway(pathway)
                except Exception:
                    skipped.append(pathway_id)

        pickle.dump(skipped, open(skip_file, 'wb'))
        return

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

            print('{}: {} zeros'.format(kb_name, counts[0]))

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