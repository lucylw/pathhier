import os
import sys
import csv
import json
import tqdm
import pickle
import itertools
import requests
from typing import List, Dict
from collections import defaultdict

import numpy as np

from bioservices.chebi import ChEBI
from bioservices.uniprot import UniProt
from bioservices.kegg import KEGG

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB, Pathway, Entity
import pathhier.constants as constants
import pathhier.utils.pathway_utils as pathway_utils
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

        # create bioservices services
        self.chebi_db = ChEBI()
        self.uniprot_db = UniProt()

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
        if ent.obj_type == 'Group':
            aliases = []
            definition = []
            xrefs = ent.xrefs
            components = [mem.uid for mem in ent.members]
        elif ent.obj_type == 'Complex':
            aliases = ent.aliases
            definition = ent.definition
            xrefs = ent.xrefs
            components = [mem.uid for mem in ent.components]
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
            components = []

        return {
            'name': ent.name,
            'aliases': aliases,
            'definition': definition,
            'obj_type': ent.obj_type,
            'xrefs': xrefs,
            'secondary_xrefs': [],
            'bridgedb_xrefs': [],
            'components': components
        }

    def _get_secondary_accession_identifiers(self, xref):
        """
        Get secondary accession identifiers from source DB
        :param xref:
        :return:
        """
        xref_db, xref_id = xref.split(':')

        if xref_db.lower() == 'chebi':
            chebi_ent = self.chebi_db.getCompleteEntity(xref_id)
            return chebi_ent.SecondaryChEBIIds
        elif xref_db.lower() == 'uniprot':
            uniprot_ent = self.uniprot_db.retrieve(xref_id, frmt='txt')
            accession_lines = [l for l in uniprot_ent.split('\n') if l.startswith('AC')]
            secondaries = []
            for ac_line in accession_lines:
                secondaries += ['UniProt:{}'.format(uid[:-1]) for uid in ac_line.split()[1:]]
            return secondaries
        return []

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
        secondary_ids = []
        bridgedb_ids = []

        for xref in ent['xrefs']:
            secondary_ids += self._get_secondary_accession_identifiers(xref)
            bridgedb_ids += self._get_bridgedb_synonym_identifiers(xref)

        ent['secondary_xrefs'] = set(secondary_ids).difference(ent['xrefs'])
        ent['bridgedb_xrefs'] = set(bridgedb_ids).difference(ent['xrefs']).difference(ent['secondary_xrefs'])

        if ent['obj_type'] == 'SmallMolecule':
            from pprint import pprint
            pprint(ent)
            input()

        return ent

    def _enrich_pathway(self, path_ents: List[Entity]) -> Dict:
        """
        Enrich all entities in pathway
        :param path_ents:
        :return:
        """
        enriched_ents = dict()

        for ent in path_ents:
            ent_dict = self._convert_ent_to_dict(ent)
            if ent.obj_type in constants.ENRICH_ENTITY_TYPES:
                enriched_ents[ent.uid] = self._enrich_entity(ent_dict)
            else:
                enriched_ents[ent.uid] = ent_dict

        return enriched_ents

    def _compute_attributes(self, p1_ents: Dict, p2_ents: Dict):
        """
        Compute attributes for two pathways
        :param p1_ents:
        :param p2_ents:
        :return:
        """
        raise NotImplementedError("Not implemented...")

    def _compute_graph_representation(self, pathway: Pathway):
        """
        Convert pathway to graph representation
        :param pathway:
        :return:
        """
        raise NotImplementedError("Not implemented...")

    def _run_graph_aligner(self, graph1, attrib1, graph2, attrib2):
        """
        Run graph alignment over pathway pair
        :param graph1:
        :param attrib1:
        :param graph2:
        :param attrib2:
        :return:
        """
        raise NotImplementedError("Not implemented...")

    def align_pair(self, path1: Pathway, path2: Pathway):
        """
        Align a pair of pathways
        :param path1:
        :param path2:
        :return:
        """
        p1_enriched = self._enrich_pathway(path1.entities)
        p2_enriched = self._enrich_pathway(path2.entities)

        p1_attrib, p2_attrib = self._compute_attributes(p1_enriched, p2_enriched)

        p1_graph = self._compute_graph_representation(path1.relations)
        p2_graph = self._compute_graph_representation(path2.relations)

        results = self._run_graph_aligner(p1_graph, p1_attrib, p2_graph, p2_attrib)

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
            alignment = self.align_pair(pathway1, pathway2)

            if alignment:
                align_score, mapping = alignment
                all_alignments.append([align_score, kb1_id, kb2_id, mapping])

        if save_path:
            pickle.dump(open(save_path, 'wb'), all_alignments)

        return all_alignments


if __name__ == '__main__':
    path_pair_file = '/Users/lwang/git/pathhier/output/model_output/clustered_groups.tsv'
    aligner = PathAligner(path_pair_file)

    import pdb
    pdb.set_trace()

    aligner.align_pathways()
