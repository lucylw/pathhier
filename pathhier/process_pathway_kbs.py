import os
import sys
import json
import tqdm
import itertools
import requests
from lxml import etree
from collections import defaultdict
from bioservices import *

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.constants as constants


# class for loading all pathways and processing everything in succession
class PathwayKBLoader:
    def __init__(self):
        paths = PathhierPaths()
        self.path_kb_dirs = {
            "humancyc": paths.humancyc_raw_data_dir,
            "kegg": paths.kegg_raw_data_dir,
            "panther": paths.panther_raw_data_dir,
            "pid": paths.pid_raw_data_dir,
            "reactome": paths.reactome_raw_data_dir,
            # "smpdb": paths.smpdb_raw_data_dir,  # non-unique uids!!!
            "wikipathways": paths.wikipathways_raw_data_dir
        }

        self.processed_data_path = paths.processed_data_dir
        self.output_path = paths.output_dir
        self.kbs = dict()

    def process_raw_pathway_kbs(self):
        """
        Load all pathway raw data, process, and save processed data
        :return:
        """
        for kb_name, kb_path in self.path_kb_dirs.items():
            sys.stdout.write('\nLoading %s\n' % kb_name)
            sys.stdout.write('\t %s\n' % kb_path)
            kb = PathKB(kb_name, kb_path)
            kb.load(kb_path)
            output_file = os.path.join(self.processed_data_path, 'kb_{}.pickle'.format(kb_name))
            kb.dump_pickle(output_file)
        return

    @staticmethod
    def _add_uniprot_identifiers(map_dict):
        """
        Given a mapping dictionary generated from pathway resources, add identifiers
        extracted from UniProt (secondary accession ids)
        :param map_dict:
        :return:
        """
        sys.stdout.write("Adding UniProt identifiers...\n")

        all_uniprot = [k for k in map_dict if k.lower().startswith('uniprot')]

        for uniprot_id in tqdm.tqdm(all_uniprot, total=len(all_uniprot)):
            db, uid = uniprot_id.split(':')

            try:
                # query UniProt API
                r = requests.get('http://www.uniprot.org/uniprot/' + uid + '.xml')
                root = etree.fromstring(r.content)

                if root:
                    for s in root[0]:
                        if s.tag.endswith('accession'):
                            map_dict[uniprot_id].add(str(s.text))
                        else:
                            break
            except Exception:
                print("Broken: %s" % uniprot_id)
                continue

        return map_dict

    @staticmethod
    def _add_chebi_identifiers(map_dict):
        """
        Given a mapping dictionary generated from pathway resources, add identifiers
        extracted from ChEBI (secondary accessory, conjugate acid/base, tautomers)
        :param map_dict:
        :return:
        """
        sys.stdout.write("Adding ChEBI identifiers...\n")

        all_chebi = [k for k in map_dict if k.lower().startswith('chebi')]

        ch = ChEBI()

        for chebi_id in tqdm.tqdm(all_chebi, total=len(all_chebi)):
            db, uid = chebi_id.split(':')

            try:
                # query ChEBI API
                result = ch.getCompleteEntity(uid)

                if hasattr(result, 'SecondaryChEBIIds'):
                    secondaries = result.SecondaryChEBIIds
                    for s in secondaries:
                        map_dict[chebi_id].add('ChEBI:' + str(s).split(':')[-1])

                if hasattr(result, 'OntologyChildren'):
                    for ent in result.OntologyChildren:
                        chd_id = 'ChEBI:' + str(ent.chebiId).split(':')[-1]
                        if ent.type == 'is conjugate acid of':
                            map_dict[chebi_id].add(chd_id)
                        if ent.type == 'is conjugate base of':
                            map_dict[chebi_id].add(chd_id)
                        if ent.type == 'is tautomer of':
                            map_dict[chebi_id].add(chd_id)

                if hasattr(result, 'OntologyParents'):
                    for ent in result.OntologyParents:
                        par_id = 'ChEBI:' + str(ent.chebiId).split(':')[-1]
                        if ent.type == 'is conjugate acid of':
                            map_dict[chebi_id].add(par_id)
                        if ent.type == 'is conjugate base of':
                            map_dict[chebi_id].add(par_id)
                        if ent.type == 'is tautomer of':
                            map_dict[chebi_id].add(par_id)
            except Exception:
                print("Broken: %s" % chebi_id)
                continue

        return map_dict

    @staticmethod
    def _add_bridge_db_identifiers(map_dict):
        """
        Given a mapping dictionary generated from pathway resources, add identifiers
        extracted from BridgeDB
        :param map_dict:
        :return:
        """
        sys.stdout.write("Adding BridgeDB identifiers...\n")

        for uniq_id in tqdm.tqdm(map_dict, total=len(map_dict)):
            db, uid = uniq_id.split(':')

            if db in constants.BRIDGEDB_MAP:
                # list of other DBs to query from
                q_dbs = constants.BRIDGEDB_MAP[db]
                for q_db in q_dbs:
                    try:
                        r = requests.get(
                            'http://webservice.bridgedb.org/Human/xrefs/{}/{}?dataSource={}'.format(
                                constants.BRIDGEDB_KEYS[db],
                                uid,
                                constants.BRIDGEDB_KEYS[q_db]
                            )
                        )

                        result = r.text
                        if len(result) > 0:
                            add_ids = [line.split('\t')[0] for line in result.split('\n')[:-1]]
                            new_ids = ['{}:{}'.format(q_db, i) for i in add_ids]
                            for n_id in new_ids:
                                map_dict[uniq_id].add(n_id)
                    except Exception:
                        sys.stdout.write('Problem with %s\n' % uniq_id)
                        continue

        return map_dict

    def get_identifiers_from_kbs(self):
        """
        build an identifier dictionary of all identifier mappings
        :param identifiers:
        :return:
        """
        id_mapping_dict = defaultdict(set)

        for kb_name in constants.PATHWAY_KBS:
            sys.stdout.write('\n%s \n' % kb_name)
            kb_path = os.path.join(self.processed_data_path, 'kb_{}.pickle'.format(kb_name))
            if os.path.exists(kb_path):
                kb = PathKB(kb_name)
                kb = kb.load_pickle(kb_name, kb_path)

                for p in tqdm.tqdm(kb.pathways, total=len(kb.pathways)):
                    for ent in p.entities:
                        id_set = list(set(ent.xrefs))
                        for p, q in itertools.combinations(id_set, 2):
                            id_mapping_dict[p].add(q)
                            id_mapping_dict[q].add(p)

        id_mapping_dict = self._add_uniprot_identifiers(id_mapping_dict)
        id_mapping_dict = self._add_chebi_identifiers(id_mapping_dict)
        id_mapping_dict = self._add_bridge_db_identifiers(id_mapping_dict)

        sys.stdout.write('Merging similar entries in mapping dict\n')
        id_mapping_dict = pathway_utils.merge_similar(id_mapping_dict)

        # write mapping dict to file
        mapping_file = os.path.join(self.output_path, 'id_map_dict.json')
        with open(mapping_file, 'w') as outf:
            json.dump(id_mapping_dict, outf, indent=4, sort_keys=True)

        return


if __name__ == "__main__":
    # initialize loader
    path_kb_loader = PathwayKBLoader()

    # # process all raw kbs
    # path_kb_loader.process_raw_pathway_kbs()

    # load processed kbs and extract identifiers
    path_kb_loader.get_identifiers_from_kbs()
