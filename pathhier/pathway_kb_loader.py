import os
import sys
import json
import tqdm
import itertools
import time
from lxml import etree
from collections import defaultdict
from bioservices import *

from pathhier.paths import PathhierPaths
from pathhier.pathway import PathKB
from pathhier.ontology import Ontology
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.base_utils as base_utils
import pathhier.constants as constants


# class for loading all pathways and processing everything in succession
class PathwayKBLoader:
    def __init__(self) -> None:
        paths = PathhierPaths()

        self.path_kb_dirs = {
            "humancyc": paths.humancyc_raw_data_dir,
            "kegg": paths.kegg_raw_data_dir,
            "panther": paths.panther_raw_data_dir,
            "pid": paths.pid_raw_data_dir,
            "reactome": paths.reactome_raw_data_dir,
            "smpdb": paths.smpdb_raw_data_dir,
            "wikipathways": paths.wikipathways_raw_data_dir
        }

        self.processed_data_path = paths.processed_data_dir
        self.output_path = paths.output_dir
        self.mapping_file = os.path.join(self.output_path, 'id_map_dict.json')
        self.pathway_ontology_file = paths.pathway_ontology_file
        self.pw_json_file = os.path.join(self.output_path, 'pw.json')

        self.kbs = []
        self.forward_map = dict()
        self.backward_map = dict()
        self.pathway_ontology = dict()

    def process_raw_pathway_kbs(self) -> None:
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

    def _get_identifiers_from_kbs(self) -> dict:
        """
        Get mapping dict from KBs
        :return:
        """
        id_mapping_dict = defaultdict(set)

        for kb in self.kbs:
            sys.stdout.write('\n%s \n' % kb.name)
            for p in tqdm.tqdm(kb.pathways, total=len(kb.pathways)):
                for ent in p.entities:
                    id_set = list(set(ent.xrefs))
                    if len(id_set) == 1:
                        id_mapping_dict[id_set.pop()] = set([])
                    for p, q in itertools.combinations(id_set, 2):
                        id_mapping_dict[p].add(q)
                        id_mapping_dict[q].add(p)

        return id_mapping_dict

    @staticmethod
    def _add_uniprot_identifiers(map_dict) -> dict:
        """
        Given a mapping dictionary generated from pathway resources, add identifiers
        extracted from UniProt (secondary accession ids)
        :param map_dict:
        :return:
        """
        sys.stdout.write("Adding UniProt identifiers...\n")
        r_session = base_utils.requests_retry_session()
        all_uniprot = [k for k in map_dict if k.lower().startswith('uniprot')]

        for uniprot_id in tqdm.tqdm(all_uniprot, total=len(all_uniprot)):
            db, uid = uniprot_id.split(':')

            try:
                # query UniProt API
                r = r_session.get(
                    'http://www.uniprot.org/uniprot/' + uid + '.xml'
                )
            except Exception as x:
                print("%s: %s" % (uniprot_id, x.__class__.__name__))
                continue

            if r.content:
                root = etree.fromstring(r.content)
                if root:
                    for s in root[0]:
                        if s.tag.endswith('accession'):
                            new_id = '{}:{}'.format('UniProt', s.text.split(':')[-1])
                            map_dict[uniprot_id].add(new_id)
                        else:
                            break

        return map_dict

    @staticmethod
    def _add_chebi_identifiers(map_dict) -> dict:
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
            except Exception as x:
                print("%s: %s" % (chebi_id, x.__class__.__name__))
                continue

            to_add = []

            if hasattr(result, 'SecondaryChEBIIds'):
                to_add += [str(s) for s in result.SecondaryChEBIIds]

            if hasattr(result, 'OntologyChildren'):
                to_add += [str(ent.chebiId) for ent in result.OntologyChildren
                           if ent.type in ('is conjugate acid of',
                                           'is conjugate base of',
                                           'is tautomer of')]

            if hasattr(result, 'OntologyParents'):
                to_add += [str(ent.chebiId) for ent in result.OntologyParents
                           if ent.type in ('is conjugate acid of',
                                           'is conjugate base of',
                                           'is tautomer of')]

            for ent_id in to_add:
                new_id = '{}:{}'.format('ChEBI', ent_id.split(':')[-1])
                map_dict[chebi_id].add(new_id)

        return map_dict

    @staticmethod
    def _add_bridge_db_identifiers(map_dict) -> dict:
        """
        Given a mapping dictionary generated from pathway resources, add identifiers
        extracted from BridgeDB
        :param map_dict:
        :return:
        """
        sys.stdout.write("Adding BridgeDB identifiers...\n")
        r_session = base_utils.requests_retry_session()

        for uniq_id in tqdm.tqdm(map_dict, total=len(map_dict)):
            db, uid = uniq_id.split(':')

            if db in constants.BRIDGEDB_MAP:
                # list of other DBs to query from
                q_dbs = constants.BRIDGEDB_MAP[db]
                for q_db in q_dbs:
                    try:
                        r = r_session.get(
                            'http://webservice.bridgedb.org/Human/xrefs/{}/{}?dataSource={}'.format(
                                constants.BRIDGEDB_KEYS[db],
                                uid,
                                constants.BRIDGEDB_KEYS[q_db]
                            )
                        )
                    except Exception as x:
                        print("%s: %s" % (uniq_id, x.__class__.__name__))
                        continue

                    result = r.text
                    if len(result) > 0:
                        add_ids = [line.split('\t')[0] for line in result.split('\n')[:-1]]
                        new_ids = ['{}:{}'.format(q_db, i) for i in add_ids if i.isalnum()]
                        for n_id in new_ids:
                            new_id = '{}:{}'.format(q_db, n_id)
                            map_dict[uniq_id].add(new_id)

                    time.sleep(0.5)

        return map_dict

    @staticmethod
    def _generate_local_identifiers(map_dict) -> (defaultdict, dict):
        """
        Create local identifiers
        :param map_dict:
        :return:
        """
        forward = defaultdict(set)
        backward = dict()
        next_id = 1

        for k, v in map_dict.items():
            group = {k}
            group.update(v)
            if any(group) in backward:
                shared_keys = group & backward.keys()
                local_id = backward[shared_keys.pop()]
            else:
                local_id = next_id
                next_id += 1

            # assign values in mapping dicts
            forward[local_id].update(group)
            for uid in group:
                backward[uid] = local_id

        return forward, backward

    def get_identifier_map(self) -> None:
        """
        build an identifier dictionary of all identifier mappings
        :param identifiers:
        :return:
        """
        id_mapping_dict = self._get_identifiers_from_kbs()
        id_mapping_dict = self._add_uniprot_identifiers(id_mapping_dict)
        id_mapping_dict = self._add_chebi_identifiers(id_mapping_dict)
        id_mapping_dict = self._add_bridge_db_identifiers(id_mapping_dict)
        id_mapping_dict = pathway_utils.merge_similar(id_mapping_dict)

        self.forward_map, self.backward_map = self._generate_local_identifiers(id_mapping_dict)
        self.save_id_dict()

    def merge_entities_on_identifiers(self) -> None:
        """
        Merge entities using local identifiers
        :return:
        """
        next_local_id = max(list(self.forward_map.keys())) + 1
        backward_keys = set(self.backward_map.keys())
        for kb in self.kbs:
            for p in kb.pathways:
                for e in p.entities:
                    if e.xrefs:
                        xref_overlap = set(e.xrefs) & backward_keys
                        if xref_overlap:
                            local_id = self.backward_map[xref_overlap.pop()]
                            e.lid = local_id
                        elif len(e.xrefs) == 1:
                            self.forward_map[next_local_id] = [e.xrefs[0]]
                            self.backward_map[e.xrefs[0]] = next_local_id
                            e.lid = next_local_id
                            next_local_id += 1
                        else:
                            print(e.xrefs)
                            raise UnboundLocalError("Unknown identifiers")

            kb.dump_pickle(kb.loc)
        self.save_id_dict()

    def process_pathway_ontology(self) -> None:
        """
        Process pathway ontology file and save as json file
        :return:
        """
        # Load pathway ontology from file
        pw = Ontology(name="PW",
                      filename=self.pathway_ontology_file)
        pw.load_from_file()

        pw_dict = dict()

        for cl in pw.owl_classes:
            pw_dict[cl] = {
                'name': pw.get_label(cl),
                'aliases': pw.get_all_labels(cl),
                'synonyms': pw.get_synonyms(cl),
                'definitions': pw.get_definition(cl),
                'subClassOf': pw.get_subClassOf(cl),
                'part_of': pw.get_part_of(cl)
            }

        with open(self.pw_json_file, 'w') as outf:
            json.dump(pw_dict, outf, indent=4, sort_keys=True)

    def load_kbs(self) -> None:
        """
        Load all kbs from file
        :return:
        """
        sys.stdout.write("Loading KBs...\n")

        for kb_name in constants.PATHWAY_KBS:
            sys.stdout.write('\n%s \n' % kb_name)
            kb_path = os.path.join(self.processed_data_path, 'kb_{}.pickle'.format(kb_name))
            if os.path.exists(kb_path):
                kb = PathKB(kb_name)
                kb = kb.load_pickle(kb_name, kb_path)
                self.kbs.append(kb)

    def load_id_dict(self) -> None:
        """
        Load identifier mapping dictionary from file
        :return:
        """
        sys.stdout.write("Loading identifier dictionaries...\n")
        assert os.path.exists(self.mapping_file)
        with open(self.mapping_file, 'r') as f:
            self.forward_map, self.backward_map = json.load(f)
        self.forward_map = {int(k): v for k, v in self.forward_map.items()}

    def save_id_dict(self) -> None:
        """
        Dump identifier mapping dictionaries to file
        :return:
        """
        # convert sets to lists
        forward_map = {k: list(v) for k, v in self.forward_map.items()}
        # write to file
        with open(self.mapping_file, 'w') as outf:
            json.dump(
                [forward_map, self.backward_map],
                outf, indent=4, sort_keys=True
            )

    def load_pw(self) -> None:
        """
        Load PW from json file
        :return:
        """
        sys.stdout.write("Loading pathway ontology from json...\n")
        assert os.path.exists(self.pw_json_file)
        with open(self.pw_json_file, 'r') as f:
            self.pathway_ontology = json.load(f)
