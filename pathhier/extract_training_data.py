import os
import csv
import json
import jsonlines
import pickle
import random

from typing import List

from pathhier.paths import PathhierPaths
from pathhier.candidate_selector import CandidateSelector
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.constants as constants


# class for extracting training data out of PW
class TrainingDataExtractor:
    def __init__(self, num_neg=1):
        self.paths = PathhierPaths()
        self.num_neg = num_neg
        self.pw = self._load_pw()
        self.kbs, self.cand_sel = self._load_kbs()
        self.kb_path_names = self._load_kb_pathway_names()

    def _kb_pickle_to_json(self, kb: List):
        """
        Convert PathKB to json
        :param kb:
        :return:
        """
        kb_dict = dict()

        for p in kb:
            rels = p.relations
            subclass_of = [obj.uid for prop, obj in rels if prop == 'subClassOf']
            part_of = [obj.uid for prop, obj in rels if prop == 'part_of']
            kb_dict[p.uid] = {
                'name': p.name,
                'aliases': p.aliases,
                'synonyms': [],
                'definition': p.definition,
                'subClassOf': subclass_of,
                'part_of': part_of,
                'instances': [p.uid]
            }

        return kb_dict

    def _load_pw(self):
        """
        Load PW from disk
        :return:
        """
        # load pathway ontology
        print('Loading pathway ontology...')
        pw_file = os.path.join(self.paths.pathway_ontology_dir, 'pw.json')
        assert os.path.exists(pw_file)

        with open(pw_file, 'r') as f:
            pw = json.load(f)

        return pw

    def _load_kbs(self):
        """
        Load KBs from disk
        :return:
        """
        print('Loading pathway databases...')
        kbs = dict()
        cand_sel = []

        kbs_to_load = ['kegg', 'smpdb', 'pid']

        for kb_name in kbs_to_load:
            print('\tLoading %s' % kb_name)

            pkl_file = os.path.join(self.paths.processed_data_dir, 'kb_{}.pickle'.format(kb_name))
            assert os.path.exists(pkl_file)

            kb = pickle.load(open(pkl_file, 'rb'))
            kbs[kb_name] = self._kb_pickle_to_json(kb)

            cand_sel.append(CandidateSelector(self.pw, kbs[kb_name]))

        return kbs, cand_sel

    def _parse_kegg_paths(self, path):
        """
        Parse all KEGG pathways from file
        :param path:
        :return:
        """
        pathways = []

        with open(path, 'r') as f:
            contents = f.readlines()

        uid = None

        for line in contents:
            line = line.strip()
            if line:
                if '.' in line:
                    pass
                else:
                    if line.isdigit():
                        uid = 'KEGG:' + line
                    else:
                        pathways.append((uid, line, ""))
                        uid = None

        return pathways

    def _parse_smpdb_paths(self, path):
        """
        Parse all SMPDB pathways from file
        :param path:
        :return:
        """
        pathways = []

        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for uid, name, _, definition in reader:
                pathways.append(
                    ('SMP:' + uid[3:], name, definition.replace('\n', ' '))
                )

        return pathways

    def _parse_pid_paths(self, path):
        """
        Parse all PID pathways from file
        :param path:
        :return:
        """
        pathways = dict()

        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for id1, id2, typ, name1, name2 in reader:
                uid1 = '{}:{}'.format('PID', id1)
                uid2 = '{}:{}'.format('PID', id2)
                if uid1 not in pathways:
                    pathways[uid1] = name1
                if uid2 not in pathways:
                    pathways[uid2] = name2

        return [(k, v, "") for k, v in pathways.items()]

    def _load_kb_pathway_names(self):
        """
        Load pathway name information from file
        :return:
        """
        print('Loading pathway names...')

        file_names = {'kegg': ('kegg_paths', self._parse_kegg_paths),
                      'smpdb': ('smpdb_paths', self._parse_smpdb_paths),
                      'pid': ('pid_paths.tsv', self._parse_pid_paths)}

        kb_path_names = dict()

        for kb_name, path_info in file_names.items():
            file_name, parse_function = path_info
            file_path = os.path.join(self.paths.raw_data_dir, file_name)
            kb_path_names[kb_name] = parse_function(file_path)

        xref_dict = dict()

        for kb_name, kb_pathways in kb_path_names.items():
            for p_id, p_name, p_def in kb_pathways:
                xref_dict[p_id] = (p_name, p_def)

        return xref_dict

    def _extract_positive_mappings(self):
        """
        Extract synonym mappings from PW
        :return:
        """
        positives = []
        positive_defs = []

        # iterate through PW and extract xrefs
        for pw_id, pw_value in self.pw.items():
            xrefs = pw_value['synonyms']

            for xref in xrefs:
                xref_db, xref_id = xref.split(':')

                if 'KEGG' in xref_db:
                    kb_id = 'KEGG:' + xref_id
                elif 'SMP' in xref_db:
                    kb_id = 'SMP:' + xref_id
                elif 'PID' in xref_db:
                    kb_id = xref
                else:
                    kb_id = ''

                if kb_id and kb_id in self.kb_path_names:
                    positives += pathway_utils.form_name_entries_special(
                        1, pw_id, pw_value, kb_id, self.kb_path_names[kb_id]
                    )
                    positive_defs += pathway_utils.form_definition_entries_special(
                        1, pw_id, pw_value, kb_id, self.kb_path_names[kb_id]
                    )

        return positives, positive_defs

    def _extract_negative_mappings(self, pos):
        """
        Extract negative mappings for training
        :param pos:
        :return:
        """

        def _clean_id(id):
            if 'kegg' in id:
                name, num = id.split(':')
                return 'KEGG:{}'.format(num[3:])
            elif 'smp' in id or 'SMP' in id:
                return 'SMP:{}'.format(id[3:])
            elif 'pid' in id:
                return id
            else:
                return None

        negatives = []
        negative_defs = []

        done_pairs = set([(entry['pw_id'], entry['kb_id']) for entry in pos])

        # iterate through PW and extract xrefs
        for pw_id, pw_value in self.pw.items():
            neg_sample = []

            # hard negatives
            for cs in self.cand_sel:
                neg_sample += cs.select(pw_id)[3:2+self.num_neg]
            # easy negatives
            for kb_name, kb in self.kbs.items():
                neg_sample += random.sample(kb.keys(), self.num_neg)

            # process kb_ids
            neg_sample = [_clean_id(i) for i in neg_sample]

            # add entry for each negative
            for neg in neg_sample:
                if (pw_id, neg) not in done_pairs:
                    if 'pid' in neg:
                        negatives += pathway_utils.form_name_entries(
                            0, pw_id, pw_value, neg, self.kbs['pid'][neg]
                        )
                        negative_defs += pathway_utils.form_definition_entries(
                            0, pw_id, pw_value, neg, self.kbs['pid'][neg]
                        )
                    elif neg in self.kb_path_names:
                        negatives += pathway_utils.form_name_entries_special(
                            0, pw_id, pw_value, neg, self.kb_path_names[neg]
                        )
                        negative_defs += pathway_utils.form_definition_entries_special(
                            0, pw_id, pw_value, neg, self.kb_path_names[neg]
                        )
                    else:
                        continue
                    done_pairs.add((pw_id, neg))

        return negatives, negative_defs

    def _save_one_to_file(self, data, file_path):
        """
        Save one dataset to file
        :param data:
        :param file_path:
        :return:
        """
        with jsonlines.open(file_path, mode='w') as writer:
            for d in data:
                writer.write(d)
        return

    def _save_to_file(self, train, dev, data_type=''):
        """
        Save data to file
        :param train:
        :param dev:
        :return:
        """
        file_name_header = 'pw_training'

        if data_type:
            file_name_header += '.' + data_type

        train_file_name = file_name_header + '.train'
        dev_file_name = file_name_header + '.dev'

        train_data_path = os.path.join(self.paths.training_data_dir, train_file_name)
        dev_data_path = os.path.join(self.paths.training_data_dir, dev_file_name)

        self._save_one_to_file(train, train_data_path)
        self._save_one_to_file(dev, dev_data_path)

        return

    def _extract_mesh_go_mappings(self):
        """
        Extract MeSH GO mappings from file
        :return:
        """

        mapping_file = os.path.join(self.paths.training_data_dir, 'mesh_go_mappings')

        mesh_go_mappings = []
        with jsonlines.open(mapping_file, 'r') as f:
            for l in f:
                mesh_go_mappings.append(l)

        name_training = []
        def_training = []

        for entry in mesh_go_mappings:
            name_training += pathway_utils.form_name_entries(
                entry['label'], entry['mesh_id'], entry['mesh_ent'], entry['go_id'], entry['go_ent']
            )
            def_training += pathway_utils.form_definition_entries(
                entry['label'], entry['mesh_id'], entry['mesh_ent'], entry['go_id'], entry['go_ent']
            )

        return name_training, def_training

    def extract_training_data(self):
        """
        Extract training data
        :return:
        """
        positives, positive_defs = self._extract_positive_mappings()
        negatives, negative_defs = self._extract_negative_mappings(positives)
        umls_names, umls_defs = self._extract_mesh_go_mappings()

        # save names to training files
        print('Saving name data to file...')
        train, dev = pathway_utils.split_data(
            positives + negatives + umls_names, constants.DEV_DATA_PORTION
        )
        self._save_to_file(train, dev)

        # save definition to training files
        print('Saving definition data to file...')
        train_def, dev_def = pathway_utils.split_data(
            positive_defs + negative_defs + umls_defs, constants.DEV_DATA_PORTION
        )
        self._save_to_file(train_def, dev_def, 'def')

        return

if __name__ == '__main__':
    extractor = TrainingDataExtractor()
    extractor.extract_training_data()