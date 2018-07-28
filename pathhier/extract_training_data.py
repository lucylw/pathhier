import os
import sys
import csv
import json
import jsonlines
import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split

from typing import List

from pathhier.paths import PathhierPaths
from pathhier.candidate_selector import CandidateSelector


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

    def _form_pw_string_entry(self, pw_id, p_entry, pw):
        """
        Form a string representation of the pathway entry
        :param p_entry:
        :return:
        """
        # superclasses = ['subClassOf: {}'.format(pw[parent_id]['name'])
        #                 for parent_id in p_entry['subClassOf'] if parent_id in pw]
        # part_supers = ['part_of: {}'.format(pw[parent_id]['name'])
        #                for parent_id in p_entry['part_of'] if parent_id in pw]

        p_string = '; '.join(set(p_entry['aliases']))
        if p_entry['definition']:
            p_string += '; ' + '; '.join(p_entry['definition'])
        # if superclasses:
        #     p_string += '; ' + '; '.join(superclasses)
        # if part_supers:
        #     p_string += '; ' + '; '.join(part_supers)
        # p_string += '; '

        return pw_id, p_string

    def _form_kb_string_entry(self, kb_id, kb_entry):
        """
        Form a string representation of the kb entry
        :param kb_entry:
        :return:
        """
        kb_string = kb_entry[0] + '; ' + kb_entry[1]
        return kb_id, kb_string

    def _form_pid_string_entry(self, kb_id, kb_entry, kb):
        """
        Form a string representation of the PID kb entry
        :param kb_id:
        :param kb_entry:
        :return:
        """
        # superclasses = ['subClassOf: {}'.format(kb[parent_id]['name'])
        #                 for parent_id in kb_entry['subClassOf'] if parent_id in kb]
        # part_supers = ['part_of: {}'.format(kb[parent_id]['name'])
        #                for parent_id in kb_entry['part_of'] if parent_id in kb]

        kb_string = '; '.join(set(kb_entry['aliases']))
        if kb_entry['definition']:
            kb_string += '; ' + '; '.join(kb_entry['definition'])
        # if superclasses:
        #     kb_string += '; ' + '; '.join(superclasses)
        # if part_supers:
        #     kb_string += '; ' + '; '.join(part_supers)
        # kb_string += '; '

        return kb_id, kb_string

    def _extract_positive_mappings(self):
        """
        Extract synonym mappings from PW
        :return:
        """
        positives = []

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
                    entry = {'label': 1,
                             'pw_cls': self._form_pw_string_entry(pw_id, pw_value, self.pw),
                             'pathway': self._form_kb_string_entry(kb_id, self.kb_path_names[kb_id])}
                    positives.append(entry)

        return positives

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
        pos_pairs = [(entry['pw_cls'][0], entry['pathway'][0]) for entry in pos]
        neg_pairs = []

        # iterate through PW and extract xrefs
        for pw_id, pw_value in self.pw.items():
            neg_sample = []

            # hard negatives
            for cs in self.cand_sel:
                neg_sample += cs.select(pw_id)[2:2+self.num_neg]
            # easy negatives
            for kb_name, kb in self.kbs.items():
                neg_sample += random.sample(kb.keys(), self.num_neg)

            # process kb_ids
            neg_sample = [_clean_id(i) for i in neg_sample]

            # add entry for each negative
            for neg in neg_sample:
                if (pw_id, neg) not in pos_pairs and (pw_id, neg) not in neg_pairs:
                    if 'pid' in neg:
                        entry = {'label': 0,
                                 'pw_cls': self._form_pw_string_entry(pw_id, pw_value, self.pw),
                                 'pathway': self._form_pid_string_entry(neg, self.kbs['pid'][neg], self.kbs['pid'])}
                        negatives.append(entry)
                        neg_pairs.append((pw_id, neg))
                    else:
                        if neg in self.kb_path_names:
                            entry = {'label': 0,
                                     'pw_cls': self._form_pw_string_entry(pw_id, pw_value, self.pw),
                                     'pathway': self._form_kb_string_entry(neg, self.kb_path_names[neg])}
                            negatives.append(entry)
                            neg_pairs.append((pw_id, neg))

        return negatives

    def _split_training_data(self, data):
        """
        Split data stratified into train, dev, test (60/20/20)
        :param data:
        :return:
        """
        labels = [(i, d['label']) for i, d in enumerate(data)]
        inds = np.array([l[0] for l in labels])
        labs = np.array([l[1] for l in labels])

        ind_train, ind_test, lab_train, lab_test = train_test_split(inds, labs,
                                                            stratify=labs,
                                                            test_size=0.20)

        ind_train, ind_dev, lab_train, lab_dev = train_test_split(ind_train, lab_train,
                                                                  stratify=lab_train,
                                                                  test_size=0.25)

        train_data = [data[i] for i in ind_train]
        dev_data = [data[i] for i in ind_dev]
        test_data = [data[i] for i in ind_test]

        return train_data, dev_data, test_data

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

    def _save_to_file(self, train, dev, test):
        """
        Save data to file
        :param train:
        :param dev:
        :param test:
        :return:
        """
        print('Saving data to file...')
        train_data_path = os.path.join(self.paths.training_data_dir, 'pw_training.train')
        dev_data_path = os.path.join(self.paths.training_data_dir, 'pw_training.dev')
        test_data_path = os.path.join(self.paths.training_data_dir, 'pw_training.test')

        self._save_one_to_file(train, train_data_path)
        self._save_one_to_file(dev, dev_data_path)
        self._save_one_to_file(test, test_data_path)

        return

    def extract_training_data(self):
        """
        Extract training data
        :return:
        """
        positives = self._extract_positive_mappings()
        negatives = self._extract_negative_mappings(positives)
        train, dev, test = self._split_training_data(positives + negatives)
        self._save_to_file(train, dev, test)
        return train, dev, test


if __name__ == '__main__':
    extractor = TrainingDataExtractor()
    train, dev, test = extractor.extract_training_data()
    print('done.')