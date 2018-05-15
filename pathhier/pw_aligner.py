# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split

from pathhier.matcher_model import PWMatcher
from pathhier.candidate_selector import CandidateSelector
from pathhier.paths import PathhierPaths
import pathhier.constants as constants


class PWAligner:
    def __init__(self, orig_data_file, kb_path, pw_path):
        # get output directory for model and temp files
        paths = PathhierPaths()
        self.output_dir = os.path.join(
            paths.output_dir,
            '{}-{}'.format('model',
                           datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        )
        os.makedirs(self.output_dir)

        # set live training data file
        assert os.path.exists(orig_data_file)
        self.live_data_file = orig_data_file

        # load bootstrap KB from file
        assert os.path.exists(kb_path)
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)

        # load PW from file
        assert os.path.exists(pw_path)
        with open(pw_path, 'r') as f:
            self.pw = json.load(f)

        # compute vocab over training data and bootstrap KB
        self.vocab = self._compute_vocab()

        # initialize model
        self.model = PWMatcher(self.vocab)

    def _compute_vocab(self):
        """
        Compute vocab of all relevant KBs and training data
        :return:
        """
        return dict()

    @staticmethod
    def _form_training_entity(l, values):
        """
        Form training json entity
        :param l: label
        :param values: provenance, as well as id, name, and definition of source entity and PW entity
        :return:
        """
        provenance, pw_id, pw_name, pw_def, kb_id, kb_name, kb_def = values
        return {
            "label": int(l),
            "provenance": provenance,
            "kb_ent": {
                "id": kb_id,
                "name": kb_name,
                "definition": kb_def
            },
            "pw_ent": {
                "id": pw_id,
                "name": pw_name,
                "definition": pw_def
            }
        }

    def _process_data(self, d_file):
        """
        Read training data and split into training/development set
        :param d_file:
        :return:
        """

        # read training data from file
        data = []
        with open(d_file) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                data.append(row)

        # split into labels and values
        labels = [d[0] for d in data]
        values = [d[1:] for d in data]

        # split into training and development data sets
        train_values, dev_values, \
        train_labels, dev_labels = train_test_split(values,
                                                    labels,
                                                    stratify=labels,
                                                    test_size=0.20)

        # create json objects with training data
        train_data = []
        dev_data = []

        for t_label, t_values in zip(train_labels, train_values):
            train_data.append(self._form_training_entity(t_label, t_values))

        for d_label, d_values in zip(dev_labels, dev_values):
            dev_data.append(self._form_training_entity(d_label, d_values))

        return train_data, dev_data

    def _write_data_to_file(self, data, output_file) -> None:
        """
        Write data to specified output file
        :param data:
        :param output_file:
        :return:
        """
        with open(output_file, 'w') as outf:
            outf.write('Match\tProvenance\tPW_id\tPW_name\tPW_def\txref_id\txref_name\txref_def\n')
            for training_line in data:
                outf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    training_line['label'],
                    training_line['provenance'],
                    training_line['pw_ent']['id'],
                    training_line['pw_ent']['name'],
                    training_line['pw_ent']['definition'],
                    training_line['kb_ent']['id'],
                    training_line['kb_ent']['name'],
                    training_line['kb_ent']['definition']
                ))

    def _apply_model_to_kb(self, iter_num):
        """
        Apply model to bootstrap KB
        :param iter_num: iteration number
        :return:
        """
        cand_sel = CandidateSelector(self.kb, self.pw)
        test_data = []
        provenance = 'bootstrap_iter{}'.format(iter_num)

        for kb_ent_id, kb_ent_values in self.kb.items():
            for pw_ent_id in cand_sel.select(kb_ent_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                pw_ent = self.pw[pw_ent_id]
                t_values = (provenance,
                            pw_ent_id, pw_ent['name'], pw_ent['definition'],
                            kb_ent_id, kb_ent_values['name'], kb_ent_values['definition'])
                test_data.append(self._form_training_entity('-1', t_values))

        # compute similarity scores using model
        predicted_scores = self.model.test(test_data)

        # zip together with data and sort by similarity score
        predictions = zip(predicted_scores, test_data)
        predictions = [pred for pred in predictions if pred[0] >= constants.SIMSCORE_THRESHOLD]
        predictions.sort(key=lambda x: x[0], reverse=True)

        # determine what to keep in bootstrap iteration
        keep_top_n = int(constants.KEEP_TOP_N_PERCENT_MATCHES * len(predictions))
        to_add = [data_entry for score, data_entry in predictions[:keep_top_n+1]]

        for entry in to_add:
            entry.update(('label', '1'))

        return to_add

    def train_model(self, total_iter: int):
        """
        Train model
        :param total_iter: total bootstrapping iterations
        :return:
        """
        for i in range(0, total_iter):
            sys.stdout.write('Iteration: %i\n' % i)

            # specify output files
            train_output_file = os.path.join(self.output_dir, 'training_data.tsv.{}'.format(i))
            dev_output_file = os.path.join(self.output_dir, 'development_data.tsv.{}'.format(i))
            model_file = os.path.join(self.output_dir, 'model.pickle.{}'.format(i))
            all_data_file = os.path.join(self.output_dir, 'all_data.tsv.{}'.format(i+1))

            # split training data into training and development set
            train_data, dev_data = self._process_data(self.live_data_file)

            # write data to file
            self._write_data_to_file(train_data, train_output_file)
            self._write_data_to_file(dev_data, dev_output_file)

            # train model on training data
            self.model.train(train_data, dev_data)

            # save model to file
            pickle.dump(self.model, open(model_file, 'wb'))

            # match entities between bootstrap KB and PW
            bootstrapped_positives = self._apply_model_to_kb(i)

            # append to previous round data
            new_data = train_data + dev_data + bootstrapped_positives

            # write all new data to file
            self._write_data_to_file(new_data, all_data_file)
            self.live_data_file = all_data_file


if __name__ == '__main__':
    data_file = sys.argv[1]
    kb_file = sys.argv[2]
    pw_file = sys.argv[3]
    total_iter = int(sys.argv[4])
    aligner = PWAligner(data_file, kb_file, pw_file)
    aligner.train_model(total_iter)