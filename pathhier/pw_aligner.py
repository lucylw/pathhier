# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import jsonlines
import tqdm
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from pathhier.candidate_selector import CandidateSelector
from pathhier.paths import PathhierPaths
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.base_utils as base_utils
import pathhier.constants as constants

from allennlp.commands.train import train_model_from_file
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from pathhier.nn.pathway_dataset_reader import PathwayDatasetReader
from pathhier.nn.pathway_model import PWAlignNN
from pathhier.nn.pathway_predictor import PathwayPredictor


class PWAligner:
    def __init__(self, kb_path, pw_path):
        # get output directory for model and temp files
        paths = PathhierPaths()

        # create nn paths
        self.nn_config_file = os.path.join(paths.base_dir, 'config', 'model_name.json')
        self.nn_model_dir = os.path.join(paths.base_dir, 'model')
        self.train_data_path = os.path.join(paths.training_data_dir, 'pw_training.train')
        self.dev_data_path = os.path.join(paths.training_data_dir, 'pw_training.dev')

        # create final output directory
        self.output_dir = os.path.join(
            paths.output_dir,
            '{}-{}'.format('model',
                           datetime.now().strftime('%Y-%m-%d'))
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        assert os.path.exists(self.nn_config_file)
        assert os.path.exists(self.nn_model_dir)

        # load  KB from file
        assert os.path.exists(kb_path)
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)

        # load PW from file
        assert os.path.exists(pw_path)
        with open(pw_path, 'r') as f:
            self.pw = json.load(f)

        # initialize candidate selector
        self.cand_sel = CandidateSelector(self.kb, self.pw)

    @staticmethod
    def _read_tsv_file(d_file):
        """
        Read specified file and return content minus header
        :param d_file:
        :return:
        """
        data = []
        with open(d_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                data.append(row)
        return data

    def _form_training_entity(self, l, pw_id, kb_id):
        """
        Form training json entity
        :param l
        :param pw_id
        :param kb_id
        :return:
        """
        return pathway_utils.form_name_entries(l, pw_id, self.pw[pw_id], kb_id, self.kb[kb_id])

    def _apply_model_to_kb(self, predictor, batch_size=32):
        """
        Apply NN model to bootstrap KB
        :param iter_num: iteration number
        :return:
        """
        matches = []
        batch_json_data = []

        for kb_ent_id, kb_ent_values in tqdm.tqdm(self.kb.items()):
            for pw_ent_id in self.cand_sel.select(kb_ent_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                batch_json_data += self._form_training_entity(0, pw_ent_id, kb_ent_id)
                if len(batch_json_data) >= batch_size:
                    batch_use = batch_json_data[:batch_size]
                    results = predictor.predict_batch_json(batch_use)
                    for model_input, output in zip(batch_use, results):
                        matches.append((
                            model_input['kb_id'],
                            model_input['pw_id'],
                            output['score'][0],
                            output['predicted_label'][0]
                        ))
                    batch_json_data = batch_json_data[batch_size:]

        results = predictor.predict_batch_json(batch_json_data)
        for model_input, output in zip(batch_json_data, results):
            matches.append((
                model_input['kb_id'],
                model_input['pw_id'],
                output['score'][0],
                output['predicted_label'][0]
            ))
        return matches

    def _append_data_to_file(self, data, file_path):
        """
        Append data to training file
        :param data:
        :param file_path:
        :return:
        """
        with jsonlines.open(file_path, mode='a') as writer:
            for d in data:
                writer.write(d)
        return

    def _keep_new_predictions(self, predictions):
        """
        Retain only predictions which do not exist in the training data
        :param predictions:
        :return:
        """
        pos_predictions = [pred for pred in predictions if pred[3] == 1]
        neg_predictions = [pred for pred in predictions if pred[3] == 0]

        pos_predictions.sort(key=lambda x: x[2], reverse=True)
        neg_predictions.sort(key=lambda x: x[2])

        keep_top_n = int(constants.KEEP_TOP_N_PERCENT_MATCHES * len(predictions))
        keep_pos_pairs = pos_predictions[:min([keep_top_n, int(len(pos_predictions) / 2)])]
        keep_neg_pairs = neg_predictions[:min([keep_top_n, int(len(neg_predictions) / 2)])]

        id_pairs = set([(i[0], i[1], i[3]) for i in keep_pos_pairs + keep_neg_pairs])

        new_training_data = base_utils.flatten([
            self._form_training_entity(label, pw_id, kb_id) for kb_id, pw_id, label in id_pairs
        ])

        new_train, new_dev = pathway_utils.split_data(new_training_data, constants.DEV_DATA_PORTION)

        self._append_data_to_file(new_train, self.train_data_path)
        self._append_data_to_file(new_dev, self.dev_data_path)

        print('Appended %i instances to training data.' % len(new_train))
        print('Appended %i instances to development data.' % len(new_dev))
        return

    def _train_nn(self, iter: int) -> str:
        """
        Train NN
        :param iter:
        :return: Path to NN model
        """
        model_dir = os.path.join(self.nn_model_dir, 'nn_model_iter' + str(iter))
        assert not(os.path.exists(model_dir))
        train_model_from_file(self.nn_config_file, model_dir)
        model_path = os.path.join(model_dir, "model.tar.gz")
        return model_path

    def train_model(self, total_iter: int, batch_size=32, cuda_device=-1):
        """
        Train model
        :param total_iter: total bootstrapping iterations
        :param cuda_device
        :return:
        """
        for i in range(0, total_iter):
            sys.stdout.write('\n\n')
            sys.stdout.write('--------------\n')
            sys.stdout.write('Iteration: %i\n' % i)
            sys.stdout.write('--------------\n')

            # train nn model
            model_file = self._train_nn(i)

            # load model as predictor from archive file
            archive = load_archive(model_file, cuda_device=cuda_device)
            predictor = Predictor.from_archive(archive, 'pw_aligner')

            # apply predictor to kb of interest
            matches = self._apply_model_to_kb(predictor, batch_size)

            # keep portion of matches with high confidence
            self._keep_new_predictions(matches)

        return

    def run_model(self, model_file, batch_size=32, cuda_device=-1):
        """
        Apply model to input data
        :return:
        """

        # match entities between bootstrap KB and PW
        sys.stdout.write("\tApplying model to KB...\n")

        # load model as predictor from archive file
        archive = load_archive(model_file, cuda_device=cuda_device)
        predictor = Predictor.from_archive(archive, 'pw_aligner')

        # apply predictor to kb of interest
        matches = self._apply_model_to_kb(predictor, batch_size)
        matches = [(kb_id, pw_id, score) for kb_id, pw_id, score, label in matches if label == 1.]

        # get output data
        output_data = []
        for kb_id, pw_id, score in matches:
            kb_ent = self.kb[kb_id]
            pw_ent = self.pw[pw_id]
            output_data.append((
                score,
                kb_id,
                kb_ent['name'],
                kb_ent['definition'],
                pw_id,
                pw_ent['name'],
                pw_ent['definition']
            ))

        # write to output file
        output_file = os.path.join(self.output_dir, 'alignments.tsv')
        with open(output_file, 'w') as outf:
            outf.write('Score\tPathway_id\tPathway_name\tPathway_def\tPW_id\tPW_name\tPW_def\n')
            for data_line in output_data:
                outf.write('\t'.join(data_line))
                outf.write('\n')

        print('Matches saved to %s' % output_file)
        print('done.')


