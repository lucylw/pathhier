# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import jsonlines
import tqdm
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from pathhier.candidate_selector import CandidateSelector
from pathhier.feature_generator import FeatureGenerator
from pathhier.paths import PathhierPaths
from pathhier.extract_training_data import TrainingDataExtractor
import pathhier.utils.pathway_utils as pathway_utils
import pathhier.utils.base_utils as base_utils
import pathhier.constants as constants

from allennlp.commands.train import train_model_from_file
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.commands.evaluate import evaluate as evaluate_allennlp

from pathhier.nn.pathway_dataset_reader import PathwayDatasetReader
from pathhier.nn.pathway_model import PWAlignNN
from pathhier.nn.pathway_predictor import PathwayPredictor


class PWAligner:
    def __init__(self, kb_path, pw_path):
        # get output directory for model and temp files
        paths = PathhierPaths()

        # create nn paths
        self.nn_name_config_file = os.path.join(paths.base_dir, 'config', 'model_name.json')
        self.nn_def_config_file = os.path.join(paths.base_dir, 'config', 'model_def.json')
        self.nn_model_dir = os.path.join(paths.base_dir, 'model')

        self.name_train_data_path = os.path.join(paths.training_data_dir, 'pw_training.train')
        self.name_dev_data_path = os.path.join(paths.training_data_dir, 'pw_training.dev')
        self.name_test_data_path = os.path.join(paths.training_data_dir, 'pw_training.test')

        self.def_train_data_path = os.path.join(paths.training_data_dir, 'pw_training.def.train')
        self.def_dev_data_path = os.path.join(paths.training_data_dir, 'pw_training.def.dev')
        self.def_test_data_path = os.path.join(paths.training_data_dir, 'pw_training.def.test')

        assert os.path.exists(self.nn_name_config_file)
        assert os.path.exists(self.nn_def_config_file)
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

    def _apply_model_to_kb(self, predictor, form_entity_function, batch_size=32):
        """
        Apply NN model to bootstrap KB
        :param iter_num: iteration number
        :return:
        """
        matches = []
        batch_json_data = []

        for kb_ent_id, kb_ent_values in tqdm.tqdm(self.kb.items()):
            for pw_ent_id in self.cand_sel.select(kb_ent_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                batch_json_data += form_entity_function(
                    0, pw_ent_id, self.pw[pw_ent_id], kb_ent_id, self.kb[kb_ent_id]
                )
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

        if batch_json_data:
            results = predictor.predict_batch_json(batch_json_data)
            for model_input, output in zip(batch_json_data, results):
                matches.append((
                    model_input['kb_id'],
                    model_input['pw_id'],
                    output['score'][0],
                    output['predicted_label'][0]
                ))
        return matches

    def _apply_lr_to_kb(self, model, form_entity_function):
        """
        Apply LR model to KB
        :param model:
        :param form_entity_function:
        :return:
        """
        batch_json_data = []

        for kb_ent_id, kb_ent_values in tqdm.tqdm(self.kb.items()):
            for pw_ent_id in self.cand_sel.select(kb_ent_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                batch_json_data += form_entity_function(
                    0, pw_ent_id, self.pw[pw_ent_id], kb_ent_id, self.kb[kb_ent_id]
                )

        feat_gen = FeatureGenerator(batch_json_data)
        labels, features = feat_gen.compute_features()
        probabilities = model.predict_proba(features)
        pos = [(d['kb_id'], d['pw_id'], p[1]) for d, p in zip(batch_json_data, probabilities)]

        pos.sort(key=lambda x: x[2], reverse=True)
        return pos

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

    def _append_new_data(self, new_data, form_entity_function, train_data_path, dev_data_path):
        """
        Split new data and append to appropriate training data files
        :param new_data:
        :param form_entity_function:
        :param train_data_path:
        :param dev_data_path:
        :return:
        """
        new_training_data = base_utils.flatten([
            form_entity_function(label, pw_id, self.pw[pw_id], kb_id, self.kb[kb_id])
            for kb_id, pw_id, label in new_data
        ])

        new_train, new_dev, _ = pathway_utils.split_data(
            new_training_data,
            constants.DEV_DATA_PORTION + constants.TEST_DATA_PORTION,
            0
        )

        self._append_data_to_file(new_train, train_data_path)
        self._append_data_to_file(new_dev, dev_data_path)
        return

    def _combine_name_definition_predictions(self, predictions):
        """
        Combine scores from names and definitions
        :param predictions:
        :return:
        """
        pred_names = defaultdict(list)
        pred_def = defaultdict(list)

        for kb_id, pw_id, score, match, data_type in predictions:
            if data_type == 'name':
                pred_names[(kb_id, pw_id)].append(score)
            elif data_type == 'def':
                pred_def[(kb_id, pw_id)].append(score)
            else:
                raise Exception('Unknown data type, should be name or def')

        max_scores = dict()
        max_list = lambda x: max(x) if x else 0.0

        for k_pair in set(list(pred_names.keys()) + list(pred_def.keys())):
            max_scores[k_pair] = (
                max_list(pred_names[k_pair]),
                max_list(pred_def[k_pair])
            )

        max_combined = [(k[0], k[1], v[0], v[1]) for k, v in max_scores.items()]

        combined = [
            (entry[0], entry[1], constants.NAME_WEIGHT * entry[2] + constants.DEF_WEIGHT * entry[3])
            for entry in max_combined
        ]
        combined.sort(key=lambda x: x[2], reverse=True)

        return combined

    def _keep_new_predictions(self, predictions):
        """
        Retain only predictions which do not exist in the training data
        :param predictions:
        :return:
        """
        keep_top_n = int(constants.KEEP_TOP_N_PERCENT_MATCHES * len(predictions) / 2)

        keep_pos_pairs = predictions[:keep_top_n]
        keep_neg_pairs = predictions[len(predictions) - keep_top_n:]

        pos_pairs = [(p[0], p[1], 1) for p in keep_pos_pairs]
        neg_pairs = [(p[0], p[1], 0) for p in keep_neg_pairs]

        id_pairs = set(pos_pairs + neg_pairs)

        self._append_new_data(
            id_pairs, pathway_utils.form_name_entries,
            self.name_train_data_path, self.name_dev_data_path
        )

        self._append_new_data(
            id_pairs, pathway_utils.form_definition_entries,
            self.def_train_data_path, self.def_dev_data_path
        )

        print('Appended %i positive and %i negatives instances to training and development data.'
              % (len(keep_pos_pairs), len(keep_neg_pairs)))
        return

    def _train_nn(self, model_dir, nn_config_file) -> str:
        """
        Train NN
        :param iter:
        :return: Path to NN model
        """
        assert not(os.path.exists(model_dir))
        train_model_from_file(nn_config_file, model_dir)
        model_path = os.path.join(model_dir, "model.tar.gz")
        return model_path

    def _match_kb(self, name_model, def_model, batch_size, cuda_device):
        """
        Apply name and definition model to KB
        :param name_model:
        :param def_model:
        :return:
        """
        # load models as predictors from archive file
        name_predictor = Predictor.from_archive(
            load_archive(name_model, cuda_device=cuda_device),
            'pw_aligner'
        )

        # apply predictor to kb of interest
        name_matches = self._apply_model_to_kb(
            name_predictor, pathway_utils.form_name_entries, batch_size
        )
        name_matches = [list(i) + ['name'] for i in name_matches]

        # load models as predictors from archive file
        def_predictor = Predictor.from_archive(
            load_archive(def_model, cuda_device=cuda_device),
            'pw_aligner'
        )

        # apply predictor to kb of interest
        def_matches = self._apply_model_to_kb(
            def_predictor, pathway_utils.form_definition_entries, batch_size
        )
        def_matches = [list(i) + ['def'] for i in def_matches]

        return name_matches + def_matches

    def _evaluate_models(self, name_model_file, def_model_file, batch_size, cuda_device):
        """

        :param name_model_file:
        :param def_model_file:
        :param batch_size:
        :param cuda_device:
        :return:
        """
        # Load from archive
        archive = load_archive(name_model_file, cuda_device)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        model.eval()

        # Load the evaluation data
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
        dataset = dataset_reader.read(self.name_test_data_path)

        # compute metrics
        dataset.index_instances(model.vocab)
        iterator = DataIterator.from_params(config.pop("iterator"))
        metrics = evaluate_allennlp(model, dataset, iterator, cuda_device)

        print('Name: p, r, a, f1 = ', metrics)

        # Load from archive
        archive = load_archive(def_model_file, cuda_device)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        model.eval()

        # Load the evaluation data
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
        dataset = dataset_reader.read(self.def_test_data_path)

        # compute metrics
        dataset.index_instances(model.vocab)
        iterator = DataIterator.from_params(config.pop("iterator"))
        metrics = evaluate_allennlp(model, dataset, iterator, cuda_device)

        print('Def: p, r, a, f1 = ', metrics)

        return

    def _write_matches_to_file(self, matches, outfile):
        """
        Writing output matches to file
        :param matches:
        :param outfile:
        :return:
        """
        group_by_kb_id = defaultdict(list)

        for kb_id, pw_id, score in matches:
            group_by_kb_id[kb_id].append((pw_id, score))

        kb_ids = list(group_by_kb_id.keys())
        kb_ids.sort()

        with open(outfile, 'w') as outf:
            outf.write('Score\tPathway_id\tPathway_name\tPathway_definition\tPW_id\tPW_name\tPW_definition\n')
            for kb_id in kb_ids:
                pw_matches = group_by_kb_id[kb_id]
                pw_matches.sort(key=lambda x: x[1], reverse=True)
                for pw_id, score in pw_matches[:constants.KEEP_TOP_N_MATCHES]:
                    outf.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        score,
                        kb_id,
                        self.kb[kb_id]['name'],
                        self.kb[kb_id]['definition'],
                        pw_id,
                        self.pw[pw_id]['name'],
                        self.pw[pw_id]['definition']
                    ))
        return

    def bootstrap_model(self, total_iter: int):
        """
        Bootstrap training data for model
        :param total_iter: total bootstrapping iterations
        :return:
        """
        print('Extracting training data from PW...')
        extractor = TrainingDataExtractor()
        extractor.extract_training_data()

        for i in range(0, total_iter):
            sys.stdout.write('\n\n')
            sys.stdout.write('--------------\n')
            sys.stdout.write('Iteration: %i\n' % i + 1)
            sys.stdout.write('--------------\n')

            train_data = base_utils.read_jsonlines(self.name_train_data_path)
            train_data += base_utils.read_jsonlines(self.name_dev_data_path)

            # train linear model on engineered features on names to get best matches
            feat_gen = FeatureGenerator(train_data)
            labels, features = feat_gen.compute_features()

            lr_model = LogisticRegressionCV()
            lr_model.fit(features, labels)

            # apply trained model to KB
            print('Applying to input KB...')
            matches = self._apply_lr_to_kb(lr_model, pathway_utils.form_name_entries)

            # keep portion of matches with high confidence
            print('Determining predictions to keep...')
            self._keep_new_predictions(matches)
        return

    def train_model(self, output_dir, batch_size=32, cuda_device=-1):
        """
        Train final model
        :param output_dir:
        :param batch_size:
        :param cuda_device:
        :return:
        """
        # train name nn model
        print('Training model on pathway names...')
        name_model_dir = os.path.join(output_dir, 'name_model')
        name_model_file = self._train_nn(name_model_dir, self.nn_name_config_file)

        # train definition nn model
        print('Training model on pathway definitions...')
        def_model_dir = os.path.join(output_dir, 'def_model')
        def_model_file = self._train_nn(def_model_dir, self.nn_def_config_file)

        # Validate models on test data
        print('Validating model on test data...')
        self._evaluate_models(
            name_model_file, def_model_file, batch_size, cuda_device
        )
        return

    def run_model(self, name_model, def_model, output_dir, batch_size=32, cuda_device=-1):
        """
        Apply model to input data
        :return:
        """

        # match entities between bootstrap KB and PW
        print('Applying to input KB...')
        matches = self._match_kb(
            name_model, def_model, batch_size, cuda_device
        )
        sorted_matches = self._combine_name_definition_predictions(matches)

        # group and write matches to file
        print('Grouping outputs and writing to file...')
        output_file = os.path.join(output_dir, 'final_matches.tsv')
        self._write_matches_to_file(sorted_matches, output_file)

        print('Matches saved to %s' % output_file)
        print('done.')
        return


