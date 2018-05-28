# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import pickle
import tqdm
from datetime import datetime

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from pathhier.matcher_model import PWMatcher
from pathhier.candidate_selector import CandidateSelector
from pathhier.utils.utility_classes import IncrementDict
import pathhier.utils.base_utils as base_utils
import pathhier.utils.string_utils as string_utils
from pathhier.paths import PathhierPaths
import pathhier.constants as constants


class PWAligner:
    def __init__(self, orig_data_file, kb_path, pw_path):
        # get output directory for model and temp files
        paths = PathhierPaths()
        self.output_dir = os.path.join(
            paths.output_dir,
            '{}-{}'.format('model',
                           datetime.now().strftime('%Y-%m-%d'))
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # load bootstrap KB from file
        assert os.path.exists(kb_path)
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)

        # load PW from file
        assert os.path.exists(pw_path)
        with open(pw_path, 'r') as f:
            self.pw = json.load(f)

        # set live training data file and load initial data
        if orig_data_file:
            assert os.path.exists(orig_data_file)
            self.live_data_file = orig_data_file
            self.init_data = self._read_tsv_file(orig_data_file)

            # load data and vocab from disk if available
            self.preprocessed_data_file = os.path.join(paths.output_dir, 'preprocessed_data.pickle')
            self.vocab_file = os.path.join(paths.output_dir, 'vocab.pickle')

            if os.path.exists(self.preprocessed_data_file) and os.path.exists(self.vocab_file):
                self.all_data = pickle.load(open(self.preprocessed_data_file, 'rb'))
                self.vocab = pickle.load(open(self.vocab_file, 'rb'))
            else:
                # collect all data in single lookup dict and compute vocab
                self.all_data, self.vocab = self._collect_and_preprocess_all_data()

            # convert each entity into vector representation
            self.all_vectors = self._convert_data_to_vector_rep()

            # initialize model
            self.model = PWMatcher(self.all_vectors, self.vocab)

    @staticmethod
    def _read_tsv_file(d_file):
        """
        Read specified file and return content minus header
        :param d_file:
        :return:
        """
        data = []
        with open(d_file) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                data.append(row)
        return data

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

    def _collect_and_preprocess_all_data(self):
        """
        Collect data from initial training file, KB, and PW, generating one lookup dict with tokenized values
        :return:
        """
        # nltk word tokenizer
        tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')

        # retain only stop words of two or more letters because of usefulness of one letter words in pathway corpus
        STOP = set([w for w in stopwords.words('english') if len(w) > 1])

        # initialize data dict
        all_data = dict()

        # word/ngram to integer index mapping dicts
        map_dict = {
            'name_token': IncrementDict(),
            'name_bigram': IncrementDict(),
            'name_trigram': IncrementDict(),
            'name_char_ngram': IncrementDict()
        }

        # add UNK to both dicts
        map_dict['name_token'].get('\0')
        map_dict['name_bigram'].get('\0')
        map_dict['name_trigram'].get('\0')
        map_dict['name_char_ngram'].get('\0')

        # populate with training data
        sys.stdout.write("Preprocessing training data...\n")
        for _, _, pw_id, pw_name, pw_def, kb_id, kb_name, kb_def in self.init_data:
            all_data[pw_id] = {
                'name_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(pw_name, tokenizer, STOP)],
                'name_bigram': [map_dict['name_bigram'].get(bg) for bg in string_utils.get_token_ngrams(pw_name, tokenizer, 2)],
                'name_trigram': [map_dict['name_trigram'].get(tg) for tg in string_utils.get_token_ngrams(pw_name, tokenizer, 3)],
                'name_char_ngram': [map_dict['name_char_ngram'].get(ng) for ng in string_utils.get_character_ngrams(pw_name, 5)],
                'def_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(pw_def, tokenizer, STOP)]
            }
            all_data[kb_id] = {
                'name_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(kb_name, tokenizer, STOP)],
                'name_bigram': [map_dict['name_bigram'].get(bg) for bg in string_utils.get_token_ngrams(kb_name, tokenizer, 2)],
                'name_trigram': [map_dict['name_trigram'].get(tg) for tg in string_utils.get_token_ngrams(kb_name, tokenizer, 3)],
                'name_char_ngram': [map_dict['name_char_ngram'].get(ng) for ng in string_utils.get_character_ngrams(kb_name, 5)],
                'def_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(kb_def, tokenizer, STOP)]
            }

        # populate with KB data
        sys.stdout.write("Preprocessing KB data...\n")
        for kb_ent_id, kb_ent_values in self.kb.items():

            kb_ent_definition = ''
            if kb_ent_values['definition']:
                kb_ent_definition = kb_ent_values['definition'][0]

            all_data[kb_ent_id] = {
                'name_token': [map_dict['name_token'].get(tok) for tok in base_utils.flatten(
                    [string_utils.tokenize_string(name, tokenizer, STOP) for name in kb_ent_values['aliases']]
                )],
                'name_bigram': [map_dict['name_bigram'].get(bg) for bg in base_utils.flatten(
                    [string_utils.get_token_ngrams(name, tokenizer, 2) for name in kb_ent_values['aliases']]
                )],
                'name_trigram': [map_dict['name_trigram'].get(tg) for tg in base_utils.flatten(
                    [string_utils.get_token_ngrams(name, tokenizer, 3) for name in kb_ent_values['aliases']]
                )],
                'name_char_ngram': [map_dict['name_char_ngram'].get(ng) for ng in base_utils.flatten(
                    [string_utils.get_character_ngrams(name, 5) for name in kb_ent_values['aliases']]
                )],
                'def_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(
                    kb_ent_definition, tokenizer, STOP
                )]
            }

        # populate with PW data
        sys.stdout.write("Preprocessing PW data...\n")
        for pw_ent_id, pw_ent_values in self.pw.items():
            if pw_ent_id not in all_data:
                pw_ent_definition = ''
                if pw_ent_values['definition']:
                    pw_ent_definition = pw_ent_values['definition'][0]

                all_data[pw_ent_id] = {
                    'name_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(
                        pw_ent_values['name'], tokenizer, STOP)],
                    'name_bigram': [map_dict['name_bigram'].get(bg) for bg in string_utils.get_token_ngrams(
                        pw_ent_values['name'], tokenizer, 2)],
                    'name_trigram': [map_dict['name_trigram'].get(tg) for tg in string_utils.get_token_ngrams(
                        pw_ent_values['name'], tokenizer, 3)],
                    'name_char_ngram': [map_dict['name_char_ngram'].get(ng) for ng in string_utils.get_character_ngrams(
                        pw_ent_values['name'], 5)],
                    'def_token': [map_dict['name_token'].get(tok) for tok in string_utils.tokenize_string(
                        pw_ent_definition, tokenizer, STOP)]
                }

        # save data to disk
        sys.stdout.write("Saving preprocessed data...\n")
        pickle.dump(all_data, open(self.preprocessed_data_file, 'wb'))

        # create vocabulary lookup dicts
        vocab = {
            'name_token': {v: k for k, v in map_dict['name_token'].content.items()},
            'name_bigram': {v: k for k, v in map_dict['name_bigram'].content.items()},
            'name_trigram': {v: k for k, v in map_dict['name_trigram'].content.items()},
            'name_char_ngram': {v: k for k, v in map_dict['name_char_ngram'].content.items()}
        }
        vocab['def_token'] = vocab['name_token']

        pickle.dump(vocab, open(self.vocab_file, 'wb'))

        return all_data, vocab

    def _convert_data_to_vector_rep(self):
        """
        Convert data tokens to vector representation
        :return:
        """
        sys.stdout.write("Converting to vector representation...\n")

        # get lengths of various vocabs
        vocab_lengths = {k: len(v) for k, v in self.vocab.items()}

        vector_dict = dict()

        # iterate through all entities and generate vector from tokens
        for ent_id, ent_vals in tqdm.tqdm(self.all_data.items()):

            vector_dict[ent_id] = dict()

            for token_type, tokens in ent_vals.items():
                v_array = np.zeros(vocab_lengths[token_type])
                for tok in tokens:
                    v_array[tok] += 1
                vector_dict[ent_id][token_type] = csr_matrix(v_array)

        return vector_dict

    def _process_data(self, d_file):
        """
        Read training data and split into training/development set
        :param d_file:
        :return:
        """
        # read training data from file
        data = self._read_tsv_file(d_file)

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

    def _apply_model_to_kb(self, iter_num=None):
        """
        Apply model to bootstrap KB
        :param iter_num: iteration number
        :return:
        """
        cand_sel = CandidateSelector(self.kb, self.pw)
        test_data = []

        if iter_num:
            provenance = 'bootstrap_iter{}'.format(iter_num)
        else:
            provenance = 'PW_aligner'

        for kb_ent_id, kb_ent_values in tqdm.tqdm(self.kb.items()):
            for pw_ent_id in cand_sel.select(kb_ent_id)[:constants.KEEP_TOP_N_CANDIDATES]:
                pw_ent = self.pw[pw_ent_id]
                t_values = (provenance,
                            pw_ent_id, pw_ent['name'], pw_ent['definition'],
                            kb_ent_id, kb_ent_values['name'], kb_ent_values['definition'])
                test_data.append(self._form_training_entity('-1', t_values))

        # compute similarity scores using model
        predicted_scores = self.model.test(test_data)

        # zip together with data and sort by similarity score
        predictions = zip([s[1] for s in predicted_scores], test_data)
        predictions = [pred for pred in predictions if pred[0] >= constants.SIMSCORE_THRESHOLD]
        predictions.sort(key=lambda x: x[0], reverse=True)

        return predictions

    @staticmethod
    def _keep_new_predictions(predictions, previous_data):
        """
        Retain only predictions which do not exist in the training data
        :param predictions:
        :param previous_data:
        :return:
        """
        prev_id_pairs = [(pair['kb_ent']['id'], pair['pw_ent']['id']) for pair in previous_data]

        novel = []
        for score, data_entry in predictions:
            if (data_entry['kb_ent']['id'], data_entry['pw_ent']['id']) not in prev_id_pairs:
                novel.append(data_entry)

        return novel

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
            sys.stdout.write("\tTraining: %i, Development: %i\n" % (len(train_data), len(dev_data)))

            # write data to file
            self._write_data_to_file(train_data, train_output_file)
            self._write_data_to_file(dev_data, dev_output_file)

            # train model on training data
            self.model.train(train_data, dev_data)

            # save model to file
            pickle.dump(self.model, open(model_file, 'wb'))

            # match entities between bootstrap KB and PW
            sys.stdout.write("\tApplying model to KB...\n")
            predicted_positives = self._apply_model_to_kb(i)

            # determine what to keep in bootstrap iteration
            novel_predictions = self._keep_new_predictions(predicted_positives, train_data + dev_data)
            keep_top_n = int(constants.KEEP_TOP_N_PERCENT_MATCHES * len(predicted_positives))
            to_add = novel_predictions[:keep_top_n]

            for entry in to_add:
                entry['label'] = 1

            # append to previous round data
            new_data = train_data + dev_data + to_add
            sys.stdout.write('\tAdded %i positives to training data\n' % len(to_add))

            # write all new data to file
            self._write_data_to_file(new_data, all_data_file)
            self.live_data_file = all_data_file

    def run_model(self, model_file):
        """
        Apply model to input data
        :return:
        """
        # load model from disk
        self.model = pickle.load(open(model_file, 'rb'))

        # match entities between bootstrap KB and PW
        sys.stdout.write("\tApplying model to KB...\n")
        predicted_positives = self._apply_model_to_kb()

        # write outputs to file
        output_file = os.path.join(self.output_dir, 'predicted_positives.tsv')
        self._write_data_to_file(predicted_positives, output_file)

