import numpy as np
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import pathhier.constants as constants
import pathhier.utils.string_utils as string_utils
from pathhier.utils.utility_classes import IncrementDict

# class for selecting candidates in target kb for annotation
class CandidateSelector:
    def __init__(self, s_kb, t_kb):
        """
        Initialize and build mapping dictionaries for tokens in source and target KB
        :param s_kb:
        :param t_kb:
        """
        self.s_kb = s_kb
        self.t_kb = t_kb

        # number of entities in kbs
        self.s_doc_total = len(self.s_kb) + 1
        self.t_doc_total = len(self.t_kb) + 1

        # nltk word tokenizer
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')

        # retain only stop words of two or more letters because of usefulness of one letter words in pathway corpus
        self.STOP = set([w for w in stopwords.words('english') if len(w) > 1])

        # dictionary mapping integer key to word
        self.vocab = {}
        self.ngrams = {}

        # dictionaries mapping tokens and ngrams to entity ids
        self.s_token_to_ents = defaultdict(set)
        self.t_token_to_ents = defaultdict(set)
        self.s_ngram_to_ents = defaultdict(set)
        self.t_ngram_to_ents = defaultdict(set)

        # # dictionaries mapping entity ids to tokens and ngrams
        # self.s_ent_to_tokens = dict()
        # self.t_ent_to_tokens = dict()
        # self.s_ent_to_ngrams = dict()
        # self.t_ent_to_ngrams = dict()

        # idf dicts for tokens
        self.s_token_to_idf = dict()
        self.t_token_to_idf = dict()

        self.tokenize_kbs()
        self.compute_idfs()

    def tokenize_kbs(self):
        """
        Tokenize two kbs, contruct a vocabulary, and create indices for looking up entities
        :return:
        """
        # word/ngram to integer index mapping dicts
        word_to_id = IncrementDict()
        ngram_to_id = IncrementDict()

        # generate token indices for source_kb
        for ent_id, ent_info in self.s_kb:
            name_tokens = string_utils.tokenize_string(ent_info['name'], self.tokenizer, self.STOP)
            name_token_ids = [word_to_id.get(token) for token in name_tokens]
            self.s_kb[ent_id]['name_tokens'] = name_token_ids
            for token_id in name_token_ids:
                self.s_token_to_ents[token_id].add(ent_id)

            def_tokens = string_utils.tokenize_string(ent_info['definition'], self.tokenizer, self.STOP)
            def_token_ids = [word_to_id.get(token) for token in def_tokens]
            self.s_kb[ent_id]['def_tokens'] = def_token_ids
            for token_id in def_token_ids:
                self.s_token_to_ents[token_id].add(ent_id)

            name_ngrams = string_utils.get_character_ngrams(ent_info['name'], constants.CHARACTER_NGRAM_LEN)
            name_ngram_ids = [ngram_to_id.get(ngram) for ngram in name_ngrams]
            self.s_kb[ent_id]['name_ngrams'] = name_ngram_ids
            for ngram_id in name_ngram_ids:
                self.s_ngram_to_ents[ngram_id].add(ent_id)

        # generate token indices for target_kb
        for ent_id, ent_info in self.t_kb:
            name_tokens = string_utils.tokenize_string(ent_info['name'], self.tokenizer, self.STOP)
            name_token_ids = [word_to_id.get(token) for token in name_tokens]
            self.t_kb[ent_id]['name_tokens'] = name_token_ids
            for token_id in name_token_ids:
                self.t_token_to_ents[token_id].add(ent_id)

            def_tokens = string_utils.tokenize_string(ent_info['definition'], self.tokenizer, self.STOP)
            def_token_ids = [word_to_id.get(token) for token in def_tokens]
            self.t_kb[ent_id]['def_tokens'] = def_token_ids
            for token_id in def_token_ids:
                self.t_token_to_ents[token_id].add(ent_id)

            name_ngrams = string_utils.get_character_ngrams(ent_info['name'], constants.CHARACTER_NGRAM_LEN)
            name_ngram_ids = [ngram_to_id.get(ngram) for ngram in name_ngrams]
            self.t_kb[ent_id]['name_ngrams'] = name_ngram_ids
            for ngram_id in name_ngram_ids:
                self.t_ngram_to_ents[ngram_id].add(ent_id)

        # create vocabulary lookup dict
        self.vocab = {v: k for k, v in word_to_id.content.items()}
        self.ngrams = {v: k for k, v in ngram_to_id.content.items()}

        return

    def compute_idfs(self):
        """
        Compute inverse document frequency for each word token
        :return:
        """
        for token_id in self.vocab:
            self.s_token_to_idf[token_id] = np.log(
                self.s_doc_total / (len(self.s_token_to_ents[token_id]) + 1)
            )
            self.t_token_to_idf[token_id] = np.log(
                self.t_doc_total / (len(self.t_token_to_ents[token_id]) + 1)
            )
        return

    def select(self, s_ent_id):
        """
        Given an s_ent_id, generate the list of candidates from t_kb by descending order of idf
        :param s_ent_id:
        :return:
        """
        s_tokens = self.s_kb[s_ent_id]['name_tokens'] + self.s_kb[s_ent_id]['def_tokens']
        s_ngrams = self.s_kb[s_ent_id]['name_ngrams']

        target_matches = defaultdict(float)

        for token in s_tokens:
            if self.s_token_to_ents.get(token) and self.t_token_to_ents.get(token) \
                    and self.s_token_to_idf[token] >= constants.IDF_LIMIT \
                    and self.t_token_to_idf[token] >= constants.IDF_LIMIT:
                for t_match in self.t_token_to_ents[token]:
                    target_matches[t_match] += self.t_token_to_idf[token]

        t_matches = [(k, v) for k, v in target_matches.items()]
        t_matches.sort(key=lambda x: x[1], reverse=True)

        for ngram in s_ngrams:
            if self.s_ngram_to_ents.get(ngram) and self.t_ngram_to_ents.get(ngram):
                for t_match in self.t_ngram_to_ents[ngram]:
                    if target_matches[t_match] == 0.0:
                        target_matches[t_match] += 0.1
                        t_matches.append((t_match, 0.1))
        return t_matches