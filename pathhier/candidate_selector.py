import numpy as np
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

import pathhier.constants as constants
import pathhier.utils.base_utils as base_utils
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

        # nltk snowball stemmer
        self.stemmer = SnowballStemmer('english')

        # retain only stop words of two or more letters because of usefulness of one letter words in pathway corpus
        self.STOP = set([w for w in stopwords.words('english') if len(w) > 1])
        self.STOP.update(['pathway', 'disease', 'diseases', 'signaling'])

        # dictionary mapping integer key to word
        self.vocab = {}
        self.ngrams = {}

        # dictionaries mapping tokens and ngrams to entity ids
        self.s_token_to_ents = defaultdict(set)
        self.t_token_to_ents = defaultdict(set)
        self.s_ngram_to_ents = defaultdict(set)
        self.t_ngram_to_ents = defaultdict(set)

        # idf dicts for tokens
        self.s_token_to_idf = dict()
        self.t_token_to_idf = dict()

        self.tokenize_kbs()
        self.compute_idfs()

    def compute_mapping_dicts(self, kb, word_dict, ngram_dict):
        """
        Compute both token and ngram mapping dicts and add to overall vocab
        :param kb:
        :param word_dict:
        :param ngram_dict:
        :return:
        """
        token_to_ents = defaultdict(set)
        ngram_to_ents = defaultdict(set)

        # generate token indices for source_kb
        for ent_id, ent_info in kb.items():
            name_tokens = string_utils.tokenize_string(ent_info['name'], self.tokenizer, self.STOP)
            name_token_stems = [self.stemmer.stem(tok) for tok in name_tokens]
            name_token_ids = [word_dict.get(token) for token in name_token_stems]
            kb[ent_id]['name_tokens'] = name_token_ids
            for token_id in name_token_ids:
                token_to_ents[token_id].add(ent_id)

            name_ngrams = string_utils.get_character_ngrams(ent_info['name'], constants.CHARACTER_NGRAM_LEN)
            name_ngram_ids = [ngram_dict.get(ngram) for ngram in name_ngrams]
            kb[ent_id]['name_ngrams'] = name_ngram_ids
            for ngram_id in name_ngram_ids:
                ngram_to_ents[ngram_id].add(ent_id)

            kb[ent_id]['alias_tokens'] = []
            for alias in ent_info['aliases']:
                alias_tokens = string_utils.tokenize_string(alias, self.tokenizer, self.STOP)
                alias_token_stems = [self.stemmer.stem(tok) for tok in alias_tokens]
                alias_token_ids = [word_dict.get(token) for token in alias_token_stems]
                kb[ent_id]['alias_tokens'].append(alias_token_ids)
                for token_id in alias_token_ids:
                    token_to_ents[token_id].add(ent_id)

            def_tokens = []
            for d_string in ent_info['definition']:
                def_tokens += string_utils.tokenize_string(d_string, self.tokenizer, self.STOP)
            def_token_stems = [self.stemmer.stem(tok) for tok in def_tokens]
            def_token_ids = [word_dict.get(token) for token in def_token_stems]
            kb[ent_id]['def_tokens'] = def_token_ids
            for token_id in def_token_ids:
                token_to_ents[token_id].add(ent_id)

            kb[ent_id]['all_tokens'] = set(base_utils.flatten(kb[ent_id]['alias_tokens']) + kb[ent_id]['def_tokens'])

        return kb, token_to_ents, ngram_to_ents, word_dict, ngram_dict

    def tokenize_kbs(self):
        """
        Tokenize two kbs, contruct a vocabulary, and create indices for looking up entities
        :return:
        """
        # word/ngram to integer index mapping dicts
        word_to_id = IncrementDict()
        ngram_to_id = IncrementDict()

        # generate token indices for source_kb
        self.s_kb, self.s_token_to_ents, self.s_ngram_to_ents, word_to_id, ngram_to_id = self.compute_mapping_dicts(
            self.s_kb, word_to_id, ngram_to_id
        )

        # generate token indices for target_kb
        self.t_kb, self.t_token_to_ents, self.t_ngram_to_ents, word_to_id, ngram_to_id = self.compute_mapping_dicts(
            self.t_kb, word_to_id, ngram_to_id
        )

        # create vocabulary lookup dicts
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

        # add idf score of each token to the target match entry in the target_matches dictionary
        for token in s_tokens:
            if self.s_token_to_ents.get(token) and self.t_token_to_ents.get(token):
                for t_match in self.t_token_to_ents[token]:
                    target_matches[t_match] += self.t_token_to_idf[token]

        # add ngram matches to target match list
        for ngram in s_ngrams:
            if self.s_ngram_to_ents.get(ngram) and self.t_ngram_to_ents.get(ngram):
                for t_match in self.t_ngram_to_ents[ngram]:
                     target_matches[t_match] += 0.1

        # convert dictionary to list of tuples
        t_matches = [(k, v) for k, v in target_matches.items()]

        # sort target matches
        t_matches.sort(key=lambda x: x[1], reverse=True)

        return [m[0] for m in t_matches]
