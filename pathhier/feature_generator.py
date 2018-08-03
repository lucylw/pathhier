import tqdm
from collections import defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import pathhier.utils.string_utils as string_utils


class FeatureGenerator:
    def __init__(self, data):
        """
        Initialize feature vector
        """
        self.data = data

        # nltk word tokenizer
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')

        # retain only stop words of two or more letters because of usefulness of one letter words in pathway corpus
        self.STOP = set([w for w in stopwords.words('english') if len(w) > 1])
        self.STOP.update(['pathway', 'pathways'])

    def compute_one(self, pair):
        """
        Compute features for one pair
        :return:
        """
        features = dict()

        kb_tokens = string_utils.tokenize_string(pair['kb_cls'], self.tokenizer, self.STOP)
        pw_tokens = string_utils.tokenize_string(pair['pw_cls'], self.tokenizer, self.STOP)

        kb_3grams = string_utils.get_character_ngrams(pair['kb_cls'], 3)
        pw_3grams = string_utils.get_character_ngrams(pair['pw_cls'], 3)

        kb_4grams = string_utils.get_character_ngrams(pair['kb_cls'], 4)
        pw_4grams = string_utils.get_character_ngrams(pair['pw_cls'], 4)

        kb_5grams = string_utils.get_character_ngrams(pair['kb_cls'], 5)
        pw_5grams = string_utils.get_character_ngrams(pair['pw_cls'], 5)

        features['len_diff_perc'] = abs(len(kb_tokens) - len(pw_tokens))/len(kb_tokens)
        features['token_jaccard'] = string_utils.jaccard(set(kb_tokens), set(pw_tokens))
        features['3gram_jaccard'] = string_utils.jaccard(set(kb_3grams), set(pw_3grams))
        features['4gram_jaccard'] = string_utils.jaccard(set(kb_4grams), set(pw_4grams))
        features['5gram_jaccard'] = string_utils.jaccard(set(kb_5grams), set(pw_5grams))

        return features

    def compute_features(self, show_progress=False):
        """
        Compute sparse features from data
        :param show_progress:
        :return:
        """
        labels = list()
        feature_mat = defaultdict(list)

        if show_progress:
            pairs = tqdm.tqdm(self.data)
        else:
            pairs = self.data

        for pair in pairs:
            labels.append(int(pair['label']))
            pair_features = self.compute_one(pair)

            feature_mat['len_diff_perc'].append(pair_features['len_diff_perc'])
            feature_mat['token_jaccard'].append(pair_features['token_jaccard'])
            feature_mat['3gram_jaccard'].append(pair_features['3gram_jaccard'])
            feature_mat['4gram_jaccard'].append(pair_features['4gram_jaccard'])
            feature_mat['5gram_jaccard'].append(pair_features['5gram_jaccard'])

        features = [(k, v) for k, v in feature_mat.items()]
        features.sort(key=lambda x: x[0])

        f_vals = [i[1] for i in features]

        v = np.vstack(f_vals).transpose()

        return labels, v
