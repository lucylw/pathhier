import tqdm
from collections import defaultdict

import numpy as np
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


class FeatureGenerator:
    def __init__(self, data, vocab):
        """
        Initialize feature vector
        """
        self.data_dict = data
        self.vocab_dict = vocab
        self.vocab_lengths = {k: len(v) for k, v in vocab.items()}

    def compute_features(self, pairs, show_progress=False):
        """
        Compute sparse features from data
        :param pairs:
        :param show_progress:
        :return:
        """
        labels = list()
        feature_mat = defaultdict(list)

        if show_progress:
            pairs = tqdm.tqdm(pairs)

        for pair in pairs:
            labels.append(int(pair['label']))
            kb_ent_vec = self.data_dict[pair['kb_ent']['id']]
            pw_ent_vec = self.data_dict[pair['pw_ent']['id']]

            for k in kb_ent_vec:
                feature_mat[k + '_l2norm'].append(norm(kb_ent_vec[k] - pw_ent_vec[k]).item())
                feature_mat[k + '_cossim'].append(cosine_similarity(kb_ent_vec[k], pw_ent_vec[k]).item())
                feature_mat[k + '_lendiff'].append(abs(np.sum(kb_ent_vec[k]) - np.sum(pw_ent_vec[k])))

            feature_mat['name_token_pw_def_token_l2norm'].append(
                norm(kb_ent_vec['name_token'] - pw_ent_vec['def_token']).item()
            )
            feature_mat['name_token_pw_def_token_cossim'].append(
                cosine_similarity(kb_ent_vec['name_token'], pw_ent_vec['def_token']).item()
            )

        features = [(k, v) for k, v in feature_mat.items()]
        features.sort(key=lambda x: x[0])

        f_vals = [i[1] for i in features]

        v = np.vstack(f_vals).transpose()

        return labels, v
