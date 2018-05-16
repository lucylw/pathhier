
import scipy.sparse as sp


class FeatureGenerator:
    def __init__(self, data, vocab):
        """
        Initialize feature vector
        """
        self.data_dict = data
        self.name_vocab, self.def_vocab = vocab
        self.name_vocab_size = len(self.name_vocab)
        self.def_vocab_size = len(self.def_vocab)

    def compute_sparse_features(self, data):
        """
        Compute sparse features from data
        :return:
        """
        labels = []
        feature_mat = []

        for pair in data:
            labels.append(int(pair['label']))
            kb_ent_id = pair['kb_ent']['id']
            pw_ent_id = pair['pw_ent']['id']
            diff = abs(self.data_dict[kb_ent_id] - self.data_dict[pw_ent_id])
            feature_mat.append(diff)

        v = sp.vstack(feature_mat, format='csr')
        return labels, v
