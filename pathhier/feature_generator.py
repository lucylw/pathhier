
import numpy as np


class FeatureGenerator:
    def __init__(self, vocab):
        """
        Initialize feature vector
        """
        self.vocab = vocab

    def compute_sparse_features(self, data):
        """
        Compute sparse features from data
        :return:
        """
        raise NotImplementedError("Not implemented!")