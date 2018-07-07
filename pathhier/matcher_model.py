
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pathhier.feature_generator import FeatureGenerator


# class for training a PW class aligner with bootstrapping
class PWMatcher:
    def __init__(self, data, vocab):
        """
        Initialize model
        """
        self.feat_gen = FeatureGenerator(data, vocab)
        self.model = RandomForestClassifier()

    def _compute_scores(self, predicted_labels, gold_labels):
        """
        Compute precision, recall, f1-score, and accuracy of predictions
        :param predicted:
        :param actual:
        :return:
        """
        tp = len([pred for pred, gold in zip(predicted_labels, gold_labels) if pred == 1 and gold == 1])
        fp = len([pred for pred, gold in zip(predicted_labels, gold_labels) if pred == 1 and gold == 0])
        fn = len([pred for pred, gold in zip(predicted_labels, gold_labels) if pred == 0 and gold == 1])
        tn = len([pred for pred, gold in zip(predicted_labels, gold_labels) if pred == 0 and gold == 0])
        total = len(gold_labels)

        p = tp / (tp + fp)          # precision
        r = tp / (tp + fn)          # recall
        f1 = 2 * p * r / (p + r)    # f1 score
        a = (tp + tn) / total       # accuracy

        return p, r, f1, a

    def train(self, train_data, dev_data):
        """
        Get features for training data and train model
        :param train_data:
        :param dev_data:
        :return:
        """
        train_labels, train_features = self.feat_gen.compute_features(train_data)
        self.model.fit(train_features, train_labels)

        dev_labels, dev_features = self.feat_gen.compute_features(dev_data)
        predicted_classes = self.model.predict(dev_features)

        p, r, f1, a = self._compute_scores(predicted_classes, dev_labels)
        sys.stdout.write('\tDevelopment: p, r, f1, a = %.2f, %.2f, %.2f, %.2f\n' % (p, r, f1, a))
        return

    def test(self, test_data):
        """
        Predict on test data
        :param test_data:
        :return:
        """
        _, test_features = self.feat_gen.compute_features(test_data, True)
        sim_scores = self.model.predict_proba(test_features)
        return sim_scores

