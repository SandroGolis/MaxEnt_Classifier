# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy as np

LEARNING_RATE = 0.0001
BATCH_SIZE = 30


class MaxEntModel(object):
    """
          This class will contain all needed information for MaxEnt model.
          1. map_name_to_id       type: Dictionary(String->Integer)
                                        feature_name -> feature_id
            feature_id is the row number of corresponding feature in
            weights_matrix 2d array.
          2. map_label_to_col     type: Dictionary(String->Integer)
                                        class_name -> class_column
            class_column is the column number of corresponding label
            in weights_matrix 2d array.
          3. weights_matrix       type: numpy.array(rows, columns)
                        class_A     class_B     . . .      class_X
                      |----------|-----------|----------|----------|
               f_id_1 |   w1_A   |   w1_B    |          |   w1_X   |
               f_id_2 |   w2_A   |   w2_B    |          |   w2_X   |
                 .    |          |           |          |          |
                 .    |          |           |          |          |
                 .    |          |           |          |          |
               f_id_N |   wN_A   |   wN_B    |          |   wN_X   |
                      |----------|-----------|----------|----------|

            For example: if map_name_to_id[feature1] = f_id_1 then
                         w1_A is a weight corresponding to feature
               f_1A(X, C) = 1 if C=classA AND feature1 is in features(X)
                            0 otherwise
            4. observed count     type: numpy.array(rows, columns)
               matrix with the same shape as weights_matrix.
               The values will be empirical counts of features in the training set.
    """
    def __init__(self, map_name_to_id, labels):
        super(MaxEntModel, self).__init__()
        self.map_label_to_col = labels
        self.map_name_to_id = map_name_to_id
        self.rows = len(map_name_to_id)
        self.cols = len(labels)
        self.weights_matrix = np.zeros((self.rows, self.cols))
        self.observed_count = np.array()


class MaxEnt(Classifier):

    def __init__(self, new_model=None):
        super(MaxEnt, self).__init__(new_model)

    def get_model(self): return self._model

    def set_model(self, new_model): self._model = new_model

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """
                Construct a statistical model from labeled instances.
                We will build 2 data structures
                1. model.map_name_to_id (type: dictionary)
                   mapping: feature_name ---> feature_id
                2. model.map_label_to_col (type: dictionary)
                   mapping: class_name ---> class_column

                :type instances: list(Documents)
                :param instances: Training data, represented as a list of Documents

                :type dev_instances: list(Documents)
                :param dev_instances: Dev data, represented as a list of Documents
        """
        map_name_to_id = {}
        labels = {}
        for instance in instances:
            fnames = instance.features()
            label = instance.label
            if label not in labels:
                labels[label] = len(labels)

            for fname in fnames:
                if fname not in map_name_to_id:
                    map_name_to_id[fname] = len(map_name_to_id)

        self.model = MaxEntModel(map_name_to_id, labels)
        self.train_sgd(instances, dev_instances, LEARNING_RATE, BATCH_SIZE)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """
            Train MaxEnt model with Mini-batch Stochastic Gradient.
            :type train_instances: list(Documents)
            :param train_instances: Training data, represented as a list of Documents

            :type dev_instances: list(Documents)
            :param dev_instances: Training data, represented as a list of Documents

            :type learning_rate: Integer
            :param learning_rate: Parameter for Mini-batch Stochastic Gradient

            :type batch_size: float64
            :param batch_size: Parameter for Mini-batch Stochastic Gradient
        """
        # Calculate empirical count of features in the given training set.
        self.model.observed_count = self.calc_empirical_count(train_instances)

        # Calculate expected count with current model
        expected_count = self.calc_expected_count(train_instances)




    def classify(self, instance):
        fnames = instance.features()
        # sparse representation is a list of feature id's
        # feature_id = feature_raw in weights_matrix
        sparse_vector = self.get_sparse_vector(fnames)
        prob_dict = {}
        for label in self.model.map_label_to_col:
            prob = self.get_prob_given_X(sparse_vector, label)
            prob_dict[label] = prob

        return max(prob_dict, key=prob_dict.get)

    def get_sparse_vector(self, fnames):
        # todo what if feature name doesn't appear in mapping
        return [self.model.map_name_to_id[name] for name in fnames]

    def get_prob_given_X(self, sparse_vector, label):
        col = self.model.map_label_to_col[label]
        # todo exponents and normalization here?
        return self.model.weights_matrix[sparse_vector, col].sum()

    def calc_empirical_count(self, train_instances):
        observed_counts = np.zeros((self.model.rows, self.model.cols))
        for instance in train_instances:
            sparse_vector = self.get_sparse_vector(instance.features())
            col = self.model.map_label_to_col[instance.label]
            observed_counts[sparse_vector, col] += 1

        return observed_counts

    def calc_expected_count(self, train_instances):
        """
            Count the probabilities of current model
            1. For each document
            2. For each class
                compute the probability P(y=class | document)
                and add to output matrix

        :param train_instances: type: list(Documents)
        :return: type: numpy.array(rows, columns)
                 with same shape like model.observed_count
        """
        expected_counts = np.zeros((self.model.rows, self.model.cols))
        for instance in train_instances:
            sparse_vector = self.get_sparse_vector(instance.features())
            for label, col in self.model.map_label_to_col:
                prob = self.get_prob_given_X(sparse_vector, label)
                expected_counts[sparse_vector, col] += prob

        return expected_counts



