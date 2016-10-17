from corpus import Document, NamesCorpus
from key_maxent import MaxEnt
from random import shuffle, seed
import sys


class BagOfWords(Document):
    def features(self):
        """Trivially tokenize words."""

        return self.data.split()


class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]]


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)


def other_metrics(classifier, test, verbose=sys.stdout):
    m_test = [x for x in test if x.label == "male"]
    f_test = [x for x in test if x.label == "female"]
    m_correct = len([x for x in m_test if classifier.classify(x) == x.label])
    f_correct = len([x for x in f_test if classifier.classify(x) == x.label])
    m_incorrect = len(m_test) - m_correct
    f_incorrect = len(f_test) - f_correct
    total_len = len(m_test) + len(f_test)
    print("Test length: %d. \tMale/Female: %d/%d" % (total_len, len(m_test), len(f_test)))
    print("Male   correct/incorrect %d/%d" % (m_correct, m_incorrect))
    print("Female correct/incorrect %d/%d" % (f_correct, f_incorrect))

    accuracy_percent = 100 * (m_correct + f_correct) / (len(m_test) + len(f_test))
    classified_as_m = m_correct + f_incorrect
    if classified_as_m == 0:
        m_precision = 0
    else:
        m_precision = 100 * m_correct / classified_as_m
    m_recall = 100 * m_correct / len(m_test)

    classified_as_f = f_correct + m_incorrect
    if classified_as_f == 0:
        f_precision = 0
    else:
        f_precision = 100 * f_correct / classified_as_f
    f_recall = 100 * f_correct / len(f_test)

    if m_precision + m_recall == 0:
        f_m_measure = 0
    else:
        f_m_measure = (2 * m_precision * m_recall) / (m_precision + m_recall)

    if f_precision + f_recall == 0:
        f_f_measure = 0
    else:
        f_f_measure = (2 * f_precision * f_recall) / (f_precision + f_recall)
    if verbose:
        print("Accuracy:\t %6.2f%%" % accuracy_percent)
        print("------MALE------")
        print("Precision:\t %6.2f%% " % m_precision)
        print("Recall:\t\t %6.2f%% " % m_recall)
        print("F_Measure:\t %6.2f%% " % f_m_measure)
        print("-----FEMALE-----")
        print("Precision:\t %6.2f%% " % f_precision)
        print("Recall:\t\t %6.2f%% " % f_recall)
        print("F_Measure:\t %6.2f%% " % f_f_measure)
    return accuracy_percent / 100
    # class MaxEntTest(TestCase):
    #     u"""Tests for the MaxEnt classifier."""
    #
    #     def split_names_corpus(self, document_class=Name):
    #         """Split the names corpus into training, dev, and test sets"""
    #         names = NamesCorpus(document_class=document_class)
    #         self.assertEqual(len(names), 5001 + 2943)  # see names/README
    #         seed(hash("names"))
    #         shuffle(names)
    #         return names[:5000], names[5000:6000], names[6000:]
    #
    #     def test_names_nltk(self):
    #         """Classify names using NLTK features"""
    #         train, dev, test = self.split_names_corpus()
    #         classifier = MaxEnt()
    #         classifier.train(train, dev)
    #         acc = accuracy(classifier, test)
    #         self.assertGreater(acc, 0.70)
    #
    #     @staticmethod
    #     def split_review_corpus(document_class):
    #         """Split the yelp review corpus into training, dev, and test sets"""
    #         reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
    #         seed(hash("reviews"))
    #         shuffle(reviews)
    #         return reviews[:10000], reviews[10000:11000], reviews[11000:14000]
    #
    #     def test_reviews_bag(self):
    #         """Classify sentiment using bag-of-words"""
    #         train, dev, test = self.split_review_corpus(BagOfWords)
    #         classifier = MaxEnt()
    #         classifier.train(train, dev)
    #         self.assertGreater(accuracy(classifier, test), 0.55)

    # if __name__ == '__main__':
    #     # Run all of the tests, print the results, and exit.
    #     main(verbosity=2)


def split_names_corpus(document_class=Name):
    """Split the names corpus into training, dev, and test sets"""
    names = NamesCorpus(document_class=document_class)
    total_len = len(names)
    print "Total number of names: ", total_len
    seed(hash("names"))
    shuffle(names)
    return names[:5000], names[5000:6000], names[6000:]


def test_names_nltk():
    """Classify names using NLTK features.
       Document_class == Name"""
    train, dev, test = split_names_corpus()

    # train = [BagOfWords("a1 b1 c1", 'Y'),
    #          BagOfWords("a1 b1 c1", 'X'),
    #          BagOfWords("a1 b1 c0", 'Y'),
    #          BagOfWords("a0 b1 c1", 'X'),
    #          BagOfWords("a0 b1 c1", 'Y'),
    #          BagOfWords("a0 b0 c1", 'Y'),
    #          BagOfWords("a0 b1 c0", 'X'),
    #          BagOfWords("a0 b0 c0", 'X'),
    #          BagOfWords("a0 b1 c1", 'Y')]
    #
    # test = [BagOfWords("a1 b0 c1"),
    #         BagOfWords("a1 b0 c0"),
    #         BagOfWords("a0 b1 c1"),
    #         BagOfWords("a0 b1 c0")]
    #
    # dev = [BagOfWords("a1 b0 c1", 'Y'),
    #        BagOfWords("a1 b0 c0", 'Y'),
    #        BagOfWords("a0 b1 c1", 'Y'),
    #        BagOfWords("a0 b1 c0", 'X')]

    classifier = MaxEnt()
    classifier.train(train, dev)
    other_metrics(classifier, test)

    # dictionaries = [classifier.classify(x, return_distribution=True) for x in test]
    # for dict in dictionaries:
    #     print dict


test_names_nltk()
