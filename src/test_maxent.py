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
    print "Total number of names: 5001 + 2943 = ", total_len
    seed(hash("names"))
    shuffle(names)
    return names[:5000], names[5000:6000], names[6000:]


def test_names_nltk():
    """Classify names using NLTK features.
       Document_class == Name"""
    train, dev, test = split_names_corpus()
    classifier = MaxEnt()
    classifier.train(train, dev)
    acc = accuracy(classifier, test)
    print acc

test_names_nltk()
