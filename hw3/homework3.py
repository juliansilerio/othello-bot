'''
Julian Silerio
jjs2245

'''

from math import log
import sys, string

def extract_words(text):
    return text.lower().translate('', '', string.punctuation).split(' ')
    # should return a list of words in the data sample.


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}
        self.word_given_label = {}


        self.collect_attribute_types(training_filename, 1)
        self.train(training_filename)

    def collect_attribute_types(self, training_filename, k):
        counter = {}
        data = get_data(training_filename)

        for datum in data:
            words = datum[1]
            for word in words:
                if tuple() in counter.keys():
                    counter[word] += 1
                else:
                    counter[word] = 1

        for key in counter.keys():
            if counter[key] >= k:
                attribute_types.add(key)

    def train(self, training_filename):
        self.label_prior = {}
        self.word_given_label = {}

        data = get_data(training_filename)
        count_label = {}
        count_word_given_label = {}

        count_label_wgl = {}

        # parse data for count(labels) and count(words|label)
        for datum in data:
            label, text = datum
            if label in label_prior.keys():
                count_label[label] = 1
            else:
                count_label[label] += 1

        # count words given label
            words = extract_words(text)
            for word in words:
                if word in self.attribute_types:

                    # wgl counts
                    if label in count_label_wgl.keys():
                        count_label_wgl[label] += 1
                    else:
                        count_label_wgl[label] = 1

                    # label counts
                    if word in count_word_given_label.keys():
                        count_word_given_label[(word, label)] += 1
                    else:
                        count_word_given_label[(word, label)] = 1

        total_labels = sum(label_count.values())

        # value of c for laplacian smoothing
        c = 1

        # build word_given_label for actual label
        for tuple in count_word_given_label.keys():
            word, label = tuple
             self.word_given_label[tuple] = count_word_given_label[tuple]/count_label_wgl[label]

            # build wgl for other label if not present
            # also gonna build label_prior in here since smoothing already
            for c_label in count_label.keys():
                self.label_prior[c_label] = count_label[c_label]/total_labels

                if (word, c_label) not in word_given_label.keys():
                    self.word_given_label[(word, c_label)] = (count_word_given_label[(word, c_label)]+c)/(count_label_wgl[c_label]+(c*len(self.attribute_types)))

    def predict(self, text):
        words = extract_words(text)
        label_prob = {}

        for label in self.label_prior.keys():
            label_prob[label] = log(self.label_prior[label])
            for word in words:
                label_prob[label] += log(self.word_given_label((word, label)))

        return label_prob

    def evaluate(self, test_filename):
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        accuracy = 0.0

        with open(test_filename) as file:


        return precision, recall, fscore, accuracy

    def get_data(training_filename):
        data = []
        with open(training_filename) as file:
            for line in file.read():
                label, text = line.split('\t')
                data.append(label, text)
        return data



def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":

    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
