'''
Julian Silerio
jjs2245

'''

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

        # build label_prior
        total_labels = sum(label_count.values())
        for label in count_label.keys():
            self.label_prior[label] = count_label[label]/total_labels

        # value of c for laplacian smoothing
        c = 1

        # build word_given_label with laplacian smoothing
        for tuple in count_word_given_label.keys():
            word, label = tuple
            self.label_prior[tuple] = (count_word_given_label[tuple]+c)/(count_label_wgl[label]+(c*len(self.attribute_types)))

    def predict(self, text):
        return {} #replace this


    def evaluate(self, test_filename):
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        accuracy = 0.0
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
