'''
Julian Silerio
jjs2245

'''

from math import log
import sys, string

def extract_words(text):
    words = text.lower()
    for c in string.punctuation:
        words = words.replace(c,'')
    words.replace('\n','')
    return words.split(' ')
    # should return a list of words in the data sample.

def get_data(training_filename):
    data = []
    print("Reading in {}".format(training_filename))
    with open(training_filename) as file:
        for line in file.readlines():
            label, text = line.split('\t')
            data.append((label, text))
    print("Finished reading {}".format(training_filename))
    return data



class NbClassifier(object):

    def __init__(self, training_filename, stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}
        self.word_given_label = {}
        self.data = get_data(training_filename)

        self.collect_attribute_types(training_filename, 1)
        self.train(training_filename)

    def collect_attribute_types(self, training_filename, k):
        counter = {}

        print("Collecting attribute types for {}".format(training_filename))
        for datum in self.data:
            words = extract_words(datum[1])
            for word in words:
                if word in counter.keys():
                    counter[word] += 1
                else:
                    counter[word] = 1

        for key in counter.keys():
            if counter[key] >= k:
                self.attribute_types.add(key)
        print("Finish attribute types")

    def train(self, training_filename):
        self.label_prior = {}
        self.word_given_label = {}

        count_label = {}
        count_word_given_label = {}

        count_label_wgl = {}

        print("Begin training on {}".format(training_filename))
        # parse data for count(labels) and count(words|label)
        for datum in self.data:
            label, text = datum
            if label in count_label:
                count_label[label] += 1
            else:
                count_label[label] = 1

        # count all words given label
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
        print(count_label)
        total_labels = sum(count_label.values())
        print(total_labels)
        # value of c for laplacian smoothing
        c = 1

        # build word_given_label for actual label
        for tuple in count_word_given_label.keys():
            word, label = tuple
            #self.word_given_label[tuple] = count_word_given_label[tuple]/count_label_wgl[label]

            # build wgl for other label if not present
            # also gonna build label_prior in here since iterating through count_label already
            for c_label in count_label.keys():
                self.label_prior[c_label] = count_label[c_label]/float(total_labels)

                count_word = 0

                if (word, c_label) in self.word_given_label.keys():
                    count_word = count_word_given_label[(word, c_label)]

                self.word_given_label[(word, c_label)] = (count_word + c)/(count_label_wgl[c_label]+(c*len(self.attribute_types)))

        print(self.label_prior)
        print("Finish training")

    # given the probabilities predict the likelihood of being a given label
    def predict(self, text):
        words = extract_words(text)
        label_prob = {}

        for label in self.label_prior.keys():
            label_prob[label] = log(self.label_prior[label])

            for word in words:
                if word in self.attribute_types:
                    label_prob[label] += log(self.word_given_label[(word, label)])
        print(label_prob)
        return label_prob

    def evaluate(self, test_filename):
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        accuracy = 0.0

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        print("Evaluating {}".format(test_filename))
        with open(test_filename) as file:
            for line in file.readlines():
                label, text = line.split('\t')
                label_candidates = self.predict(text)
                #print(label_candidates)
                label_predict = max(label_candidates,key=label_candidates.get)
                #print("predicted:{}\nactual:{}\n".format(label_predict,label))
                if label_predict == label:
                    if label == 'spam':
                        tp += 1
                    else:
                        tn += 1
                else:
                    if label == 'spam':
                        fn += 1
                    else:
                        fp += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = (2*tp)/(2*tp+fp+fn)
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        print("Finish evaluation\n")
        print("TP:{} TN:{} FP:{} FN:{}".format(tp, tn, fp, fn))
        return precision, recall, fscore, accuracy



def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":

    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
