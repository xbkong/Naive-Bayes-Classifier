from sklearn.metrics import accuracy_score
import numpy as np


# Generate a matrix from data
def get_matrix(words, data, labels):
    mtx = []
    for i in range(len(labels)):
        mtx.append([0] * len(words))
    for i in data:
        mtx[i[0] - 1][i[1] - 1] = 1
    return mtx


def data_loader():
    print "Data loading begin"
    with open('trainData.txt', 'r') as f:
        train_data = [[int(x) for x in line.split()] for line in f]
    with open('trainLabel.txt', 'r') as f:
        train_labels = [line.split()[0] for line in f]
    with open('testData.txt', 'r') as f:
        test_data = [[int(x) for x in line.split()] for line in f]
    with open('testLabel.txt', 'r') as f:
        test_labels = [line.split()[0] for line in f]
    with open('words.txt', 'r') as f:
        words = [line.split()[0] for line in f]
    print "Data loading Done"
    return words, train_data, np.array(train_labels), test_data, test_labels


class NaiveBayesClassifier:
    def __init__(self):
        self.labels = None
        self.label_totals = None
        self.word_frequencies = None
        self.word_probabilities = None
        self.label_priors = None

    def train(self, src, gt, words):
        num_words = src.shape[1]
        # Convert gt to ndarray
        gt_mtx = []
        for i in gt:
            if i == '1':
                gt_mtx.append([0])
            else:
                gt_mtx.append([1])
        gt_mtx = np.array(gt_mtx)

        self.labels = np.array(['1', '2'])
        gt_mtx = np.concatenate((1 - gt_mtx, gt_mtx), axis=1)

        # Allocate arrays for counting labels and word frequencies
        self.label_totals = np.zeros(gt_mtx.shape[1])
        self.word_frequencies = np.zeros((gt_mtx.shape[1], num_words))

        # Count word frequencies and label totals
        self.word_frequencies = np.dot(gt_mtx.T, src)
        self.label_totals = gt_mtx.sum(axis=0)

        # Laplace smoothing by adding 1 and 2, then take log diff
        self.word_probabilities = (np.log(self.word_frequencies + 1) -
                                   np.log((self.label_totals + 2).reshape(-1, 1)))

        # Compute the 10 most discriminative words
        differences = [abs(self.word_probabilities[0][i] - self.word_probabilities[1][i])
                       for i in range(self.word_probabilities.shape[1])]
        for i in range(10):
            max_diff = max(differences)
            most_disc_word = differences.index(max_diff)
            print words[most_disc_word], max_diff
            differences[most_disc_word] = 0
        # prior prob for each label class
        self.label_priors = np.log(self.label_totals) - np.log(self.label_totals.sum())
        return self

    def infer(self, src):
        one_minus_p_ary = np.log(1 - np.exp(self.word_probabilities))
        log_likelihoods = np.dot(src, (self.word_probabilities - one_minus_p_ary).T) + \
                          self.label_priors + one_minus_p_ary.sum(axis=1)
        return self.labels[np.argmax(log_likelihoods, axis=1)]


def main():
    words, train_data, train_labels, test_data, test_labels = data_loader()
    trn_mtx = np.array(get_matrix(words, train_data, train_labels))
    tst_mtx = np.array(get_matrix(words, test_data, test_labels))

    learner = NaiveBayesClassifier()
    learner.train(trn_mtx, train_labels, words)
    train_pred = learner.infer(trn_mtx)
    test_pred = learner.infer(tst_mtx)
    print "Accuracy on TRAIN SET: ", accuracy_score(train_labels, train_pred)
    print "Accuracy on TEST SET: ", accuracy_score(test_labels, test_pred)


main()
