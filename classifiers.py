from numpy import savetxt
from numpy import loadtxt
from numpy import load
from numpy import save
from numpy import savez_compressed
import re, os
import numpy
import csv
from numpy import array, dot
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
import pandas as pd
import hazm
import math
class Classifier:
    __directory_trained_models = 'trained_models/'
    output_csv_fieldnames = ['iterations', 'feature_set', 'learning_rate', 'type', 'set_split', 'matches', 'mismatches',
                  'true_positives', 'true_negatives', 'true_neutral',
                  'predicted_positives', 'predicted_negatives', 'predicted_neutral',
                  'actual_positives', 'actual_negatives', 'actual_neutral', 'accuracy', 'error_rate',
                  'precision_positives', 'precision_negatives', 'precision_neutral',
                  'recall_positives', 'recall_negatives', 'recall_neutral',
                  'average_positives', 'average_negatives', 'average_neutral',
                  'f_score_positives', 'f_score_negatives', 'f_score_neutral']
    feature_set_names = {1: 'UNIGRAMS', 2: 'BIGRAMS', 3: '3-GRAMS', 4: '4-GRAMS', 5: '5-GRAMS', 6: '6-GRAMS'}
    initial_data = []
    all_feature_sets = []
    selected_feature_set_names = ''
    selected_split_name = ''
    hash_dictionary = {}
    orientations = ['+', '-', '=']
    history = {'accuracy': 0, 'patience': 0, 'errors': math.inf}
    table_symbol_orientation = {'p': '+', 'n': '-', 0: '=', '0': '='}
    patience = 10
    normalizer = {}
    stopwords = {}
    regex_words = {}
    def __init__(self, feature_set, orientations=None, language='english'):
        self.language = language
        self.normalizer[language] = hazm.Normalizer()
        if language == 'persian':
            self.stopwords[language] = hazm.stopwords_list()
            self.regex_words[language] = r"[\w']+|[.,!?;،؟؛]"
        else:
            self.stopwords[language] = set(stopwords.words('english'))
            self.regex_words[language] = r"[\w']+|[.,!?;]"

        if orientations:
            self.orientations = orientations

        self.feature_set = feature_set
        self.weights = {}
        self.hash_dictionary[self.language] = {}

    def save_model_csv(self, data, filename):
        savetxt(self.__directory_trained_models + '{}-{}.csv'.format(filename,self.language), data, delimiter=',')

    def load_model_csv(self, filename):
        self.weights[self.language] = loadtxt(self.__directory_trained_models + '{}.csv'.format(filename), delimiter=',')
        return self.weights[self.language]

    def save_model_npy(self, data, filename):
        '''
            Sometimes we have a lot of data in NumPy arrays that we wish to save efficiently, but which we only need to use in another Python program.
            Therefore, we can save the NumPy arrays into a native binary format that is efficient to both save and load.
        '''
        save(self.__directory_trained_models + '{}-{}.npy'.format(filename, self.language), data)

    def load_model_npy(self, filename):
        self.weights[self.language] = load(self.__directory_trained_models + '{}.npy'.format(filename))
        return self.weights[self.language]
    def save_model_npz(self, data, filename):
        '''
            Sometimes, we prepare data for modeling that needs to be reused across multiple experiments, but the data is large.
            This might be pre-processed NumPy arrays like a corpus of text (integers) or a collection of rescaled image data (pixels). In these cases, it is desirable to both save the data to file, but also in a compressed format.
            This allows gigabytes of data to be reduced to hundreds of megabytes and allows easy transmission to other servers of cloud computing for long algorithm runs.
        '''
        savez_compressed(self.__directory_trained_models + '{}-{}.npz'.format(filename,self.language), data)

    def load_model_npz(self, filename):
        dict_data = load(self.__directory_trained_models + '{}.npz'.format(filename))
        # a dict with the names ‘arr_0’ for the first array, ‘arr_1’ for the second, and so on.
        # data = dict_data['arr_0']
        self.weights[self.language] = dict_data['arr_0']
        return dict_data
    def save_word_dictionary(self):
        df = pd.DataFrame.from_dict(self.hash_dictionary[self.language], orient="index")
        df.to_csv(self.__directory_trained_models + 'word_dictionary_{}.csv'.format(self.language))

    def load_word_dictionary(self):
        df = pd.read_csv(self.__directory_trained_models + 'word_dictionary_{}.csv'.format(self.language), index_col=0)
        self.hash_dictionary[self.language] = df.to_dict("split")
        self.hash_dictionary[self.language] = dict(zip(self.hash_dictionary[self.language]["index"], self.hash_dictionary[self.language]["data"]))

    def read_text_file(self, directory, orientation=None):
        with open(directory, 'r', encoding="ISO-8859-1") as file:
            if orientation:
                for item in file:
                    self.initial_data.append([item.replace(' \n', '').replace('\n', ''), orientation])
            else:
                for item in file:
                    lines = item.split(',')
                    self.initial_data.append([lines[0].replace(' \n', '').replace('\n', ''), lines[1]])


    def read_excel_file(self, directory, columns, orientation=None):
        log_file = open('log_read_file.txt', 'w')
        all_excel_files = [x for x in os.listdir(directory) if '.xlsx' in x]
        for file in all_excel_files:
            data = pd.read_excel(directory + file)
            df = pd.DataFrame(data, columns=columns) #['comment', 'orientation']
            if orientation:
                for i in range(len(df[columns[0]])):
                    self.initial_data.append([
                        self.normalizer[self.language].normalize(
                            df[columns[0]][i].replace('_x005F', '').replace('\r', '').replace('\n', '').replace('\u200e', ' ')), orientation])
            else:
                for i in range(len(df[columns[0]])):
                    try:
                        self.initial_data.append([
                            self.normalizer[self.language].normalize(
                                df[columns[0]][i].replace('_x005F', '').replace('\r', '').replace('\n', '').replace('\u200e', ' ')),
                        self.table_symbol_orientation[df[columns[1]][i]]])
                    except KeyError as e:
                        print(e, file, i+2, df[columns[0]][i], df[columns[1]][i])
                        log_file.write(str(e) + ' ' + file + ' ' +  str(i+2) + '\n')
                    except AttributeError as e:
                        print(e, file, i+2, df[columns[0]][i], df[columns[1]][i])
                        log_file.write(str(e) + ' ' + file + ' ' +  str(i+2) + '\n')


    def apply_feature_set(self, feature_set=None):
        if feature_set:
            self.feature_set = feature_set
        self.selected_feature_set_names = ''
        for feat in self.feature_set:
            self.selected_feature_set_names += self.feature_set_names[feat]
        for sentence, orientation in self.initial_data:
            _words = re.findall(self.regex_words[self.language], sentence.rstrip())
            words = []
            for w in _words:
                if w.lower() not in self.stopwords[self.language]:
                    words.append(w)
            xt = []
            for feature in self.feature_set:
                n_grams = ngrams(words, feature)
                for word in n_grams:
                    xt.append(word)

            if len(xt) > 0:
                if orientation == '+':
                    yt = 1
                elif orientation == '-':
                    yt = -1
                else:
                    yt = 0
                self.all_feature_sets.append((xt, yt))

    def train_test_split(self, test_size, random_state=0, shuffle=True):
        train, test = train_test_split(self.all_feature_sets, test_size=test_size, random_state=random_state,
                                       shuffle=shuffle)
        self.selected_split_name = 'test_size-{}_random_state-{}_{}'.format(test_size,random_state, 'shuffle' if shuffle else '')
        return [(train, test)]

    def k_fold_split(self, n_splits=2, random_state=0, shuffle=True):
        all = []
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        for train, test in kf.split(self.all_feature_sets):
            all.append((train, test))
        self.selected_split_name = '$$k_fold-{}_random_state-{}_{}'.format(n_splits,random_state, 'shuffle' if shuffle else '')
        return all

    def repeated_k_fold_split(self, n_splits=2, n_repeats=2, random_state=12883823):
        all = []
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for train, test in rkf.split(self.all_feature_sets):
            all.append((train, test))
        self.selected_split_name = '$$repeated_k_fold-{}-{}_random_state-{}'.format(n_splits,n_repeats,random_state)

        return all

    def create_dictionary(self, xy_train):
        index_word = 0
        for words, orientation in xy_train:
            for word in words:
                if not word in self.hash_dictionary[self.language]:
                    self.hash_dictionary[self.language][word] = index_word
                    index_word += 1

    def set_patience(self, patience):
        self.patience = patience

    def train(self, xy_train, xy_test, classifier='perceptron', list_iteration=[10, 100, 1000, 5000], learning_rate=1):
        for iter in range(max(list_iteration) + 1):
            # TODO: create thread for experiment function to run per iter to calc accuracy and check with patience if not improve then break loop
            # self.experiment(xy_test)

            if iter in list_iteration:
                self.save_outcome(xy_train, xy_test, classifier,
                                  {'iterations': iter, 'feature_set': self.selected_feature_set_names,
                                   'learning_rate': learning_rate})
            if classifier == 'perceptron':
                errors = self.perceptron(xy_train, learning_rate)
            elif classifier == 'winnow':
                errors = self.winnow(xy_train, learning_rate)
            elif classifier == 'passive_aggressive':
                errors = self.passive_aggressive_binary(xy_train, learning_rate)
            else:
                return 0
            print('{} Iteration # {}\tErrors= {}'.format(classifier, iter, errors))
            if errors == 0:
                self.save_outcome(xy_train, xy_test, classifier,
                                  {'iterations': iter, 'feature_set': self.selected_feature_set_names,
                                   'learning_rate': learning_rate})
                break
            if self.history['patience'] == self.patience:
                self.save_outcome(xy_train, xy_test, classifier,
                                  {'iterations': iter, 'feature_set': self.selected_feature_set_names,
                                   'learning_rate': learning_rate})
                break

            if errors >= self.history['errors']:
                self.history['patience'] += 1
            else:
                self.history['errors'] = errors


    def create_weights_array(self, value=1):
        self.weights[self.language] = numpy.array(array([value] * len(self.hash_dictionary[self.language])), dtype=numpy.float64)
        self.history = {'accuracy': 0, 'patience': 0, 'errors': math.inf}

    def predict(self, xt, wt=None):
        # Initiate the feature vector
        if wt is None:
            wt = array([0] * len(self.weights[self.language]))
        wt *= 0

        # Calculate the dot product (learned label)
        for word in xt:
            if word in self.hash_dictionary[self.language]:
                wt[self.hash_dictionary[self.language][word]] = 1

        wx = dot(self.weights[self.language], wt)
        y = numpy.sign(wx)

        # if y <= 0:
        #     y = -1
        # else:
        #     y = 1
        #
        # if (0.5 > y > -0.5) and len(self.orientations) > 2:
        #     y = 0
        return y

    def perceptron(self, xy_train, learning_rate=1.0):
        errors = 0
        wt = array([0] * len(self.weights[self.language]))
        for xt, yt in xy_train:
            wt *= 0

            for word in xt:
                if word in self.hash_dictionary[self.language]:
                    wt[self.hash_dictionary[self.language][word]] = 1

            wx = dot(self.weights[self.language], wt)

            if wx <= 0:
                y = -1
            else:
                y = 1

            if (0.5 > wx > -0.5) and len(self.orientations) > 2:
                y = 0

            # Set the respective values to the feature vector
            error = yt - y
            # If error then update weight vector
            if error != 0:
                errors += 1
                self.weights[self.language] += learning_rate * error * wt
        return errors

    def winnow(self, xy_train, learning_rate=1):
        errors = 0
        theta = len(self.hash_dictionary[self.language])
        wt = array([0] * len(self.weights[self.language]))

        # For each entry t in the training data set
        for xt, yt in xy_train:
            wt *= 0

            for word in xt:
                if word in self.hash_dictionary[self.language]:
                    wt[self.hash_dictionary[self.language][word]] = 1

            # Calculate the dot product
            wx = dot(self.weights[self.language], wt)

            # TODO: for neutral should be set
            if  wx < theta and yt == 0 and len(self.orientations) > 2:
                errors += 1
                self.weights[self.language] += 0.5 * wt * learning_rate
            # If error then update weight vector
            elif wx < theta and yt == 1:
                errors += 1
                self.weights[self.language] += 2.0 * wt * learning_rate

            elif wx > theta and yt == -1:
                errors += 1
                self.weights[self.language] += 0.5 * wt * learning_rate
        return errors

    def PA(self, loss, xn):
        return loss / xn

    def PA1(self, learning_rate, loss, xn):
        return min([learning_rate, loss / xn])

    def PA2(self, learning_rate, loss, xn):
        return loss / (xn + 1 / (2 * learning_rate))

    def passive_aggressive_binary(self, xy_train, learning_rate=1):
        errors = 0
        wt = array([0] * len(self.weights[self.language]))

        for xt, yt in xy_train:
            wt *= 0
            for word in xt:
                if word in self.hash_dictionary[self.language]:
                    wt[self.hash_dictionary[self.language][word]] = 1

            yh = yt * dot(self.weights[self.language], wt)
            if yh < 1:
                loss = 1 - yh
                xn = numpy.sum(wt ** 2)
                tau = self.PA(loss, xn)
                self.weights[self.language] = self.weights[self.language] + yt * tau * wt
                errors += 1

        return errors

    def experiment(self, xy):
        # Keep count of matches and mismatches
        matches = 0
        mismatches = 0
        true_positives = 0
        true_negatives = 0
        true_neutral = 0

        predicted_positives = 0
        predicted_negatives = 0
        predicted_neutral = 0
        actual_set = {1: 0, 0: 0, -1: 0}
        wt = array([0] * len(self.weights[self.language]))

        # For each entry t in the testing data set
        for xt, yt in xy:
            actual_set[yt] += 1
            # Calculate the dot product
            y = self.predict(xt, wt)
            if y == 1:
                predicted_positives += 1
            elif y == 0:
                predicted_neutral += 1
            elif y == -1:
                predicted_negatives += 1

            # Update the respective counter
            if y == yt:
                matches += 1
                if y == 1:
                    true_positives += 1
                elif y == -1:
                    true_negatives += 1
                else:
                    true_neutral += 1
            else:
                mismatches += 1

        accuracy = float(true_positives + true_negatives + true_neutral) / float(
            predicted_positives + predicted_negatives + predicted_neutral)
        error_rate = 1 - accuracy

        if accuracy <= self.history['accuracy']:
            # self.history['patience'] +=1
            pass
        else:
            self.history['accuracy'] = accuracy

        precision_positives = float(true_positives) / float(predicted_positives) if predicted_positives > 0 else 0
        recall_positives = float(true_positives) / float(
            true_positives + predicted_negatives - true_negatives + predicted_neutral - true_neutral)
        average_positives = float(precision_positives + recall_positives) / 2.0
        f_score_positives = (2.0 * precision_positives * recall_positives) / (precision_positives + recall_positives) if (precision_positives + recall_positives) > 0 else 0

        precision_negatives = float(true_negatives) / float(predicted_negatives) if predicted_negatives > 0 else 0
        recall_negatives = float(true_negatives) / float(
            true_negatives + predicted_positives - true_positives + predicted_neutral - true_neutral)
        average_negatives = float(precision_negatives + recall_negatives) / 2.0
        f_score_negatives = (2.0 * precision_negatives * recall_negatives) / float(
            precision_negatives + recall_negatives) if (precision_negatives + recall_negatives) > 0 else 0
        if len(self.orientations) > 2:
            precision_neutral = float(true_neutral) / float(predicted_neutral) if predicted_neutral > 0 else 0
            recall_neutral = float(true_neutral) / float(
                true_neutral + predicted_positives - true_positives + predicted_negatives - true_negatives)
            average_neutral = float(precision_neutral + recall_neutral) / 2.0
            f_score_neutral = (2.0 * precision_neutral * recall_neutral) / float(precision_neutral + recall_neutral) if (precision_neutral + recall_neutral) > 0 else 0
        else:
            precision_neutral=0
            recall_neutral=0
            average_neutral=0
            f_score_neutral=0
        outcomes = {'matches': matches,
                    'mismatches': mismatches,
                    'true_positives': true_positives,
                    'true_negatives': true_negatives,
                    'true_neutral': true_neutral,
                    'predicted_positives': predicted_positives,
                    'predicted_negatives': predicted_negatives,
                    'predicted_neutral': predicted_neutral,
                    'actual_positives': actual_set[1],
                    'actual_negatives': actual_set[-1],
                    'actual_neutral': actual_set[0],
                    'accuracy': accuracy,
                    'error_rate': error_rate,
                    'precision_positives': precision_positives,
                    'precision_negatives': precision_negatives,
                    'precision_neutral': precision_neutral,
                    'recall_positives': recall_positives,
                    'recall_negatives': recall_negatives,
                    'recall_neutral': recall_neutral,
                    'average_positives': average_positives,
                    'average_negatives': average_negatives,
                    'average_neutral': average_neutral,
                    'f_score_positives': f_score_positives,
                    'f_score_negatives': f_score_negatives,
                    'f_score_neutral': f_score_neutral}
        return outcomes

    def save_outcome(self, xy_train, xy_test, classifier, init_settings):
        self.save_model_npy(self.weights[self.language], '{}-{}-{}-{}_{}'
                            .format(classifier, init_settings['iterations'],
                                    init_settings['feature_set'], init_settings['learning_rate'],
                                    self.selected_split_name))
        print("+-------------------------------+")
        print("| PERFORMANCE FOR TRAINING SET  |")
        print("+-------------------------------+")
        row = {'type':'train', 'set_split': self.selected_split_name}
        row.update(init_settings)
        outcomes = self.experiment(xy_train)
        row.update(outcomes)
        self.write_output_csv(row, 'new_outcomes/{}-{}.csv'.format(classifier,self.language))
        print(row)
        print("+------------------------------+")
        print("| PERFORMANCE FOR TESTING SET  |")
        print("+------------------------------+")
        row = {'type':'test', 'set_split': self.selected_split_name}
        row.update(init_settings)
        outcomes = self.experiment(xy_test)
        row.update(outcomes)
        self.write_output_csv(row, 'new_outcomes/{}-{}.csv'.format(classifier,self.language))
        print(row)

    def write_output_csv(self, row, file_directory):
        if os.path.isfile(file_directory):
            with open(file_directory, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.output_csv_fieldnames)
                writer.writerow(row)

        else:
            with open(file_directory, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.output_csv_fieldnames)
                writer.writeheader()
                writer.writerow(row)

    def predict_one(self, sentence, wt=None):
        _words = re.findall(self.regex_words[self.language], sentence.rstrip())
        words = []
        for w in _words:
            if w.lower() not in self.stopwords[self.language]:
                words.append(w)
        xt = []
        for feature in self.feature_set:
            n_grams = ngrams(words, feature)
            for word in n_grams:
                xt.append(str(word))

        if wt is None:
            wt = array([0] * len(self.weights[self.language]))
        wt *= 0
        for word in xt:
            if word in self.hash_dictionary[self.language]:
                wt[self.hash_dictionary[self.language][word]] = 1

        wx = dot(self.weights[self.language], wt)
        y = numpy.sign(wx)

        # if y <= 0:
        #     y = -1
        # else:
        #     y = 1
        #
        # if (0.5 > y > -0.5) and len(self.orientations) > 2:
        #     y = 0
        return y

