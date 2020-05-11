from classifiers import Classifier
classifier = Classifier([1], ['+', '-'], 'english')
# data = numpy.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])


directory_of_file = 'polarityData/rt-polaritydata/rt-polarity-neg.txt'
classifier.read_text_file(directory_of_file, '-')
directory_of_file = 'polarityData/rt-polaritydata/rt-polarity-pos.txt'
classifier.read_text_file(directory_of_file, '+')
classifier.apply_feature_set()  # [1,2], [3]
sets = classifier.train_test_split(0.01)
# sets = [(classifier.all_feature_sets, sets[0][1])] # for extract model

classifier.set_patience(300)

# perceptron
k=1
for train, test in sets:
    if '$' in classifier.selected_split_name:
        classifier.selected_split_name = str(k)+classifier.selected_split_name[1:]
        k+=1
    classifier.create_dictionary(train)
    classifier.save_word_dictionary()
    # print(classifier.hash_dictionary)
    for lr in [0.6]: # [0.1, 0.2, 0.25, 0.4, 0.6, 0.75, 0.8, 1]
        classifier.create_weights_array(1)
        print('{} \tLearning Rate= {}\tSplit Set: {}'.format('perceptron', lr, classifier.selected_split_name))
        classifier.train(train, test, classifier='perceptron',
                         list_iteration=[10, 100,1000], # [10, 100, 1000, 5000]
                         learning_rate=lr)
        print('history: {}'.format(classifier.history))


# # winnow
# k=1
# for train, test in sets:
#     if '$' in classifier.selected_split_name:
#         classifier.selected_split_name = str(k)+classifier.selected_split_name[1:]
#         k+=1
#     classifier.create_dictionary(train)
#     for lr in [0.1, 0.2, 0.25, 0.4, 0.6, 0.75, 0.8, 1]:
#         classifier.create_weights_array(1)
#         print('{} \tLearning Rate= {}\tSplit Set: {}'.format('winnow', lr, classifier.selected_split_name))
#         classifier.train(train, test, classifier='winnow', list_iteration=[10, 100, 1000, 5000], learning_rate=lr)
#         print('history: {}'.format(classifier.history))
#
#
# # passive_aggressive
# k=1
# for train, test in sets:
#     if '$' in classifier.selected_split_name:
#         classifier.selected_split_name = str(k)+classifier.selected_split_name[1:]
#         k+=1
#     classifier.create_dictionary(train)
#     for lr in [0.1, 0.2, 0.25, 0.4, 0.6, 0.75, 0.8, 1]:
#         classifier.create_weights_array(0)
#         print('{} \tLearning Rate= {}\tSplit Set: {}'.format('passive_aggressive', lr, classifier.selected_split_name))
#         classifier.train(train, test, classifier='passive_aggressive', list_iteration=[10, 100, 1000, 5000], learning_rate=lr)
#         print('history: {}'.format(classifier.history))