from classifiers import Classifier
from langdetect import detect
languages = {'en': 'english', 'fa': 'persian', 'ar': 'persian'}

textfa = '''عالیه!!'''

texten = '''though excessively tiresome , the uncertainty principle , as verbally pretentious as the title may be , has its handful of redeeming features , as long as you discount its ability to bore .'''
lang = detect(textfa)

print(lang)


classifier = Classifier([1], ['+', '-', '='], languages[lang])
# first read dictionary and weights from cache if not read from file
# classifier.hash_dictionary[languages[lang]] = cache[lang]

classifier.load_word_dictionary()
classifier.load_model_npy('perceptron-100-UNIGRAMS-0.4-persian')
# print(classifier.hash_dictionary)
print(classifier.predict_one(textfa))


lang = detect(texten)

print(lang)

classifier = Classifier([1], ['+', '-'], languages[lang])
# first read dictionary and weights from cache if not read from file
# classifier.hash_dictionary[languages[lang]] = cache[lang]
classifier.load_word_dictionary()
classifier.load_model_npy('perceptron-128-UNIGRAMS-0.6_test_size-0.01_random_state-0_shuffle-english')
# print(classifier.hash_dictionary)
print(classifier.predict_one(texten))
