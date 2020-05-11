from collections import defaultdict
import re
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import operator
from collections import Counter
import math
sentences = []

with open("test_data") as posSentences:
    for i in posSentences:
        sentences.append(i)
n_docs = len(sentences)
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
def preprocess(s, lowercase=False):
    tokens = word_tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
count_stop_single = Counter()
for text in sentences:
    terms_stop = [term for term in preprocess(text) if term not in stop]
    count_stop_single.update(terms_stop)

com = defaultdict(lambda: defaultdict(int))

# f is the file pointer to the JSON data set
for line in sentences:
    terms_only = [term for term in preprocess(line)
                  if term not in stop
                  and not term.startswith(('#', '@'))]

    # Build co-occurrence matrix
    for i in range(len(terms_only) - 1):
        for j in range(i + 1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])
            if w1 != w2:
                com[w1][w2] += 1

# n_docs is the total n. of tweets
p_t = {}
p_t_com = defaultdict(lambda : defaultdict(int))



for term, n in count_stop_single.items():
    p_t[term] = n / n_docs
    for t2 in com[term]:
        p_t_com[term][t2] = com[term][t2] / n_docs
positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
    # shall we also include game-specific terms?
    # 'triumph', 'triumphal', 'triumphant', 'victory', etc.
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
    # 'defeat', etc.
]

pmi = defaultdict(lambda: defaultdict(int))
for t1 in p_t:
    for t2 in com[t1]:
        denom = p_t[t1] * p_t[t2]
        pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)

semantic_orientation = {}
for term, n in p_t.items():
    positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
    negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
    semantic_orientation[term] = positive_assoc - negative_assoc

semantic_sorted = sorted(semantic_orientation.items(),
                         key=operator.itemgetter(1),
                         reverse=True)
top_pos = semantic_sorted[:10]
top_neg = semantic_sorted[-10:]

print(top_pos)
print(top_neg)
# print("ITA v WAL: %f" % semantic_orientation['#itavwal'])
# print("SCO v IRE: %f" % semantic_orientation['#scovire'])
# print("ENG v FRA: %f" % semantic_orientation['#engvfra'])
# print("#ITA: %f" % semantic_orientation['#ita'])
# print("#FRA: %f" % semantic_orientation['#fra'])
# print("#SCO: %f" % semantic_orientation['#sco'])
# print("#ENG: %f" % semantic_orientation['#eng'])
# print("#WAL: %f" % semantic_orientation['#wal'])
# print("#IRE: %f" % semantic_orientation['#ire'])

print("#IRE: %f" % semantic_orientation['good'])