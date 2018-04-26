import re, os
import math
import numpy
import csv
from numpy import array, dot
from nltk.corpus import stopwords
from nltk.util import ngrams



POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')

fieldnames = ['Max Iterations', 'Feature Set','Learning Rate', 'Type Set', 'MATCHES', 'MISMATCHES', 'TRUE POSITIVES', 'TRUE NEGATIVES',
              'PREDICTED POSITIVES', 'PREDICTED NEGATIVES', 'ACTUAL POSITIVES', 'ACTUAL NEGATIVES', 'ACCURACY',
              'PRECISION (POS)', 'RECALL (POS)', 'AVERAGE (POS)', 'F-SCORE (POS)', 'PRECISION (NEG)', 'RECALL (NEG)',
              'AVERAGE (NEG)', 'F-SCORE (NEG)']

csvfile = open("winnow_output_table.csv", 'w')
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
row = {}
indexRow = 0
dictionary = {}
def read_file():
    _posSentences = []
    _negSentences = []

    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:

            _posSentences.append([i,'+'])
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
          _negSentences.append([i, '-'])
    return _posSentences,_negSentences


def buildSet(sentences, featureSet, useDictionary=1):

    # Initialize the dictionary and the data
    wordCount = 0

    # The set to be built
    localSet = []
    nSet = 0
    pSet = 0


    # For each line in the file
    for line in sentences:

        # Get the sentence from the line and remove grammatical symbols
        sentence = line[0]
        _words = re.findall(r"[\w']+|[.,!?;]", sentence.rstrip())

        stops = set(stopwords.words('english'))
        words = []
        for w in _words:
            if w.lower() not in stops:
                words.append(w)

        # Save the label of the entry (1 for +, 0 for -)
        yt = 1 if line[1] == '+' else 0
        if yt == 1:
            pSet += 1

        # Initialize the feature set
        xt = []

        # Build unigrams
        if featureSet == 1:

            # For each valid word in the sentence
            for word in words:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)

        # Build bigrams
        elif featureSet == 2:
            _ngrams = ngrams(words, featureSet)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
        # Build 3-grams
        elif featureSet == 3:
            _ngrams = ngrams(words, featureSet)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)

        # Build 4-grams
        elif featureSet == 4:
            _ngrams = ngrams(words, featureSet)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
        # Build 5-grams
        elif featureSet == 5:
            _ngrams = ngrams(words, featureSet)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
        # Build 6-grams
        elif featureSet == 6:
            _ngrams = ngrams(words, featureSet)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)

        # Build unigrams-bigrams-3456-grams
        else:

            # Add unigrams first
            for word in words:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)

            # Now add bigrams
            _ngrams = ngrams(words, 2)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
            _ngrams = ngrams(words, 3)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
            _ngrams = ngrams(words, 4)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
            _ngrams = ngrams(words, 5)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
            _ngrams = ngrams(words, 6)
            for word in _ngrams:

                # Add word to dictionary in case it's not there
                if useDictionary == 1 and not word in dictionary:
                    dictionary[word] = wordCount
                    wordCount += 1

                # Add word to xt
                xt.append(word)
        # Add tuple to the set
        if len(xt) > 0:
            localSet.append((xt, yt))
            nSet += 1

# Return the generated set and the counters


    return localSet, nSet, pSet


def predict_one(weights, input_snippet):
    # Initiate the feature vector
    wt = array([0] * len(weights))
    wt *= 0

    # Calculate the dot product (learned label)
    for word in input_snippet:
        if word in dictionary:
            wt[dictionary[word]] = 1

    y = dot(weights, wt)
    y = 0 if y <= 0 else 1
    return y


def winnow(set, weights, theta, learningRate=1):
    # Initialize error count
    errors = 0

    # Initiate the feature vector
    wt = array([0] * len(weights))

    # For each entry t in the training data set
    for xt, yt in set:

        # Convert label to the winnow specification
        yt = -1 if yt == 0 else 1

        # Set the respective values to the feature vector
        wt *= 0
        for word in xt:
            wt[dictionary[word]] = 1

        # Calculate the dot product
        wx = dot(weights, wt)

        # If error then update weight vector
        if wx < theta and yt == 1:
            errors += 1
            weights += 2.0 * wt *learningRate

        if wx > theta and yt == -1:
            errors += 1
            weights += 0.5 * wt * learningRate

    # Returns the results
    return [weights, errors]


def experiment(set, weights):
    # Keep count of matches and mismatches
    matches = 0
    mismatches = 0
    truePositives = 0
    trueNegatives = 0
    predictedPositives = 0
    predictedNegatives = 0


    # For each entry t in the testing data set
    for xt, yt in set[0]:

        # Calculate the dot product
        y = predict_one(weights, xt)
        if y == 1:
            predictedPositives += 1
        elif y == 0:
            predictedNegatives += 1
        # Update the respective counter
        if y == yt:
            matches += 1
            if y == 1:
                truePositives += 1
            else:
                trueNegatives += 1
        else:
            mismatches += 1

    # Calculate metrics
    accuracy = (float(truePositives) / float(set[2])) * (float(trueNegatives) / float((set[1] - set[2])))
    precisionPos = float(truePositives) / float(predictedPositives)
    recallPos = float(truePositives) / float(set[2])
    averagePos = (precisionPos + recallPos) / 2.0
    fScorePos = (2.0 * precisionPos * recallPos) / (precisionPos + recallPos)

    precisionNeg = float(trueNegatives) / float(predictedNegatives) if predictedNegatives > 0 else 0
    recallNeg = float(trueNegatives) / float((set[1] - set[2]))
    averageNeg = (precisionNeg + recallNeg) / 2.0
    fScoreNeg = (2.0 * precisionNeg * recallNeg) / (precisionNeg + recallNeg) if (precisionNeg + recallNeg) > 0 else 0

    # Print results
    print ("MATCHES: ", matches)
    print ("MISMATCHES: ", mismatches)
    print ("TRUE POSITIVES: ", truePositives)
    print ("TRUE NEGATIVES: ", trueNegatives)
    print ("PREDICTED POSITIVES: ", predictedPositives)
    print ("PREDICTED NEGATIVES: ", predictedNegatives)
    print ("ACTUAL POSITIVES: ", set[2])
    print ("ACTUAL NEGATIVES: ", set[1] - set[2])
    print ("ACCURACY: ", accuracy)
    print ("PRECISION (POS): ", precisionPos)
    print ("RECALL (POS): ", recallPos)
    print ("AVERAGE (POS): ", averagePos)
    print ("F-SCORE (POS): ", fScorePos)
    print ("PRECISION (NEG): ", precisionNeg)
    print ("RECALL (NEG): ", recallNeg)
    print ("AVERAGE (NEG): ", averageNeg)
    print ("F-SCORE (NEG): ", fScoreNeg)
    print ("")

    global indexRow
    tIndexRow = indexRow
    tIndexRow +=1
    tRow = row.copy()
    tRow[fieldnames[tIndexRow]] = matches
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = mismatches
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = truePositives
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = trueNegatives
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = predictedPositives
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = predictedNegatives
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = set[2]
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = set[1] - set[2]
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = accuracy
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = precisionPos
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = recallPos
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = averagePos
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = fScorePos
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = precisionNeg
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = recallNeg
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = averageNeg
    tIndexRow += 1
    tRow[fieldnames[tIndexRow]] = fScoreNeg
    writer.writerow(tRow)


def main(trainSentences, testSentences, featureSet, maxIterations, learningRate=1):
    global indexRow
    print("Max iterations: " + str(maxIterations))
    row[fieldnames[indexRow]] = maxIterations
    indexRow +=1
    if(featureSet == 1):
        print("Feature set: UNIGRAMS")
        row[fieldnames[indexRow]] = "UNIGRAMS"
        indexRow += 1
    elif(featureSet == 2):
        print("Feature set: BIGRAMS")
        row[fieldnames[indexRow]] = "BIGRAMS"
        indexRow += 1
    elif(featureSet == 3 or featureSet == 4 or featureSet == 5 or featureSet == 6):
        print("Feature set: " + str(featureSet) + "-GRAMS")
        row[fieldnames[indexRow]] = str(featureSet) + "-GRAMS"
        indexRow += 1
    else:
        print("Feature set: UNIGRAMS + BIGRAMS + 3-GRAMS + 4-GRAMS + 5-GRAMS + 6-GRAMS")
        row[fieldnames[indexRow]] = "1to6-GRAMS"
        indexRow += 1
    print("Learning rate: " + str(learningRate))
    row[fieldnames[indexRow]] = learningRate
    indexRow += 1
    print("")

    training = buildSet(trainSentences, featureSet)
    testing = buildSet(testSentences, featureSet, 0)

    nWords = len(dictionary)
    weights = numpy.array(array([1] * nWords), dtype=numpy.float64)

    for iteration in range(0, maxIterations):
        weights, errors = winnow(training[0], weights, nWords, learningRate)
        print("Winnow  Iteration # ", iteration, "\tErrors = ", errors)
        # Stop iterating if no mistakes are found
        if errors == 0: break

    print("")
    # Show performance results for the training set
    print ("+-------------------------------+")
    print ("| PERFORMANCE FOR TRAINING SET  |")
    print ("+-------------------------------+")
    print ("")
    row[fieldnames[indexRow]] = "Training"
    experiment(training, weights)

    # Show performance results for the testing set
    print ("+------------------------------+")
    print ("| PERFORMANCE FOR TESTING SET  |")
    print ("+------------------------------+")
    print ("")
    row[fieldnames[indexRow]] = "Test"
    experiment(testing, weights)

def setArg(trainSentences, testSentences, grams, iteration , learningRate):
    global indexRow
    main(trainSentences, testSentences, grams, iteration, learningRate)
    dictionary.clear()
    row.clear()
    indexRow = 0
print ("Algorithm: WINNOW")

_Sentences= read_file()

posCutoff = int(math.floor(len(_Sentences[0]) * 3 / 4))
negCutoff = int(math.floor(len(_Sentences[1]) * 3 / 4))
trainSentences = _Sentences[0][:posCutoff] + _Sentences[1][:negCutoff]
testSentences = _Sentences[0][posCutoff:] + _Sentences[1][negCutoff:]

setArg(trainSentences,testSentences,1,10,1)
setArg(trainSentences,testSentences,1,100,1)
setArg(trainSentences,testSentences,1,1000,1)
setArg(trainSentences,testSentences,1,5000,1)

setArg(trainSentences,testSentences,1,10,0.75)
setArg(trainSentences,testSentences,1,100,0.75)
setArg(trainSentences,testSentences,1,1000,0.75)
setArg(trainSentences,testSentences,1,5000,0.75)

setArg(trainSentences,testSentences,1,10,0.5)
setArg(trainSentences,testSentences,1,100,0.5)
setArg(trainSentences,testSentences,1,1000,0.5)
setArg(trainSentences,testSentences,1,5000,0.5)

setArg(trainSentences,testSentences,1,10,0.25)
setArg(trainSentences,testSentences,1,100,0.25)
setArg(trainSentences,testSentences,1,1000,0.25)
setArg(trainSentences,testSentences,1,5000,0.25)


setArg(trainSentences,testSentences,2,10,1)
setArg(trainSentences,testSentences,2,100,1)
setArg(trainSentences,testSentences,2,1000,1)
setArg(trainSentences,testSentences,2,5000,1)

setArg(trainSentences,testSentences,2,10,0.75)
setArg(trainSentences,testSentences,2,100,0.75)
setArg(trainSentences,testSentences,2,1000,0.75)
setArg(trainSentences,testSentences,2,5000,0.75)

setArg(trainSentences,testSentences,2,10,0.5)
setArg(trainSentences,testSentences,2,100,0.5)
setArg(trainSentences,testSentences,2,1000,0.5)
setArg(trainSentences,testSentences,2,5000,0.5)

setArg(trainSentences,testSentences,2,10,0.25)
setArg(trainSentences,testSentences,2,100,0.25)
setArg(trainSentences,testSentences,2,1000,0.25)
setArg(trainSentences,testSentences,2,5000,0.25)


setArg(trainSentences,testSentences,3,10,1)
setArg(trainSentences,testSentences,3,100,1)
setArg(trainSentences,testSentences,3,1000,1)
setArg(trainSentences,testSentences,3,5000,1)

setArg(trainSentences,testSentences,3,10,0.75)
setArg(trainSentences,testSentences,3,100,0.75)
setArg(trainSentences,testSentences,3,1000,0.75)
setArg(trainSentences,testSentences,3,5000,0.75)

setArg(trainSentences,testSentences,3,10,0.5)
setArg(trainSentences,testSentences,3,100,0.5)
setArg(trainSentences,testSentences,3,1000,0.5)
setArg(trainSentences,testSentences,3,5000,0.5)

setArg(trainSentences,testSentences,3,10,0.25)
setArg(trainSentences,testSentences,3,100,0.25)
setArg(trainSentences,testSentences,3,1000,0.25)
setArg(trainSentences,testSentences,3,5000,0.25)


setArg(trainSentences,testSentences,4,10,1)
setArg(trainSentences,testSentences,4,100,1)
setArg(trainSentences,testSentences,4,1000,1)
setArg(trainSentences,testSentences,4,5000,1)

setArg(trainSentences,testSentences,4,10,0.75)
setArg(trainSentences,testSentences,4,100,0.75)
setArg(trainSentences,testSentences,4,1000,0.75)
setArg(trainSentences,testSentences,4,5000,0.75)

setArg(trainSentences,testSentences,4,10,0.5)
setArg(trainSentences,testSentences,4,100,0.5)
setArg(trainSentences,testSentences,4,1000,0.5)
setArg(trainSentences,testSentences,4,5000,0.5)

setArg(trainSentences,testSentences,4,10,0.25)
setArg(trainSentences,testSentences,4,100,0.25)
setArg(trainSentences,testSentences,4,1000,0.25)
setArg(trainSentences,testSentences,4,5000,0.25)


setArg(trainSentences,testSentences,5,10,1)
setArg(trainSentences,testSentences,5,100,1)
setArg(trainSentences,testSentences,5,1000,1)
setArg(trainSentences,testSentences,5,5000,1)

setArg(trainSentences,testSentences,5,10,0.75)
setArg(trainSentences,testSentences,5,100,0.75)
setArg(trainSentences,testSentences,5,1000,0.75)
setArg(trainSentences,testSentences,5,5000,0.75)

setArg(trainSentences,testSentences,5,10,0.5)
setArg(trainSentences,testSentences,5,100,0.5)
setArg(trainSentences,testSentences,5,1000,0.5)
setArg(trainSentences,testSentences,5,5000,0.5)

setArg(trainSentences,testSentences,5,10,0.25)
setArg(trainSentences,testSentences,5,100,0.25)
setArg(trainSentences,testSentences,5,1000,0.25)
setArg(trainSentences,testSentences,5,5000,0.25)


setArg(trainSentences,testSentences,6,10,1)
setArg(trainSentences,testSentences,6,100,1)
setArg(trainSentences,testSentences,6,1000,1)
setArg(trainSentences,testSentences,6,5000,1)

setArg(trainSentences,testSentences,6,10,0.75)
setArg(trainSentences,testSentences,6,100,0.75)
setArg(trainSentences,testSentences,6,1000,0.75)
setArg(trainSentences,testSentences,6,5000,0.75)

setArg(trainSentences,testSentences,6,10,0.5)
setArg(trainSentences,testSentences,6,100,0.5)
setArg(trainSentences,testSentences,6,1000,0.5)
setArg(trainSentences,testSentences,6,5000,0.5)

setArg(trainSentences,testSentences,6,10,0.25)
setArg(trainSentences,testSentences,6,100,0.25)
setArg(trainSentences,testSentences,6,1000,0.25)
setArg(trainSentences,testSentences,6,5000,0.25)


setArg(trainSentences,testSentences,7,10,1)
setArg(trainSentences,testSentences,7,100,1)
setArg(trainSentences,testSentences,7,1000,1)
setArg(trainSentences,testSentences,7,5000,1)

setArg(trainSentences,testSentences,7,10,0.75)
setArg(trainSentences,testSentences,7,100,0.75)
setArg(trainSentences,testSentences,7,1000,0.75)
setArg(trainSentences,testSentences,7,5000,0.75)

setArg(trainSentences,testSentences,7,10,0.5)
setArg(trainSentences,testSentences,7,100,0.5)
setArg(trainSentences,testSentences,7,1000,0.5)
setArg(trainSentences,testSentences,7,5000,0.5)

setArg(trainSentences,testSentences,7,10,0.25)
setArg(trainSentences,testSentences,7,100,0.25)
setArg(trainSentences,testSentences,7,1000,0.25)
setArg(trainSentences,testSentences,7,5000,0.25)
