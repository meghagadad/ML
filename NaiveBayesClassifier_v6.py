import os
from timeit import default_timer as timer
import math
from collections import Counter

start = timer()

negativeReviews = []
positiveReviews = []
negativeWords = []
positiveWords = []
stopWords = ['is', 'a', 'of', 'the', 'it', 'and', 'for', 'with', 'be', 'in',
             'this', 'an', 'to', 'has', 'that', 'she', 'he', 'it', 'i', 'was',
             'as', 'are', 'on', 'his', 'her', 'at', 'have']

# The following variables are used to calculate the probability of a review being of a certain class.
# These field are initialized in the initialize() function.

global combinedWords # The total amount of words in all scanned reviews.
global UniqueWords # The total amount of UNIQUE words in all scanned reviews.
global PositiveAmount # The amount of positive reviews processed.
global NegativeAmount # The amount of negative reviews processed.
global positiveFrequencies
global negativeFrequencies
global PProb  # The probability that the class is POSITIVE
global NProb  # The probability that the class is NEGATIVE


#  The function below collects all words of the parameter type POSITIVE or NEGATIVE
#  and puts them into a list depending on the class.

def collect_words(type):
    if type == "positive":
        reviews = positiveReviews
        wordlist = positiveWords
    if type == "negative":
        reviews = negativeReviews
        wordlist = negativeWords
    counter = 0
    for filename in os.listdir(type+'/'):
        try:
            if filename.endswith(".txt"):
                reviews.append(str(filename))
                counter += 1
                with open(type+'/' + filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        for word in line.split():
                            if word.lower() not in stopWords:
                                wordlist.append(word)
        except Exception as e:
            raise e
            print("No files found!")
    print('In total', counter, type, 'reviews were scanned')
    return wordlist

# This function sets the numbers for how many positive/negative and total unique words there are.
# It also finds the probability for each class.
# These numbers are used when calculating the probability of the class of review in the function class_probabilities


def initialize():
    global combinedWords
    combinedWords = positiveWords + negativeWords
    global PositiveAmount
    PositiveAmount = len(positiveWords)
    PositiveAmount = PositiveAmount / 10000
    global NegativeAmount
    NegativeAmount = len(negativeWords)
    NegativeAmount = NegativeAmount / 10000
    global UniqueWords
    UniqueWords = len(set(combinedWords))
    UniqueWords = UniqueWords / 10000
    global NProb
    NProb = len(negativeReviews) / len(negativeReviews + positiveReviews)
    global PProb
    PProb = len(positiveReviews) / len(negativeReviews + positiveReviews)
    all_word_frequencies()


#  This function prints a list of top word frequencies of the parameter list.
#  We probably do not need this function.

def all_word_frequencies():
    global positiveFrequencies
    positiveFrequencies = Counter(positiveWords)
    global negativeFrequencies
    negativeFrequencies = Counter(negativeWords)


# This function finds the frequency of a specific word among all words of the parameter type.

def word_frequency(word, type):
    if type == "positive":
        count = positiveWords.count(word)
    elif type == "negative":
        count = negativeWords.count(word)
    return count


# This function will calculate the probability that a review is positive.
# Right now it prints out all the probabilities for each of the words in the review.

def class_probabilities(filename, type):
    if type == "positive":
        amount = PositiveAmount
        ClassProb = PProb
        frequency = positiveFrequencies
    elif type == "negative":
        amount = NegativeAmount
        ClassProb = NProb
        frequency = negativeFrequencies
    result = 1
    with open('test/' + filename, 'r', encoding='utf-8') as file:
        for line in file:
            for word in line.split():
                if word.lower() not in stopWords:
                    result *= ((frequency[word]) + 1) / (amount + UniqueWords)
    return result * ClassProb


# This function assigns a class to a review based on the greatest of the two probabilities: positive and negative.

def max_prob(filename):
    positive = class_probabilities(filename,"positive")
    negative = class_probabilities(filename,"negative")
    if positive > negative:
        # print('This review is POSITIVE. The probability was:', positive)
        decision = "positive"
    elif positive < negative:
        # print('This review is NEGATIVE. The probability was:', negative)
        decision = "negative"
    else:
        print('You are extremely unlucky! The probabilities were exactly the same!.. Or more likely, the number was too small for python to handle. '
              'Positive probability:',positive, 'Negative Probability:',negative
              , "This review will be ignored.")
        return 0
    if not (positive == negative):
        if test_accuracy(filename, decision):
            print("The prediction was correct!"+"\n")
            return 1
        elif not test_accuracy(filename, decision):
            print("The prediction was wrong!"+"\n")
            return -1

#  This function tests if the class-prediction assigned corresponds to the actual class of the review.
#  It does so my comparing the prediction to the numbers at the end of the text files.


def test_accuracy(filename, decision):
    if ((filename.endswith("_10.txt") or filename.endswith("_9.txt")
         or filename.endswith("_8.txt") or filename.endswith("_7.txt")) and (decision == "positive")):
        return True
    elif ((filename.endswith("_1.txt") or filename.endswith("_2.txt")
           or filename.endswith("_3.txt") or filename.endswith("_4.txt")) and (decision == "negative")):
        return True
    else: return False


# This function tests all reviews in the test folder.
# It also keeps track of how many predictions were correct,and finds the overall accuracy percentage.

def test_all_reviews():
    right = 0
    wrong = 0
    discarded = 0
    for filename in os.listdir('test/'):
        try:
            if filename.endswith(".txt"):
                    print("Now scanning:", filename, "Number in queue:", right+wrong)
                    result = max_prob(filename)
                    if result == 1:
                        right += 1
                    elif result == -1:
                        wrong += 1
                    else:
                        discarded += 1
                    # print("So far", right+wrong, "reviews have been scanned.")
                    # print("Accuracy so far:",(int((right/(right + wrong))*100)),"percent.")
                    print("\n")
        except Exception as e:
            raise e
            print("No files found!")
    print(right + wrong + discarded, "reviews were scanned in total.")
    print("Of these", discarded, "reviews were discarded because of probabilities were inf or 0.0")
    print('This means', right+wrong, 'reviews were actually tested')
    print(right, "of them were predicted correctly.")
    print(wrong, "of them were predicted incorrectly.")
    print("Total Accuracy was:", (int((right/(right + wrong))*100)),'percent.')


# CODE TESTING AREA

print("\n")
collect_words('positive')
collect_words('negative')

initialize()

print("\n")
test_all_reviews()

end = timer()
print("Total Time elapsed:", int(end - start), "seconds.")



