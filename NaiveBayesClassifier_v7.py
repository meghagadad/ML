import os
from timeit import default_timer as timer
import math
from collections import Counter

# This timer will count how many seconds it takes to execute the entire program. It ends at the end of the document.
start = timer()

negativeReviews = [] # a list containing names of all the text-files of negative training reviews.
positiveReviews = [] # a list containing names of all the text-files of positive training reviews.
negativeWords = []  # a list containing all words in all negative training reviews.
positiveWords = []  # a list containing all words in all positive training reviews.
stopWords = ['is', 'a', 'of', 'the', 'it', 'and', 'for', 'with', 'be', 'in',
             'this', 'an', 'to', 'has', 'that', 'she', 'he', 'it', 'i', 'was',
             'as', 'are', 'on', 'his', 'her', 'at', 'have']

# stopWords are words that will be ignored when collecting words for training pruposes.
# These are typically very commonly-used words.


# The following variables are used to calculate the probability of a review being of a certain class.
# These field are initialized in the initialize() function.

global combinedWords # The total amount of words in all scanned reviews.
global UniqueWords # The total amount of UNIQUE words in all scanned reviews.
global PositiveAmount # The amount of positive reviews processed.
global NegativeAmount # The amount of negative reviews processed.
global positiveFrequencies # a dictionary containing the pairs: (positive word, frequency of this word)
global negativeFrequencies # a dictionary containing the pairs: (negative word, frequency of this word)
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
# Finally it finds the frequencies of all positive and negative words and stores them in dictionaries.
# These numbers are used when calculating the probability of the class of review in the function class_probabilities
# Some of the variables are being divided by 10000 to avoid having the probabilities being either too small or too big.
# This division does not seem to affect the accuracy at all, which is good.
# It is just meant to avoid the inf or 0.0 probability results.

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
    global positiveFrequencies
    positiveFrequencies = Counter(positiveWords)
    global negativeFrequencies
    negativeFrequencies = Counter(negativeWords)


# This function is responsible for calculating the actual probability that will be used
# to determine which class it will be predicted to be.


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
# It also uses the test_accuracy function to find out if our prediction were correct or not.

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
        print("The probabilities were exactly the same!. "
              "This is likely because the probabilities ended up too small for Python to handle. "
              "This review will be discarded from test results.")
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
# It also keeps track of how many predictions were correct, and it finds the overall accuracy percentage of our test.

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
    print("Of these,", discarded, "reviews were discarded because of class probabilities were inf or 0.0")
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



