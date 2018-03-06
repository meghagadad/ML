import os
import math
from collections import Counter

negativeReviews = []
positiveReviews = []
negativeWords = []
positiveWords = []
stopWords = ['is', 'a', 'of', 'the', 'it', 'and', 'for', 'with', 'be', 'in',
             'this', 'an', 'to', 'has', 'that', 'she', 'he', 'it', 'i', 'was',
             'as', 'are', 'on', 'his', 'her', 'at', 'have']
global combinedWords
global UniqueWords
global PositiveAmount
global NegativeAmount
global PProb  # The probability that the class is POSITIVE
global NProb  # The probability that the class is NEGATIVE


#  The function below collects all words of the parameter type POSITIVE or NEGATIVE and puts them into a list.

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
    global NegativeAmount
    NegativeAmount = len(negativeWords)
    global UniqueWords
    UniqueWords = len(set(combinedWords))
    global NProb
    NProb = len(negativeReviews) / len(negativeReviews + positiveReviews)
    global PProb
    PProb = len(positiveReviews) / len(negativeReviews + positiveReviews)


#  This function prints a list of top word frequencies of the parameter list.
#  We probably do not need this function.
            #

            # def all_word_frequency(listOfWords, top):
            #     listOfWords = Counter(listOfWords)
            #     listOfWords = listOfWords.most_common(top)
            #     return listOfWords


# This function finds the frequency of a specific word among all words of the parameter type.

def word_frequency(word, type):
    count = 0
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
    elif type == "negative":
        amount = NegativeAmount
        ClassProb = NProb
    result = 1
    with open('test/'+ filename, 'r', encoding='utf-8') as file:
        for line in file:
            for word in line.split():
                if word.lower() not in stopWords:
                    result *= math.log((word_frequency(word, type) + 1) / ((amount + UniqueWords)))
    return result * math.log(ClassProb)


# This function assigns a class to a review based on the greatest of the two probabilities: positive and negative.

def max_prob(filename):
    positive = class_probabilities(filename,"positive")
    negative = class_probabilities(filename,"negative")
    decision = ""
    if positive > negative:
        print('The review is POSITIVE. The probability was:', positive)
        decision = "positive"
    elif positive < negative:
        print('The review is NEGATIVE. The probability was:', negative)
        decision = "negative"
    else:
        print('You are extremely unlucky! The probabilities were exactly the same!')
    if test_accuracy(filename, decision) == True:
        print("The prediction was correct!")
    else: print("The prediction was wrong!")

# This function tests all reviews in the test folder.

def test_all_reviews():
    counter = 0
    for filename in os.listdir('test/'):
        try:
            if filename.endswith(".txt"):
                    counter += 1
                    max_prob(filename)
        except Exception as e:
            raise e
            print("No files found!")
    print('In total', counter, 'reviews were tested')


def test_accuracy(filename, decision):
    if ((filename.endswith("_10.txt") or filename.endswith("_9.txt") or filename.endswith("_8.txt") or filename.endswith("_7.txt")) and (decision == "positive")):
        return True
    elif ((filename.endswith("_1.txt") or filename.endswith("_2.txt") or filename.endswith("_3.txt") or filename.endswith("_4.txt")) and (decision == "negative")):
        return True
    else: return False

#
#  CODE TESTING AREA
#

collect_words('positive')
collect_words('negative')
initialize()
print("Total number of positive words:", PositiveAmount)
print("Total number of negative words:", NegativeAmount)
print("Total number of UNIQUE words:", UniqueWords)
print("Probability of class Positive:", PProb)
print("Probability of class Negative:", NProb)
print("The word awful appears", word_frequency("awful", "positive"), "times in positive reviews")
print("The word awful appears", word_frequency("awful", "negative"), "times in negative reviews")
test_all_reviews()





