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
global PProb    #The probabilty that the class is Positive
global NProb    #The probabilty that the class is Negative

#  The Function below collects all positive words in all reviews into one list.

def positive_reviews(path):
    counter = 0
    for filename in os.listdir(path):
            try:
                if filename.endswith(".txt"):
                    positiveReviews.append(str(filename))
                    counter += 1
                    with open(path+filename, 'r', encoding='utf-8') as f:
                        for line in f:
                            for word in line.split():
                                if word.lower() not in stopWords:
                                    positiveWords.append(word)
            except Exception as e:
                raise e
                print("No files found!")
    print('In total', counter, 'positive reviews were scanned')
    return positiveWords

#  The Function below collects all negative words in all reviews into one list.

def negative_reviews(path):
    counter = 0
    for filename in os.listdir(path):
        try:
            if filename.endswith(".txt"):
                negativeReviews.append(str(filename))
                counter += 1
                with open(path + filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        for word in line.split():
                            if word.lower() not in stopWords:
                                negativeWords.append(word)
        except Exception as e:
            raise e
            print("No files found!")
    print('In total', counter, 'negative reviews were scanned')
    return negativeWords


#  This function prints a list of top word frequencies of the parameter list. We probably do not need this function. 

def word_frequency(listOfWords, top):
    listOfWords = Counter(listOfWords)
    listOfWords = listOfWords.most_common(top)
    return listOfWords


# This function sets the numbers for how many positive/negative and total unique words there are.
# It also finds the probability for each class.

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
    NProb = len(negativeReviews)/len(negativeReviews + positiveReviews)
    global PProb
    PProb = len(positiveReviews)/len(negativeReviews + positiveReviews)


# This function finds the frequency of a specific word among all positive words.

def word_positive(word):
   count = positiveWords.count(word)
   return count


# This function finds the frequency of a specific word among all negative words.

def word_negative(word):
   count = negativeWords.count(word)
   return count


# This function will calculate the probability that a review is positive.
# DOES NOT  WORK YET, IT ALWAYS RETURNS 0.0 when attempting to return the combined probability.
# Right now it prints out all the probabilities for each of the words in the review.

def test_if_positive():
    result = 1
    with open("test/5_0.txt", 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if word.lower() not in stopWords:
                    result *= ((word_positive(word)+1)/(PositiveAmount+UniqueWords))
    return result * PProb

def test_if_negative():
    result = 1
    with open("test/5_0.txt", 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if word.lower() not in stopWords:
                    result *= ((word_negative(word)+1)/(NegativeAmount+UniqueWords))
    return result * NProb

# fUNCTION RETURN THE MAXIMUM NUMBER OF PROBABILITY in order to clasify  to positive or negative class 
def maxProb():
    if (test_if_positive() > test_if_negative()):
        print('The review is positive.')
    elif (test_if_positive() < test_if_negative()):
        print('The review is negative.')
    else:
        print('You are extremally unlucky! It can be negative or positive equally.')
    return max(test_if_positive(),test_if_negative())


    
    
    

#  CODE TESTING AREA

positive_reviews('positive/')
# print(word_frequency(positiveWords, 1000))
negative_reviews('negative/')
# print(word_frequency(negativeWords, 1000))
initialize()
print("Total number of positive words:", PositiveAmount)
print("Total number of negative words:", NegativeAmount)
print("Total number of UNIQUE words:", UniqueWords)
print("Probability of class Positive:", PProb)
print("Probability of class Negative:", NProb)
print("The word awful appears", word_positive("awful"), "times in positive reviews")
print("The word awful appears", word_negative("awful"), "times in positive reviews")
print('Positive words probability:', test_if_positive())
print('Negative words probability:',test_if_negative())
print("Max",maxProb())








