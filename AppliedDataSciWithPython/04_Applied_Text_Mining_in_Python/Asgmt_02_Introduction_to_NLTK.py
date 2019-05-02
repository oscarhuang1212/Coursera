# File name:   Asgmt_02_Introduction_to_NLTK.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course4: Applied Text Mining in Python
#               Week2:  Basic Natural Language Processing


"""
Assignment 2 - Introduction to NLTK

    In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create 
        a spelling recommender function that uses nltk to find words similar to the misspelling.
"""


#Part 1 - Analyzing Moby Dick

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


"""
Question 1

    What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
    This function should return a float.
"""

def answer_one():
    
    unique_tokens = len(set(nltk.word_tokenize(moby_raw)))
    tokens = len(nltk.word_tokenize(moby_raw))
    return unique_tokens/tokens



"""
Question 2

    What percentage of tokens is 'whale'or 'Whale'?
    This function should return a float.
"""

def answer_two():
    dist = nltk.probability.FreqDist(text1)
        
    return (dist['whale']+dist['Whale'])/len(text1)*100
    

"""
Question 3

    What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

    This function should return a list of 20 tuples where each tuple is of the form (token, frequency). 
        The list should be sorted in descending order of frequency.
"""

def answer_three():
    tokens =nltk.word_tokenize(moby_raw)
    dist = nltk.probability.FreqDist(tokens)
    
    return dist.most_common(20) 

"""
Question 4

    What tokens have a length of greater than 5 and frequency of more than 150?
    This function should return a sorted list of the tokens that match the above constraints. To sort your list, use sorted()
"""

def answer_four():
    tokens =nltk.word_tokenize(moby_raw)
    dist = nltk.probability.FreqDist(tokens)
    freqwords = [w for w in dist if len(w)>5 and dist[w]>150]
    
    return sorted(freqwords)


"""
Question 5

    Find the longest word in text1 and that word's length.
    This function should return a tuple (longest_word, length).

"""

def answer_five():
    # if len(word) > 15]
    long_words = [(word,len(word)) for word in text1]

    return sorted(long_words, key=lambda tup: tup[1])[-1]


"""
Question 6

    What unique words have a frequency of more than 2000? What is their frequency?
    This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.
"""

def answer_six():
    dist = nltk.probability.FreqDist(text1)
    freqwords = [(dist[w],w) for w in dist if dist[w]>2000 and w.isalpha()]
    
    return sorted(freqwords,reverse = True)

"""
Question 7

    What is the average number of tokens per sentence?
    This function should return a float.
"""
def answer_seven():
    
    sents = nltk.sent_tokenize(moby_raw)
    sent_lens = [len(nltk.word_tokenize(s)) for s in sents]    
    
    return  sum(sent_lens)/len(sents)
    
"""
Question 8

    What are the 5 most frequent parts of speech in this text? What is their frequency?
    This function should return a list of tuples of the form (part_of_speech, frequency) sorted in descending order of frequency.
"""

def answer_eight():
    pos = nltk.pos_tag(text1)
    dist = nltk.probability.FreqDist([tup[1] for tup in pos]) 
    freqpos = [(pos,dist[pos]) for pos in dist]
    
    return sorted(freqpos, key = lambda tup: tup[1], reverse = True)[:5]



"""
Part 2 - Spelling Recommender

    For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words 
        and recommends a correctly spelled word for every word in the list.

    For every misspelled word, the recommender should find find the word in correct_spellings that has the shortest distance*, 
        and starts with the same letter as the misspelled word, and return that word as a recommendation.

    *Each of the three different recommenders will use a different distance measure (outlined below).

    Each of the recommenders should provide recommendations for the three default words provided: ['cormulent', 'incendenece', 'validrate'].

"""

from nltk.corpus import words

correct_spellings = words.words()



"""
Question 9

    For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

            Jaccard distance on the trigrams of the two words.

    This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk import ngrams
    from nltk.metrics import jaccard_distance 

    answer=[]
    for enter in entries:
        gen = ((x for x in correct_spellings if x[0].lower() == enter[0].lower() ))
        max_jac=100
        max_jac_char=''
        for correct in gen:
            jac_dist = jaccard_distance(set(ngrams(enter,3)),set(ngrams(correct,3)))
            if jac_dist < max_jac:
                max_jac = jac_dist
                max_jac_char = correct

        answer.append(max_jac_char)
            
    return answer
    

"""
Question 10

    For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

            Jaccard distance on the 4-grams of the two words.

    This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk import ngrams
    from nltk.metrics import jaccard_distance 
    
    
    answer=[]
    for enter in entries:
        gen = ((x for x in correct_spellings if x[0].lower() == enter[0].lower() ))
        max_jac=100
        max_jac_char=''
        for correct in gen:
            jac_dist = jaccard_distance(set(ngrams(enter,4)),set(ngrams(correct,4)))
           
            if jac_dist < max_jac:
                max_jac = jac_dist
                max_jac_char = correct

        answer.append(max_jac_char)
                
    return answer

"""
Question 11

    For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:

            Edit distance on the two words with transpositions.

    This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].
"""

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics import edit_distance 
        
    answer=[]
    for enter in entries:
        gen = ((x for x in correct_spellings if x[0].lower() == enter[0].lower() ))
        min_edit=100
        min_edit_char=''
        for correct in gen:
            Edit_dist = edit_distance(enter,correct)
            
            if Edit_dist < min_edit:
                min_edit = Edit_dist
                min_edit_char = correct

        answer.append(min_edit_char)
            
    return answer
    
