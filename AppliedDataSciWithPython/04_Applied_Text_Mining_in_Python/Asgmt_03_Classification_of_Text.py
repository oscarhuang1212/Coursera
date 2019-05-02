# File name:   Asgmt_03_Classification_of_Text.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course4: Applied Text Mining in Python
#               Week3:  Classification of Text


"""
Assignment 3
    In this assignment you will explore text message data and create models to predict if a message is spam or not.
"""

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')
spam_data['target'] = np.where(spam_data['target']=='spam',1,0)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], spam_data['target'], random_state=0)


"""
Question 1

    What percentage of the documents in spam_data are spam?
    This function should return a float, the percent value (i.e. ratio∗100ratio∗100).
"""

def answer_one():
    
    return spam_data.target.sum()/len(spam_data.target)*100


"""
Question 2

    Fit the training data X_train using a Count Vectorizer with default parameters. 
    What is the longest token in the vocabulary?

    This function should return a string.
"""
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vec = CountVectorizer().fit(X_train)
    return max(vec.get_feature_names(),key=len)


"""
Question 3

    Fit and transform the training data X_train using a Count Vectorizer with default parameters.
    Next, fit a fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1. Find the area under the curve (AUC) score using 
        the transformed test data.
    This function should return the AUC score as a float.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():

    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    clrMNNB = MultinomialNB(alpha=0.1).fit(X_train_vectorized,y_train)
    
    
    test_predictions=clrMNNB.predict(X_test_vectorized)
    
    
    return roc_auc_score(y_test,test_predictions)


"""
Question 4

    Fit and transform the training data X_train using a Tfidf Vectorizer with default parameters.
    What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?

    Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. 
        The index of the series should be the feature name, and the data should be the tf-idf.

    The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with 
        largest tf-idfs should be sorted largest first.

    This function should return a tuple of two series (smallest tf-idfs series, largest tf-idfs series).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    feature_names = np.array(vect.get_feature_names())
    
    
    tfid = X_train_vectorized.max(0).toarray()[0]
    sorted_tfidf_index = tfid.argsort()
    
    mys = pd.Series(tfid[sorted_tfidf_index[:]] ,index=feature_names[sorted_tfidf_index[:]])
    
    sorted_tfid = mys.iloc[np.lexsort([mys.index, mys.values])]
    
    min_tfid = pd.Series(tfid[sorted_tfidf_index[:]] ,index=feature_names[sorted_tfidf_index[:]])
    max_tfid = pd.Series(tfid[sorted_tfidf_index[-20:]] ,index=feature_names[sorted_tfidf_index[-20:]])
    
    min_tfid_sorted=min_tfid.iloc[np.lexsort([min_tfid.index, min_tfid.values])]
    max_tfid_sorted=max_tfid.iloc[np.lexsort([max_tfid.index, -max_tfid.values])]

    return (min_tfid_sorted[:20],max_tfid_sorted)

"""
Question 5

    Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 3.
    Then fit a multinomial Naive Bayes classifier model with smoothing alpha=0.1 and compute the area under the curve (AUC) score using 
        the transformed test data.
    This function should return the AUC score as a float.
"""

def answer_five():
    vect = TfidfVectorizer(min_df = 3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    NBclr = MultinomialNB(alpha=0.1).fit(X_train_vectorized,y_train)
    
    test_predict = NBclr.predict(X_test_vectorized)
    
    return roc_auc_score(y_test,test_predict)


"""
Question 6

    What is the average length of documents (number of characters) for not spam and spam documents?
        This function should return a tuple (average length not spam, average length spam).
"""

def answer_six():
    
    not_spam_len = spam_data[spam_data.target!=1].text.apply(len).mean()
    spam_len = spam_data[spam_data.target==1].text.apply(len).mean()
    
    return   (not_spam_len,spam_len)




#The following function has been provided to help you combine new features into the training data:

#Returns sparse feature matrix with added feature.
#feature_to_add can also be a list of features.
def add_feature(X, feature_to_add):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


"""
Question 7

    Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5.

    Using this document-term matrix and an additional feature, the length of document (number of characters), fit a Support Vector Classification model 
        with regularization C=10000. Then compute the area under the curve (AUC) score using the transformed test data.

    This function should return the AUC score as a float.
"""

from sklearn.svm import SVC

def answer_seven():
    vect = TfidfVectorizer(min_df = 5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    X_train_vect = add_feature(X_train_vectorized,X_train.str.len())
    X_test_vect = add_feature(X_test_vectorized,X_test.str.len())
    
    clr = SVC(C=10000,gamma='auto').fit(X_train_vect,y_train)
    test_predict = clr.predict(X_test_vect)
    
    
    return roc_auc_score(y_test,test_predict)


"""
Question 8

    What is the average number of digits per document for not spam and spam documents?
    This function should return a tuple (average # digits not spam, average # digits spam).
"""

def answer_eight():
    
    
    not_spam_digits =spam_data[spam_data.target!=1].text.str.extractall('(\d)').apply(len)/len(spam_data[spam_data.target!=1])
    spam_digits =spam_data[spam_data.target==1].text.str.extractall('(\d)').apply(len)/len(spam_data[spam_data.target==1])
    
    return (float(not_spam_digits), float(spam_digits))
    

"""
Question 9

    Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than 5 
        and using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams).

    Using this document-term matrix and the following additional features:

            the length of document (number of characters)
            number of digits per document

    fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
    This function should return the AUC score as a float.
"""

from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vect = TfidfVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    X_train_vect = add_feature(X_train_vectorized,X_train.str.len())
    X_train_vect = add_feature(X_train_vect,X_train.str.findall('\d').apply(len))
    
    X_test_vect = add_feature(X_test_vectorized,X_test.str.len())
    X_test_vect = add_feature(X_test_vect,X_test.str.findall('\d').apply(len))
    
    clr = LogisticRegression(C=100,solver = 'liblinear').fit(X_train_vect,y_train)
    test_predict = clr.predict(X_test_vect)
    
    return roc_auc_score(y_test,test_predict)



"""
Question 10

    What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
    Hint: Use \w and \W character classes

    This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).
"""

def answer_ten():
    
    not_spam_W =float(spam_data[spam_data.target!=1].text.str.extractall('(\W)').apply(len)/len(spam_data[spam_data.target!=1]))
    spam_W =float(spam_data[spam_data.target==1].text.str.extractall('(\W)').apply(len)/len(spam_data[spam_data.target==1]))
    
    return (not_spam_W,spam_W)



"""
Question 11

    Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency 
        strictly lower than 5 and using character n-grams from n=2 to n=5.

    To tell Count Vectorizer to use character n-grams pass in analyzer='char_wb' which creates character n-grams only from text inside word boundaries. 
        This should make the model more robust to spelling mistakes.

    Using this document-term matrix and the following additional features:

            the length of document (number of characters)
            number of digits per document
            number of non-word characters (anything other than a letter, digit or underscore.)

    fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.

    Also find the 10 smallest and 10 largest coefficients from the model and return them along with the AUC score in a tuple.

    The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.

    The three features that were added to the document term matrix should have the following names should they appear in 
        the list of coefficients: ['length_of_doc', 'digit_count', 'non_word_char_count']

    This function should return a tuple (AUC score as a float, smallest coefs list, largest coefs list).
"""
def answer_eleven():
    from collections import Counter
        
    vect= CountVectorizer(min_df = 5, ngram_range = (2,5),analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

        
    X_train_vect = add_feature(X_train_vectorized,X_train.str.len())
    X_train_vect = add_feature(X_train_vect,X_train.str.findall('\d').apply(len))
    X_train_vect = add_feature(X_train_vect,X_train.str.findall('\W').apply(len))
    
    
    X_test_vect = add_feature(X_test_vectorized,X_test.str.len())
    X_test_vect = add_feature(X_test_vect,X_test.str.findall('\d').apply(len))
    X_test_vect = add_feature(X_test_vect,X_test.str.findall('\W').apply(len))
    
    
    clr = LogisticRegression(C=100,solver = 'liblinear').fit(X_train_vect,y_train)
    test_predict = clr.predict(X_test_vect)
    roc_score = roc_auc_score(y_test,test_predict)
    
    feature_names = np.array(vect.get_feature_names())
    feature_names = np.append(feature_names,['length_of_doc', 'digit_count', 'non_word_char_count'])
    
    sorted_coef_index = clr.coef_[0].argsort()

    small_coef = pd.Series(clr.coef_[0][sorted_coef_index[:10]],index = feature_names[sorted_coef_index[:10]])
    large_coef = pd.Series(clr.coef_[0][sorted_coef_index[-11:-1]],index = feature_names[sorted_coef_index[-11:-1]]).sort_values(ascending=False)


    return (roc_score,small_coef,large_coef)
    

print(answer_eleven())