# File name:   Asgmt_04_Document_Similarity_and_Topic_Modelling.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course4: Applied Text Mining in Python
#               Week4:   Topic Modeling

"""
Part 1 - Document Similarity

    For the first part of this assignment, you will complete the functions doc_to_synsets and similarity_score which will be used by 
        document_path_similarity to find the path similarity between two documents.

    The following functions are provided:

            convert_tag: converts the tag given by nltk.pos_tag to a tag used by wordnet.synsets. You will need to use this function in doc_to_synsets.
            document_path_similarity: computes the symmetrical path similarity between two documents by finding the synsets in each document 
                using doc_to_synsets, then computing similarities using similarity_score.

    You will need to finish writing the following functions:

            doc_to_synsets: returns a list of synsets in document. This function should first tokenize and part of speech tag the document 
                using nltk.word_tokenize and nltk.pos_tag. Then it should find each tokens corresponding synset using wn.synsets(token, wordnet_tag). 
                The first synset match should be used. If there is no match, that token is skipped.
            similarity_score: returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). 
                For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together 
                and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which 
                should be floats. Missing values should be ignored.

    Once doc_to_synsets and similarity_score have been completed, submit to the autograder which will run test_document_path_similarity to test 
        that these functions are running correctly.

    Do not modify the functions convert_tag, document_path_similarity, and test_document_path_similarity.
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

#Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets
def convert_tag(tag):
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


"""
Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc. Then finds the first synset for each word/tag combination. 
        If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted
    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
"""
def doc_to_synsets(doc):      
    str01='Fish are nvqjp friends'
    str_token = nltk.word_tokenize(doc)
    str_token_pos = nltk.pos_tag(str_token)
    str_wn = [wn.synsets(token,convert_tag(tag))[0] for (token,tag) in str_token_pos if len(wn.synsets(token,convert_tag(tag)))>0]
        
    return str_wn



"""
Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
"""
def similarity_score(s1, s2):
    
    synsets1 = doc_to_synsets('I like cats')
    synsets2 = doc_to_synsets('I like dogs')
    Sum_of_sim=0
    Count = 0
    
    for sset1 in s1:
        Largest_sim=0
        for sset2 in s2:
            if sset1.path_similarity(sset2) != None:
                if sset1.path_similarity(sset2)>Largest_sim:
                    Largest_sim = sset1.path_similarity(sset2)
        if Largest_sim !=0:
            Sum_of_sim = Sum_of_sim + Largest_sim
            Count = Count +1
        
    return Sum_of_sim/Count


#Finds the symmetrical similarity between doc1 and doc2
def document_path_similarity(doc1, doc2):
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


"""
paraphrases is a DataFrame which contains the following columns: Quality, D1, and D2.
    
    Quality is an indicator variable which indicates if the two documents D1 and D2 are paraphrases of one another 
        (1 for paraphrase, 0 for not paraphrase).
"""
paraphrases = pd.read_csv('paraphrases.csv')

"""
most_similar_docs

    Using document_path_similarity, find the pair of documents in paraphrases which has the maximum similarity score.
    This function should return a tuple (D1, D2, similarity_score)
"""
def most_similar_docs():
    
    paraphrases['Sim_score'] = paraphrases.apply(lambda row: document_path_similarity(row['D1'],row['D2']),axis=1)
    
    sorted_index = np.argsort(-paraphrases.Sim_score)
    Max_sim_row = paraphrases.iloc[sorted_index[0]]
    
    return (Max_sim_row.D1,Max_sim_row.D2,Max_sim_row.Sim_score)

"""
label_accuracy

    Provide labels for the twenty pairs of documents by computing the similarity for each pair using document_path_similarity. 
        Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). 
        Report accuracy of the classifier using scikit-learn's accuracy_score.
    This function should return a float.
"""

def label_accuracy():
    from sklearn.metrics import accuracy_score
    
    paraphrases['Sim_score'] = paraphrases.apply(lambda row: document_path_similarity(row['D1'],row['D2']),axis=1)
    paraphrases['label'] = paraphrases.apply(lambda x: 1 if x['Sim_score']>0.75 else 0, axis = 1)
    
    return accuracy_score(paraphrases.label,paraphrases.Quality)



"""
Part 2 - Topic Modelling

    For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in newsgroup_data. 
    You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters 
        on the corpus, and save to the variable ldamodel. Extract 10 topics using corpus and id_map, and with passes=25 and random_state=34.
"""

import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`
ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=id_map, num_topics=10, passes=25, random_state=34)


"""
lda_topics

    Using ldamodel, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as 
        a list of 10 tuples where each tuple takes on the form:

            (9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" 
            + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')

    for example.

    This function should return a list of tuples.
"""

def lda_topics():

    return ldamodel.print_topics(10)
