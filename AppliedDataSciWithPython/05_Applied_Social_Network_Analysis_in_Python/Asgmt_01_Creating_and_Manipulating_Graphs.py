# File name:   Asgmt_01_Creating_and_Manipulating_Graphs.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course5: Applied_Social_Network_Analysis_in_Python
#               Week1:   Why Study Networks and Basics on NetworkX

"""
Assignment 1 - Creating and Manipulating Graphs

    Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night. 
        These choices are stored in the file Employee_Movie_Choices.txt.

    A second file, Employee_Relationships.txt, has data on the relationships between different coworkers.

    The relationship score has value of -100 (Enemies) to +100 (Best Friends). A value of zero means the two employees haven't interacted 
        or are indifferent.
    
    Both files are tab delimited.
"""
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    import matplotlib.pyplot as plt

    #G: a networkx G
    #weight_name: name of the attribute for plotting edge weights (if G is weighted)
 
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights)
    else:
        nx.draw_networkx(G, pos, edges=edges)
    plt.show()

"""
Question 1
    Using NetworkX, load in the bipartite graph from Employee_Movie_Choices.txt and return that graph.
    This function should return a networkx graph with 19 nodes and 24 edges
"""
def answer_one():
    
    B = nx.read_edgelist('Employee_Movie_Choices.txt', delimiter= '\t')
    
    return B


"""
Question 2

    Using the graph from the previous question, add nodes attributes named 'type' where movies have the value 'movie' 
        and employees have the value 'employee' and return that graph.
    This function should return a networkx graph with node attributes {'type': 'movie'} or {'type': 'employee'}
"""
def answer_two():
    B = answer_one()
    
    for e in employees:
        B.add_node(e,type='employee')
        
    for m in movies:
        B.add_node(m,type='movie')    
    return B 


"""
Question 3

    Find a weighted projection of the graph from answer_two which tells us how many movies different pairs of employees have in common.
    This function should return a weighted projected graph.
"""
def answer_three():
    
    return bipartite.weighted_projected_graph(answer_two(),employees)



"""
Question 4

    Suppose you'd like to find out if people that have a high relationship score also like the same types of movies.
    Find the Pearson correlation ( using DataFrame.corr() ) between employee relationship scores and the number of movies they have in common. 
        If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.
    This function should return a float.
"""
def answer_four():
    
    df_movie = nx.to_pandas_adjacency(answer_three())
    df_relationship = pd.read_csv('Employee_Relationships.txt', header = None, delimiter = '\t', names = ['name1','name2','relationship'])
        
    df_relationship['movie']=df_relationship.apply(lambda x: df_movie.loc[x.name1][x.name2], axis = 1)
    
    return  df_relationship.relationship.corr(df_relationship.movie)
