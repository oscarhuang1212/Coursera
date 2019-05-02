# File name:   Asgmt_03_Influence_Measures_and_Network_Centralization.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course5: Applied_Social_Network_Analysis_in_Python
#               Week3:   Influence Measures and Network Centralization

"""
Assignment 3
    In this assignment you will explore measures of centrality on two networks, a friendship network in Part 1, and a blog network in Part 2.
"""

"""
Part 1

    Answer questions 1-4 using the network G1, a network of friendships at a university department. Each node corresponds to a person, and an edge indicates friendship.
    The network has been loaded as networkx graph object G1.
"""

import networkx as nx

G1 = nx.read_gml('friendships.gml')

"""
Question 1

    Find the degree centrality, closeness centrality, and normalized betweeness centrality (excluding endpoints) of node 100.
    This function should return a tuple of floats (degree_centrality, closeness_centrality, betweenness_centrality).
"""

def answer_one():
    
    return  (nx.degree_centrality(G1)[100], 
             nx.closeness_centrality(G1)[100],
             nx.betweenness_centrality(G1)[100])


#For Questions 2, 3, and 4, assume that you do not know anything about the structure of the network, except for the all the centrality values of the nodes. That is, use one of the covered centrality measures to rank the nodes and find the most appropriate candidate.

"""
Question 2
    
    Suppose you are employed by an online shopping website and are tasked with selecting one user in network G1 to send an online shopping voucher to. 
        We expect that the user who receives the voucher will send it to their friends in the network. You want the voucher to reach 
        as many nodes as possible. The voucher can be forwarded to multiple users at the same time, but the travel distance of the voucher is 
        limited to one step, which means if the voucher travels more than one step in this network, it is no longer valid. Apply your knowledge 
        in network centrality to select the best candidate for the voucher.
    This function should return an integer, the name of the node.
"""

def answer_two():
    max_degree =sorted(nx.degree_centrality(G1).items(), key = lambda x:x[1],reverse = True)[0]
    
    return max_degree[0]


"""
Question 3

    Now the limit of the voucher’s travel distance has been removed. Because the network is connected, regardless of who you pick, every node 
        in the network will eventually receive the voucher. However, we now want to ensure that the voucher reaches the nodes in the 
        lowest average number of hops.
    How would you change your selection strategy? Write a function to tell us who is the best candidate in the network under this condition.
    This function should return an integer, the name of the node.
"""

def answer_three():
    max_closeness =sorted(nx.closeness_centrality(G1).items(), key = lambda x:x[1],reverse = True)[0]
    
    return max_closeness[0]


"""
Question 4

    Assume the restriction on the voucher’s travel distance is still removed, but now a competitor has developed a strategy to remove a person 
        from the network in order to disrupt the distribution of your company’s voucher. Your competitor is specifically targeting people who are 
        often bridges of information flow between other pairs of people. Identify the single riskiest person to be removed under your 
        competitor’s strategy?
    This function should return an integer, the name of the node.
"""
def answer_four():
    max_betweenness =sorted(nx.betweenness_centrality(G1).items(), key = lambda x:x[1],reverse = True)[0]
    
    return max_betweenness[0]


"""
Part 2
    G2 is a directed network of political blogs, where nodes correspond to a blog and edges correspond to links between blogs. 
        Use your knowledge of PageRank and HITS to answer Questions 5-9.
"""

G2 = nx.read_gml('blogs.gml')

"""
Question 5

    Apply the Scaled Page Rank Algorithm to this network. Find the Page Rank of node 'realclearpolitics.com' with damping value 0.85.
    This function should return a float.
"""

def answer_five():
        
    return nx.pagerank(G2,alpha=0.85)['realclearpolitics.com']


"""
Question 6

    Apply the Scaled Page Rank Algorithm to this network with damping value 0.85. Find the 5 nodes with highest Page Rank.
    This function should return a list of the top 5 blogs in desending order of Page Rank.
"""

def answer_six():
        
    sorted_SPR=sorted(nx.pagerank(G2,alpha=0.85).items(),key = lambda x:x[1],reverse = True)[:5]
    
    return  [key for key,SPR in sorted_SPR]



"""
Question 7

    Apply the HITS Algorithm to the network to find the hub and authority scores of node 'realclearpolitics.com'.
    Your result should return a tuple of floats (hub_score, authority_score).
"""


def answer_seven():
    h,a = nx.hits(G2)
    
    return (h['realclearpolitics.com'],a['realclearpolitics.com'])


"""
Question 8

    Apply the HITS Algorithm to this network to find the 5 nodes with highest hub scores.
    This function should return a list of the top 5 blogs in desending order of hub scores.
"""

def answer_eight():
    h,a = nx.hits(G2)   
    sorted_h = sorted(h.items(), key = lambda x:x[1],reverse = True)[:5]
    
    return [keys for keys,h_scores in sorted_h]
    

"""
Question 9

    Apply the HITS Algorithm to this network to find the 5 nodes with highest authority scores.
    This function should return a list of the top 5 blogs in desending order of authority scores.
"""
def answer_nine():
    h,a = nx.hits(G2)   
    sorted_a = sorted(a.items(), key = lambda x:x[1],reverse = True)[:5]
    
    return [keys for keys,a_value in sorted_a]
    