# File name:   Asgmt_02_Network_Connectivity.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course5: Applied_Social_Network_Analysis_in_Python
#               Week2:   Network Connectivity

"""
Assignment 2 - Network Connectivity

    In this assignment you will go through the process of importing and analyzing an internal email communication network between 
        employees of a mid-sized manufacturing company. Each node represents an employee and each directed edge between two nodes represents 
        an individual email. The left node represents the sender and the right node represents the recipient.
"""

import networkx as nx

"""
Question 1

    Using networkx, load up the directed multigraph from email_network.txt. Make sure the node names are strings.
    This function should return a directed multigraph networkx graph.
"""
def answer_one():
    G = nx.read_edgelist('email_network.txt',create_using = nx.MultiDiGraph(), data=(('time',int),))
    
    return G


"""
Question 2

    How many employees and emails are represented in the graph from Question 1?
    This function should return a tuple (#employees, #emails).
"""
def answer_two():
        
    return (answer_one().number_of_nodes(),answer_one().number_of_edges())


"""
Question 3

    Part 1. Assume that information in this company can only be exchanged through email.
    When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information 
        to the receiver, but not vice versa.
    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

    Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways.
    Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?

    This function should return a tuple of bools (part1, part2).
"""

def answer_three():
        
    return (nx.is_strongly_connected(answer_one()),nx.is_weakly_connected(answer_one()))


"""
Question 4

    How many nodes are in the largest (in terms of nodes) weakly connected component?
    This function should return an int.
"""

def answer_four():
        
    return len(max(nx.weakly_connected_components(answer_one()),key=len))


"""
Question 5

    How many nodes are in the largest (in terms of nodes) strongly connected component?
    This function should return an int
"""

def answer_five():
        
    return len(max(nx.strongly_connected_components(answer_one()),key=len))


"""
Question 6

    Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
        Call this graph G_sc.

    This function should return a networkx MultiDiGraph named G_sc.
"""
def answer_six():
        
    G_sc = max(nx.strongly_connected_components(answer_one()), key=len)
    G2 = answer_one().subgraph(G_sc).copy()

    return G2


"""
Question 7

    What is the average distance between nodes in G_sc?
    This function should return a float.
"""
def answer_seven():
    
    return nx.average_shortest_path_length(answer_six())


"""
Question 8

    What is the largest possible distance between two employees in G_sc?
    This function should return an int.
"""
def answer_eight():
        
    return nx.diameter(answer_six())


"""
Question 9

    What is the set of nodes in G_sc with eccentricity equal to the diameter?
    This function should return a set of the node(s).
"""
def answer_nine():
       
    return set(nx.periphery(answer_six()))


"""
Question 10

    What is the set of node(s) in G_sc with eccentricity equal to the radius?
    This function should return a set of the node(s).
"""
def answer_ten():
        
    return set(nx.center(answer_six()))




"""
Question 11

    Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
    How many nodes are connected to this node?
    This function should return a tuple (name of node, number of satisfied connected nodes).
"""
def answer_eleven():
        
    G_sc = answer_six()
    diameter = answer_eight()
        
    node_list={}
    
    for node in G_sc:
        node_list.update({node:sum(1 for x in nx.single_source_shortest_path_length(G_sc,source = node).values() if x ==3)})
    
    return sorted(node_list.items(), key=lambda x: x[1], reverse = True)[0]


"""
Question 12

    Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, 
        what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question 
        or the center nodes)?
    This function should return an integer.
"""
def answer_twelve():
    
    center = nx.center(answer_six())[0]
    prevent_node = answer_eleven()[0]
        
    return len(nx.minimum_node_cut(answer_six(),center,prevent_node))

print(answer_twelve())



"""
Question 13

    Construct an undirected graph G_un using G_sc (you can ignore the attributes).
    This function should return a networkx Graph.
"""

def answer_thirteen():
    G_un = nx.Graph(answer_six())
    
    return G_un


"""
Question 14

    What is the transitivity and average clustering coefficient of graph G_un?
    This function should return a tuple (transitivity, avg clustering).
"""


def answer_fourteen():
    
    tran = nx.transitivity(answer_thirteen())
    acc=nx.average_clustering(answer_thirteen())

    return (tran,acc)