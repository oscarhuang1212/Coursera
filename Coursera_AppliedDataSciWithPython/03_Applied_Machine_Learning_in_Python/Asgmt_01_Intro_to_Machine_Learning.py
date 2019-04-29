# File name:   Asgmt_01_Intro_to_Machine_Learning.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course3: Applied Machine Learning in Python
#               Week1: Fundamentals of Machine Learning - Intro to SciKit Learn


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

"""
Question 0
    How many features does the breast cancer dataset have? 
    This function should return an integer.
"""

def answer_zero():
    return len(cancer['feature_names'])


"""
Question 1

    Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is 
        not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, 
        so let's practice creating a classifier with a pandas DataFrame.

    Convert the sklearn.dataset cancer to a DataFrame.
"""

def answer_one():
    
    df = pd.DataFrame(data = cancer['data'],columns = cancer['feature_names'])
    df['target'] = cancer['target']
    
    
    return df



"""
Question 2

    What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
    This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']
"""

def answer_two():
    cancerdf = answer_one()

    s = cancerdf.target.sum()
    target= pd.Series([len(cancerdf)-s,s],index=['malignant','benign'])
    
    return target


"""
Question 3

    Split the DataFrame into X (the data) and y (the labels).

    This function should return a tuple of length 2: (X, y), where

        X, a pandas DataFrame, has shape (569, 30)
        y, a pandas Series, has shape (569,).
"""

def answer_three():
    cancerdf = answer_one()

    X=cancerdf.iloc[:,:-1]
    y=cancerdf.iloc[:,-1]

    return X,y


"""
Question 4

    Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
    Set the random number generator state to 0 using random_state=0 to make sure your results match the autograder!
"""
from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
    
    return X_train, X_test, y_train, y_test


"""
Question 5

    Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).
    This function should return a sklearn.neighbors.classification.KNeighborsClassifier.
"""

from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn= KNeighborsClassifier(n_neighbors = 1)
    
    return  knn.fit(X_train,y_train)


"""
Question 6

    Using your knn classifier, predict the class label using the mean value for each feature.

    Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean value for each feature, 
        ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).

    This function should return a numpy array either array([ 0.]) or array([ 1.])

"""

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    
    return knn.predict(means) 


"""
Question 7

    Using your knn classifier, predict the class labels for the test set X_test.
    This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.
"""

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    return knn.predict(X_test)

answer_seven()


"""
Question 8

    Find the score (mean accuracy) of your knn classifier using X_test and y_test.
    This function should return a float between 0 and 1
"""

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    return knn.score(X_test,y_test)


"""
Optional plot

    Try using the plotting function below to visualize the differet predicition scores between training and test sets, 
    as well as malignant and benign cells. 
"""

def accuracy_plot():
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()

accuracy_plot()

