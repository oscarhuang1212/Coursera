# File name:   Asgmt_01_Working_with_Text_in_Python.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course4: Applied Text Mining in Python
#               Week1:  Working with Text in Python


"""
Assignment 1

    In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data.

    Each line of the dates.txt file corresponds to a medical note. Each note has a date that needs to be extracted, 
        but each date is encoded in one of many formats.

    The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize 
        and sort the dates.

    Here is a list of some of the variants you might encounter in this dataset:

            04/20/2009; 04/20/09; 4/20/09; 4/3/09
            Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
            20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
            Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
            Feb 2009; Sep 2009; Oct 2010
            6/2008; 12/2009
            2009; 2010

    Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to 
        the following rules:

            Assume all dates in xx/xx/xx format are mm/dd/yy
            Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
            If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
            If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
            Watch out for potential typos as this is a raw, real-life derived dataset.

    With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

    For example if the original series was this:

            0    1999
            1    2010
            2    1978
            3    2015
            4    1985

    Your function should return this:

            0    2
            1    4
            2    0
            3    1
            4    3

    This function should return a Series of length 500 and dtype int.
"""

def date_sorter():
    import pandas as pd
    import re

    doc = []
    with open('dates.txt') as file:
        for line in file:
            doc.append(line)
    
    df_ori = pd.Series(doc)
    df = pd.DataFrame(df_ori, columns = ['text'])
    
    #    04/20/2009; 04/20/09; 4/20/09; 4/3/09
    df2 = df['text'].str.extractall(r'(?P<month>\d?\d)[/-](?P<date>\d{,2})[/-](?P<year>\d{2}(\d{2})?)')
    df2 = df2.reset_index(level=1, drop=True)
    df2 = df2[pd.to_numeric(df2.month)<=12]
    df2 = df2[pd.to_numeric(df2.date)<=31]
    
    #    20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
    df3 = df['text'].str.extractall(r'(?P<date>\d{1,2})[\s.,]*(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[\s.,]*(?P<year>\d{4})')
    df3 = df3.reset_index(level=1, drop=True)
    
    #    Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    df4 = df['text'].str.extractall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.,\s]*(?P<date>\d\d?)[,\s]*(?P<year>\d{4})')
    df4 = df4.reset_index(level=1, drop=True)
    
    #    Feb 2009; Sep 2009; Oct 2010
    df5 =df.text[228:].str.extractall(r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))[a-z]*[.,\s]*(?P<year>\d{4})')
    df5 = df5.reset_index(level=1, drop=True)
    
    #    6/2008; 12/2009
    df6 = df.text[343:].str.extractall(r'(?P<month>\d?\d)/(?P<year>\d{4})')
    df6 = df6.reset_index(level=1, drop=True)
    
    #    2009; 2010
    df7 = df.text[455:].str.extractall(r'(?P<year>[1|2]\d{3})')
    df7 = df7.reset_index(level=1, drop=True)
    
    
    result = pd.concat([df2,df3,df4,df5,df6,df7], ignore_index=False, sort = True)
    result = result[~result.index.duplicated(keep='first')].sort_index()
    result.drop(3,axis=1,inplace = True)
    result.fillna(1,inplace=True)
    

    #convert years
    result.year = result.year.astype(int).apply(lambda x: x+1900 if x<100 else x)
    result.year = result.year.astype(str)

    #convert months
    result.month = result.month.astype(str)


    #convert date
    result.date = result.date.astype(str)


    result['time'] = result.month + '/' + result.date + '/' + result.year 

    r = pd.to_datetime(result.time)


    return r.argsort()

print(date_sorter())