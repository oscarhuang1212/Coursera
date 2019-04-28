# File name:   Asgmt_04_Hypothesis_Testing.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course1: Introduction to Data Science in Python
#               Week4: Statistical Analysis in Python and Project

"""
Assignment 4 - Hypothesis Testing

    This assignment requires more individual learning than previous assignments - you are encouraged to check out the pandas documentation 
        to find functions or methods you might not have used yet, or ask questions on Stack Overflow and tag them as pandas and python related. 
    And of course, the discussion forums are open for interaction with your peers and the course staff.


    Definitions:

        A quarter is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, 
            Q4 is October through December.
        A recession is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
        A recession bottom is the quarter within a recession which had the lowest GDP.
        A university town is a city which has a high percentage of university students compared to the total population of the city.

    Hypothesis: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of 
                    houses in university towns the quarter before the recession starts compared to the recession bottom. 
                    (price_ratio=quarter_before_recession/recession_bottom)

    The following data files are available for this assignment:

        From the Zillow research data site there is housing data for the United States. In particular the datafile for all homes at a city level, 
            City_Zhvi_AllHomes.csv, has median home sale prices at a fine grained level.
        From the Wikipedia page on college towns is a list of university towns in the United States which has been copy and pasted into 
            the file university_towns.txt.
        From Bureau of Economic Analysis, US Department of Commerce, the GDP over time of the United States in current dollars 
            (use the chained value in 2009 dollars), in quarterly intervals, in the file gdplev.xls. For this assignment, only look at GDP data from 
                the first quarter of 2000 onward.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import datetime as dt



# Use this dictionary to map state names to two letter acronyms
def gdp_annual():
    gdp = (pd.read_excel('gdplev.xls')
          .iloc[4:,:3]
          )
    gdp.columns = gdp.iloc[0]
    gdp['year']=gdp.iloc[:,0]
    gdp = (gdp.set_index('year')
           .iloc[3:,2:]
           .dropna(how = 'all')
          )
    gdp.index =gdp.index.astype('int')
    return gdp
    
def gdp_q():
    gdp = (pd.read_excel('gdplev.xls')
          .iloc[4:,4:7]
          )
    gdp.columns = gdp.iloc[0]
    gdp['year']=gdp.iloc[:,0]
    gdp = gdp.set_index(['year']).iloc[3:,2:]
    gdp = gdp[gdp.index.str.split('q').str[0].astype('int')>=2000]
    
    return gdp


"""
Returns a DataFrame of towns and the states they are in from the university_towns.txt list. The format of the DataFrame should be:
    
    The following cleaning needs to be done:
    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
"""
def get_list_of_university_towns():
    
    ut = pd.read_csv('university_towns.txt',sep = '\n',header = None)
    
    
    ut['State']=ut[ut[0].str.contains('\[ed*')]
    ut['State']=ut['State'].fillna(method='ffill')
    
    ut.columns = ['RegionName','State']
        
    ut=ut[ut.RegionName != ut.State]
    ut['State'] = ut['State'].str.replace('\[.*','')
    ut['RegionName'] = ut['RegionName'].str.replace(' \(.*','')
    ut=ut[['State','RegionName']].reset_index(drop=True) 
       
    return ut




# Returns the year and quarter of the recession start time as a string value in a format such as 2005q3
def get_recession_start():
    
    gdp=gdp_q()
    gdp.columns =['GDP']
    
    gdp['recession'] = 0
    recessionIdx=1
    
    for i in range(2,len(gdp)):
        if gdp.recession[i-1] == 0:
            if gdp.GDP[i]<gdp.GDP[i-1] and gdp.GDP[i+1]<gdp.GDP[i]:
                gdp.iloc[i,1] = recessionIdx
        else:
            if gdp.GDP[i-1]>gdp.GDP[i-2] and gdp.GDP[i-2]>gdp.GDP[i-3]:            
                gdp.iloc[i,1] = 0
                recessionIdx+=1
            else:
                gdp.iloc[i,1] = recessionIdx
        
    return gdp[gdp.recession>0].index[0]



# Returns the year and quarter of the recession end time as a string value in a format such as 2005q3
def get_recession_end():
    
    gdp=gdp_q()
    gdp.columns =['GDP']
    
    gdp['recession'] = 0
    recessionIdx=1
    
    for i in range(2,len(gdp)):
        if gdp.recession[i-1] == 0:
            if gdp.GDP[i]<gdp.GDP[i-1] and gdp.GDP[i+1]<gdp.GDP[i]:
                gdp.iloc[i,1] = recessionIdx
        else:
            if gdp.GDP[i-1]>gdp.GDP[i-2] and gdp.GDP[i-2]>gdp.GDP[i-3]:            
                gdp.iloc[i,1] = 0
                recessionIdx+=1
            else:
                gdp.iloc[i,1] = recessionIdx

    return gdp[gdp.recession>0].index[-1]



# Returns the year and quarter of the recession bottom time as a string value in a format such as 2005q3
def get_recession_bottom():
    
    gdp=gdp_q()
    gdp.columns =['GDP']
    
    gdp['recession'] = 0
    recessionIdx=1
    
    for i in range(2,len(gdp)):
        if gdp.recession[i-1] == 0:
            if gdp.GDP[i]<gdp.GDP[i-1] and gdp.GDP[i+1]<gdp.GDP[i]:
                gdp.iloc[i,1] = recessionIdx
        else:
            if gdp.GDP[i-1]>gdp.GDP[i-2] and gdp.GDP[i-2]>gdp.GDP[i-3]:            
                gdp.iloc[i,1] = 0
                recessionIdx+=1
            else:
                gdp.iloc[i,1] = recessionIdx

    recession = gdp[gdp.recession>0]

    return recession.astype(float).idxmin().GDP




"""
Converts the housing data to quarters and returns it as mean values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index in the shape of ["State","RegionName"].
    
Note:   Quarters are defined in the assignment description, they are not arbitrary three month periods.
        The resulting dataframe should have 67 columns, and 10,730 rows.
"""
def convert_housing_data_to_quarters():
  
    housing = pd.read_csv('City_Zhvi_AllHomes.csv')
    
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
    
    housing['State'].replace(states,inplace = True)
    housing = housing.set_index(['State','RegionName']).iloc[:,49:]
    
    housing.columns = pd.to_datetime(housing.columns)
                    
    housing = (housing.resample('Q',axis=1).mean()
               .rename(columns=lambda x: str(x.to_period('Q')).lower()))    
    return housing



"""
First creates new data showing the decline or growth of housing prices between the recession start and 
    the recession bottom. Then runs a ttest comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same) is true or not as well as the p-value of the confidence. 
    
Return the tuple (different, p, better) where different=True if the t-test is True at a p<0.01 (we reject the null hypothesis), 
    or different=False if  otherwise (we cannot reject the null hypothesis). The variable p should be equal to the exact p value 
    returned from scipy.stats.ttest_ind(). The value for better should be either "university town" or "non-university town" 
    depending on which has a lower mean price ratio (which is equivilent to a reduced market loss).
"""
def run_ttest():
    housing = convert_housing_data_to_quarters()
    recessionStart = get_recession_start()
    recessionBottom = get_recession_bottom()
    universityTown=get_list_of_university_towns()
    
    qtr_bfr_rec_start = housing.columns[housing.columns.get_loc(recessionStart)-1]
    
    housing['priceRatio'] = housing[qtr_bfr_rec_start].div(housing[recessionBottom] )
    
    df = pd.merge(housing.reset_index(),
                 universityTown,
                 on=universityTown.columns.tolist(),
                 indicator = '_flag', how = 'outer')
    ut = df[df['_flag']=='both']
    nut = df[df['_flag']!='both']
    
    diff = ttest_ind(ut['priceRatio'],nut['priceRatio'], nan_policy='omit')[1] < 0.01
    p = ttest_ind(ut['priceRatio'],nut['priceRatio'], nan_policy='omit')[1]
    
    if ut['priceRatio'].mean() > nut['priceRatio'].mean():
        better = 'non-university town'
    else:
        better ='university town'

    return (diff,p,better)

print(run_ttest())