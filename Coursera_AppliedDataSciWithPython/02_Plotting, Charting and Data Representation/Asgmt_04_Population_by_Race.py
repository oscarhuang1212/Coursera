# File name:   Asgmt_04_Population_by_Race.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course2: Applied Plotting, Charting & Data Representation in Python
#               Week4: Applied Visualizations


"""
Assignment 4

This assignment requires that you to find at least two datasets on the web which are related, and that you visualize these datasets to answer a question 
    with the broad topic of religious events or traditions (see below) for the region of Moscow, Idaho, United States, or United States more broadly.

You can merge these datasets with data from different regions if you like! For instance, you might want to compare Moscow, Idaho, United States to
    Ann Arbor, USA. In that case at least one source file must be about Moscow, Idaho, United States.

You are welcome to choose datasets at your discretion, but keep in mind they will be shared with your peers, so choose appropriate datasets. 
    Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload 
    datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the 
    Coursera terms of service with respect to intellectual property.

Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. 
    You are welcome to provide multiple visuals in different languages if you would like!

As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) 
    and aligning with Cairo’s principles of truth, beauty, function, and insight.

Here are the assignment instructions:
    1. State the region and the domain category that your data sets are about (e.g., Moscow, Idaho, United States and religious events or traditions).
    2. You must state a question about the domain category and region that you identified as being interesting.
    3. You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, 
        or links to websites which might have data in tabular form, such as Wikipedia pages.
    4. You must upload an image which addresses the research question you stated. In addition to addressing the question, 
        this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
    5. You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob, os
import matplotlib.gridspec as gridspec

    
skipcols = ['HC01_VC01','HC01_VC02','HC01_VC03','HC01_VC04']
list_race = ['White','Black or African American','American Indian and Alaska Native','Asian','Native Hawaiian and Other Pacific Islander','Some other race']
list_race_nw=['American Indian and Alaska Native','Asian','Black or African American','Native Hawaiian and Other Pacific Islander','Some other race']

def import_file():
    allFiles = glob.glob("Data/*.csv")    
    allFiles.sort()
    df = pd.DataFrame()

    for files in allFiles:
        frame = pd.read_csv(files,index_col=None, header = 0)
        frame =frame.drop(frame.index[0])
        df = pd.concat([df,frame],sort = True)

    return df

def clean_data(df):
    df_13to16 = df.set_index([['2010','2011','2012','2013','2014','2015','2016']]).iloc[3:,3:].T
    df_10to12 = df.set_index([['2010','2011','2012','2013','2014','2015','2016']]).iloc[:3,3:].T

    for row in df_13to16.index:
        if (row.split('_')[0] == 'HC04') or (row.split('_')[0] == 'HC03'):
            df_13to16.drop(row, inplace = True)
        elif (int(row.split('VC')[1])) not in range(78,84):
            df_13to16.drop(row, inplace = True)
                      
    for row in df_10to12.index:
        if (row.split('_')[0] == 'HC04') or (row.split('_')[0] == 'HC03'):
            df_10to12.drop(row, inplace = True)
        elif (int(row.split('VC')[1])) not in range(72,78):
            df_10to12.drop(row, inplace = True)
    
    df_10to12.set_index([df_10to12.index.str.split('_').str.get(0),df_10to12.index.str.split('_').str.get(1)], inplace = True)
    df_10to12.rename(index={'HC01':'Estimate','HC02':'Error'}, inplace = True)
    df_10to12.index.set_levels(list_race,level=1,inplace=True)
    
    df_13to16.set_index([df_13to16.index.str.split('_').str.get(0),df_13to16.index.str.split('_').str.get(1)], inplace = True)
    df_13to16.rename(index={'HC01':'Estimate','HC02':'Error'}, inplace = True)
    df_13to16.index.set_levels(list_race,level=1,inplace=True)

    
    df_clean = pd.concat([df_10to12,df_13to16],sort = True).groupby(level=[0,1]).last().T

    df_clean=df_clean.astype(float)
    return df_clean


df=clean_data(import_file())
df_nw = df.drop('White', axis=1, level=1)

fig,ax2 = plt.subplots()
gspec = gridspec.GridSpec(10,1)
ax2 = plt.subplot(gspec[:8])





def pick(event):
    
    ax2.cla()
    bars = (df_nw.Estimate['{}'.format(event.artist.get_label())]
                .plot(kind='bar',ax=ax2,alpha=0, 
                      yerr=df_nw.Error['{}'.format(event.artist.get_label())]
                      ,capsize=3
                      ,error_kw=dict(ecolor=event.artist.get_color(),elinewidth=0.8,alpha = 0.7)))
    
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    draw_lines()
    ax2.set_ylim(-100,1300)
    ax2.set_xticklabels(['2010','2011','2012','2013','2014','2015','2016'], rotation=0)

    for line in ax2.lines:
        if (line.get_label() != event.artist.get_label()):
            line.set_alpha(0.5)
        else:
            line.set_alpha(1.0)
        
    plt.show()

    
def draw_lines():
    df_nw.Estimate.plot(kind = 'line',ax = ax2,picker = 5)

    ax2.set_xlim(-0.5,6.5)
    ax2.set_title('Population by race (Without White) in Moscow, Idaho')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Population')
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_xticklabels(['2009','2010','2011','2012','2013','2014','2015','2016'], rotation=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    leg = ax2.legend(list_race_nw,loc='lower center', bbox_to_anchor=(0.5, -0.35), frameon=False, prop={'size': 8}, ncol =2)
    
    for line in leg.get_lines():
        line.set_linewidth(4.0)
        line.set_picker(5)


def plot_white():
    fig,ax1 = plt.subplots()
    gspec = gridspec.GridSpec(10,1)
    ax1 = plt.subplot(gspec[:8])
    df.Estimate.plot(kind = 'line',ax = ax1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population')
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.set_title('Population by race in Moscow, Idaho')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), frameon=False, prop={'size': 8}, ncol =2)


draw_lines()
ax2.set_ylim(-100,1300)

plt.gcf().canvas.mpl_connect('pick_event', pick)

###To plot the population by race with Whites, use the Following Function:
#plot_white()


plt.show()
