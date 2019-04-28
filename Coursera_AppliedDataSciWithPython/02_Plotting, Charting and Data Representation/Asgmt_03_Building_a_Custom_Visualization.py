# File name:   Asgmt_02_Basic_Charting.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course2: Applied Plotting, Charting & Data Representation in Python
#               Week3: Charting Fundamentals

"""
Assignment 3 - Building a Custom Visualization

In this assignment you must choose one of the options presented below and submit a visual as well as your source code for peer grading. 
      The details of how you solve the assignment are up to you, although your assignment must use matplotlib so that your peers can evaluate your work. 
      The options differ in challenge level, but there are no grades associated with the challenge level you chose. However, your peers will be asked to 
      ensure you at least met a minimum quality for a given technique in order to pass. Implement the technique fully (or exceed it!) and 
      you should be able to earn full grades for the assignment.


Ferreira, N., Fisher, D., & Konig, A. C. (2014, April). Sample-oriented task-driven visualizations: allowing users to make better, 
      more confident decisions.       In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM. (video)


In this paper the authors describe the challenges users face when trying to make judgements about probabilistic data generated through samples. 
      As an example, they look at a bar chart of four years of data (replicated below in Figure 1). Each year has a y-axis value, 
      which is derived from a sample of a larger dataset. For instance, the first value might be the number votes in a given district or riding for 1992, 
      with the average being around 33,000. On top of this is plotted the 95% confidence interval for the mean (see the boxplot lectures for more information, 
      and the yerr parameter of barcharts).

A challenge that users face is that, for a given y-axis value (e.g. 42,000), it is difficult to know which x-axis values are most likely to be 
      representative, because the confidence levels overlap and their distributions are different (the lengths of the confidence interval bars are unequal). 
      One of the solutions the authors propose for this problem (Figure 2c) is to allow users to indicate the y-axis value of interest (e.g. 42,000) 
      and then draw a horizontal line and color bars based on this value. So bars might be colored red if they are definitely above this value 
      (given the confidence interval), blue if they are definitely below this value, or white if they contain this value.


Easiest option: Implement the bar coloring as described above - a color scale with only three colors, 
                        (e.g. blue, white, and red). Assume the user provides the y axis value of interest as a parameter or variable.

Harder option: Implement the bar coloring as described in the paper, where the color of the bar is actually based on the amount of data covered 
                        (e.g. a gradient ranging from dark blue for the distribution being certainly below this y-axis, to white if the value 
                        is certainly contained, to dark red if the value is certainly not contained as the distribution is above the axis).

Even Harder option: Add interactivity to the above, which allows the user to click on the y axis to set the value of interest. 
                        The bar colors should change with respect to what value the user has selected.

Hardest option: Allow the user to interactively set a range of y values they are interested in, and recolor based on this (e.g. a y-axis band, 
                        see the paper for more details).

Note: The data given for this assignment is not the same as the data used in the article and as a result the visualizations may look a little different.
"""



# Use the following data for this assignment:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np, scipy.stats as st
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker


np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
                  
df['Mean']=df.apply(np.mean, axis = 1)
df['Sem']= df.apply(st.sem, axis = 1)
cmap = mpl.cm.get_cmap('Reds')


def to_color(mouse_y1, mouse_y2, bar_y, bar_sem):
    mouse_max = max(mouse_y1, mouse_y2)
    mouse_min = min(mouse_y1, mouse_y2)
    bar_max = bar_y + bar_sem
    bar_min = bar_y - bar_sem
    
    if mouse_y1 == mouse_y2:
        if mouse_y1 > bar_min and mouse_y1< bar_max:
            cover = 0.99
        else:
            cover = 0.01
    elif mouse_min > bar_max or mouse_max < bar_min:
        cover = 0.01
    else:
        cover = (min(bar_max,mouse_max) -max(bar_min,mouse_min))/(mouse_max-mouse_min)
        
    if cover >= 1:
        cover =0.99
        
    return cover    

fig,ax1 = plt.subplots()
gspec = gridspec.GridSpec(1,11 )
ax1 = plt.subplot(gspec[:9])
x = np.arange(1991, 1996, 0.1)
mutable_object= {'key':10000}
bars = ax1.bar(df.index, df.Mean, width=1.0, edgecolor ='black',linewidth=0.5, color = cmap(.9))
errorbars = ax1.errorbar(df.index,df.Mean,yerr=df.Sem, linestyle="None",ecolor='black',elinewidth=0.8,capsize=5)

ax1.xaxis.set_ticks(df.index)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(1991,1996)
ymax=ax1.get_ylim()[1]

ax2 = ax1.twiny()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)


def onclick(event):
    ax2.cla()
    mutable_object['key'] = event.ydata
    
    ax2.set_xlim(1991,1996)
    ax2.set_ylim(0,ymax)
    
    plt.show()
    
def onrelease(event):
    if event.inaxes == ax2 and mutable_object['key']!= None:
        ax2.axhline(y=event.ydata,color='black',alpha =0.5)#,label = '{}'.format(event.ydata))
        ax2.axhline(y=mutable_object['key'],color='black',alpha =0.5)#,label = '{}'.format(event.ydata))
        

        if mutable_object['key']<event.ydata:
            ax2.text(1996,mutable_object['key'],'{0:.0f} '.format(mutable_object['key']),verticalalignment='top',color='black',alpha=0.8)
            ax2.text(1996,event.ydata,'{0:.0f} '.format(event.ydata),verticalalignment='bottom',color='black',alpha=0.8)
        elif mutable_object['key']>event.ydata:
            ax2.text(1996,mutable_object['key'],'{0:.0f} '.format(mutable_object['key']),verticalalignment='bottom',color='black',alpha=0.8)
            ax2.text(1996,event.ydata,'{0:.0f} '.format(event.ydata),verticalalignment='top',color='black',alpha=0.8)
        else:
            ax2.text(1996,mutable_object['key'],'{0:.0f} '.format(mutable_object['key']),verticalalignment='center',color='black',alpha=0.8)
            ax2.text(1996,event.ydata,'{0:.0f} '.format(event.ydata),verticalalignment='center',color='black',alpha=0.8)
            
        
        
        ax2.fill_between(x,mutable_object['key'],event.ydata, color = 'grey',alpha = 0.5)
        ax2.set_xlim(1991,1996)
        ax2.set_ylim(0,ymax)

        for n in range(0,len(bars)):
            h = bars[n].get_height()
            sem = df.Sem.iloc[n]
            bars[n].set_color(cmap(to_color(mutable_object['key'],event.ydata,h,sem)))
            bars[n].set_edgecolor('black')

        click = False
        plt.show()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('button_release_event', onrelease)


norm = mpl.colors.Normalize(vmin=0, vmax=1)
cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='coolwarm')
cb.set_label('% Selected Range Covered by Confidence Interval')

tick_locs   = [0,0.2,0.4,0.6,0.8,1.0]
tick_labels = ['0','20','40','60','80','100']
cb.locator     = mpl.ticker.FixedLocator(tick_locs)
cb.formatter   = mpl.ticker.FixedFormatter(tick_labels)
cb.update_ticks()

ax1.set_title('Value for Sales (1992 to 1995)')
ax1.set_ylabel('Value  for Sales')
ax1.set_xlabel('Year')

plt.show()
