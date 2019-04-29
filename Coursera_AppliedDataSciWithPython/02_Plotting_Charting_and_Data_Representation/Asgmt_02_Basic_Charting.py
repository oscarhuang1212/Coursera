# File name:   Asgmt_02_Plotting_Weather_Patterns.py
# Author:      Oscar Huang
# Description:  "Applied Data Science with Python" Specialization by University of Michigan on Coursera
#               Course2: Applied Plotting, Charting & Data Representation in Python
#               Week2: Basic Charting

"""
An NOAA dataset has been stored in the file data/C2A2_data/BinnedCsvs_d400/8644828ba50ed8e126342705be1077aa540587e23b45fbcf1209fa7b.csv. 
    The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) Daily Global Historical 
    Climatology Network (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.

Each row in the assignment datafile corresponds to a single observation.

The following variables are provided to you:

    id : station identification code
    date : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
    element : indicator of element type
        TMAX : Maximum temperature (tenths of degrees C)
        TMIN : Minimum temperature (tenths of degrees C)
    value : data value for element (tenths of degrees C)

For this assignment, you must:

    Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of 
        the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high 
        and record low temperatures for each day should be shaded.
    Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
    Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
    Make the visual nice! Leverage principles from the first module in this course when developing your solution. 
        Consider issues such as legends, labels, and chart junk.

The data you have been given is near Moscow, Idaho, United States, and the stations the data comes from are shown on the map below.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def c2f(temp):
    return (temp/5.*9.)+32
    
    
def main(binsize, hashid):
    df = (pd.read_csv('{}.csv'.format(hashid))
        .sort_values('Date'))
    
    #Organizing data
    ##Split data to 2005-2014 and 2015, Delete data on leap day
    df2015 = df[df['Date']>='2015-01-01']
    df= df[df['Date']<'2015-01-01']
    df['Date']=pd.to_datetime(df['Date']).dt.strftime('%m-%d')
    df2015['Date'] = pd.to_datetime(df2015['Date']).dt.strftime('%m-%d')
    df = df[df['Date']!='02-29']
    df2015 = df2015[df2015['Date']!='02-29']
    
    ##Get Max, Min Temperature (Groupby date)
    Temp = df.groupby('Date', as_index=False)[['Data_Value']].max()
    Temp.columns=['Date','Max']
    Temp['Max']= Temp['Max']/10
    Temp['Min'] = df.groupby('Date', as_index=False)[['Data_Value']].min()['Data_Value']/10
    Temp['Max2015'] = df2015.groupby('Date', as_index=False)[['Data_Value']].max()['Data_Value']/10
    Temp['Min2015'] = df2015.groupby('Date', as_index=False)[['Data_Value']].min()['Data_Value']/10
      
    ##Check if Temperature in 2015 is higher/lower than 2005-2014
    Temp['Lower'] = Temp.apply(lambda row: row['Min2015']<row['Min']  ,axis=1)
    Temp['Higher'] = Temp.apply(lambda row: row['Max2015']>row['Max']  ,axis=1)
    Temp['Date']=pd.to_datetime('2015-'+Temp['Date'])
    
     
    
    #Plot
    fig=plt.figure(figsize=(20, 10))
    ax=fig.add_subplot(111)
    Date = np.arange('2015-01-01', '2016-01-01', dtype='datetime64[D]')
      
     
            
    ##Scatters
    plt.plot(Date,Temp.Min2015,'o',markevery=Temp[Temp['Lower'] == True].index.tolist(), color = 'blue',ms= 5)
    plt.plot(Date,Temp.Max2015,'o',markevery=Temp[Temp['Higher'] == True].index.tolist(), color = 'red',ms= 5)
    
    
    
    ##Lines
    ax.plot(Date,Temp.Min, '-',color = 'blue', alpha = 0.5)   
    ax.plot(Date,Temp.Max, '-',color = 'red', alpha = 0.5)   
    
    ###Fill Between Lines
    plt.gca().fill_between(Date,
                      Temp.Min, Temp.Max, 
                      facecolor='gray', 
                      alpha=0.1)
    
    
    #Ticks and Ticks Lables
    myFmt = mdates.DateFormatter('%b')
    months = mdates.MonthLocator()
    
    #Put Month on Minor ticks
    plt.gca().xaxis.set_major_locator(dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(ticker.NullFormatter())
    plt.gca().xaxis.set_minor_locator(dates.MonthLocator(bymonthday=15))
    plt.gca().xaxis.set_minor_formatter(dates.DateFormatter('%b'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f$^\circ$C '))
       
    ##Set limits to X and Y axis
    datemin = datetime.date(2015, Temp.Date.min().month, 1)
    datemax = datetime.date(2015, Temp.Date.max().month, 31)
    ax.set_xlim(datemin, datemax)
    ax.set_yticks(np.linspace(-40,40,5))
    
    ###Add Second Axis (Fahrenheit)
    ymin,ymax = ax.get_ylim()
    ax_f=ax.twinx()
    ax_f.set_ylim(c2f(ymin),c2f(ymax))
    ax_f.yaxis.set_major_formatter(FormatStrFormatter('%.0f$^\circ$F'))
    ax_f.set_yticks(np.linspace(-40,104,5))
    
    
    ##Hide the Minor Ticks
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    ##Set tick size
    ax.tick_params('both', length=10, width=2, labelsize=20)
    ax.tick_params('x', which = 'minor', labelsize=20)
    ax_f.tick_params('both', length=10, width=2,labelsize=20)
    
    
    
    
    #Remove the Top Spines
    ax.spines['top'].set_visible(False)
    ax_f.spines['top'].set_visible(False)
    
    #Set Title
    plt.gca().set_title('Daily High and Low Temperature in Moscow, Idaho Area', size = 28)
    
    #Legends
    Lhandles,Llabels = ax.get_legend_handles_labels()
    
    Lhandles = [Lhandles[2], Lhandles[3], Lhandles[0], Lhandles[1]]
    Llabels = [Llabels[2], Llabels[3], Llabels[0], Llabels[1]]
    legend = ax.legend(Lhandles, Llabels, loc='lower center', shadow = False,frameon=False)   
    
    ##Edit Legends' text
    legend.get_texts()[0].set_text('2005-2014 Minimum temperature')
    legend.get_texts()[1].set_text('2005-2014 Maximum temperature')
    legend.get_texts()[2].set_text('2015 Temperature (Lower than 2005-2014)')
    legend.get_texts()[3].set_text('2015 Temperature (Higher than 2005-2014)')
    
    plt.setp(ax.get_legend().get_texts(), fontsize='20' )
    
    
    ##Edit Legends' Marks and Lines
    legend.legendHandles[0]._legmarker.set_markersize(50)
    legend.legendHandles[1]._legmarker.set_markersize(50)
    legend.legendHandles[2]._legmarker.set_markersize(8)
    legend.legendHandles[3]._legmarker.set_markersize(8)   
    legend.get_lines()[0].set_linewidth(3.0)
    legend.get_lines()[1].set_linewidth(3.0)
    
    plt.show()    
    return  

main(400,'8644828ba50ed8e126342705be1077aa540587e23b45fbcf1209fa7b')
