##################################################
####        FinPlot.py
####        Kevin Osicki
####        
####        additional code modified from Derek Banas
####        https://github.com/derekbanas/Python4Finance
####
####
##################################################


import pandas as pd
import pandas_ta as ta
import yfinance as yf

#import matplotlib.pyplot as plt

import numpy as np

import os
from os import listdir
from os.path import isfile, join

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FinPlot():

    def __init__(self, symbol, **kwargs):
    
        self.symbol = symbol
        
        #start and end dates for plotting
        self.start = kwargs.get("start", '')
        self.end = kwargs.get("end", '')
        
        #default to 5 years of data returned for plotting purposes
        self.period = kwargs.get("period", '5y')
        self.interval = kwargs.get("interval", '1d')
        
        #main data frame
        self.df = pd.DataFrame()
        
        
        
    def get_data(self, **kwargs):
        """
        Retrieves data from file or yfinance call based on supplied criteria
        
        keys:
            default = Boolean, use default settings or not
            
            folder = String, location of csv folder
            
            file = string, full file name to read

        Args:
            () returns default values from yfinance, 5Y, 1D

            (default=False) pulls from supplied time frame or interval

            (folder='') reads csv data from folder and (symbol).csv

            (file='') reads csv data from file name directly
            

        """ 
        
        
        #check if OS location given to CSV file, else load from ta's ticker function from yfinance
        if 'file' in kwargs:
            print('Getting data from CSV')
            # Try to get the file and if it doesn't exist issue a warning
            try:
                self.df = pd.read_csv(kwargs['file'], index_col=0)
            except FileNotFoundError:
                print("File Doesn't Exist")
            else:
                return self.df
        
        #check if folder name has been supplied and try read file
        elif 'folder' in kwargs:
            print('Getting data from CSV')
            # Try to get the file and if it doesn't exist issue a warning
            try:
                self.df = pd.read_csv(kwargs['folder'] + self.symbol + '.csv' , index_col=0)
            except FileNotFoundError:
                print("File Doesn't Exist")
            else:
                return self.df
        
        #default is to read directly from yahoo finance via pandas_ta
        else:
            print('Getting data from yfinance')

            #default values check, will use 5y / 1d values, ELSE use given date time frame
            if kwargs.get('default', True) is True:
                print('Using default values, 5Y and 1d')
                self.df = self.df.ta.ticker(self.symbol, period=self.period, interval=self.interval)
            else:
                self.df = self.df.ta.ticker(self.symbol, start=self.start, end=self.end)
        
        #append cummulitive gains to df
        #self.df.ta.log_return(cumulative=True, append=True)
        
    def save_data (self, folder):

        try:
            file = folder + self.symbol.replace(".", "_") + '.csv'
            self.df.to_csv(file)
        except Exception as ex:
            print("Couldn't save file")



    def plot_RSI(self, **kwargs):
        """
        Retrieves RSI (14) and plots the data based upon 
        
        Upper limit of 70% indicated Over Bought

        lower limit of 30% indicates Over Sold


        keys:
            time frame    

        Args:
            () - plots all data in data frame

            (span = True) plots data within class time span values
        
        """ 

        #create figure object
        fig = go.Figure()
        
        #add RSI (14) data to the dataframe
        self.df.ta.rsi(append=True)

        #create plot object with RSI data
        RSI = go.Scatter(x=self.df.index, y=self.df['RSI_14'], line=dict(color='rgba(0,0,255,1)', width=1), name="RSI")
        
        #plot RSI and upper and lower bands
        fig.add_trace(RSI)
        fig.add_hline(y=70, line=dict(color='rgba(0,255,0,0.75)', width=1, dash='dot'), name="Over Bought")
        fig.add_hline(y=30, line=dict(color='rgba(255,0,0,0.75)', width=1, dash='dot'), name="Over Sold")

        #add titles and axis labels

        fig.update_layout(title=f"RSI Plot of {self.symbol}")
        fig.update_yaxes(title="RSI index", range=[0,100])


        fig.update_layout(height=1000, width=1400)

        fig.show()



    def get_fill_color(label):
        ''' Create fill colors of green and red for ichimoku plots '''


        if label >= 1:
            #returns Green, alpha 20%
            return 'rgba(0,250,0,0.2)'
        else:
            #returns Red, alpha 20%
            return 'rgba(250,0,0,0.2)'



    def plot_ichimoku(self):










        #   ICS_26 - Chikou span -Lagging Span = Price shifted back 26 periods
        #   IKS_26 - Kijun-sen - Base Line = (Highest Value in period + Lowest value in period)/2 (26 Sessions)
        #   ITS_9 - Tenkan-sen - Conversion Line = (Highest Value in period + Lowest value in period)/2 (9 Sessions)

        #   ISA_9 - Senkou span A - Leading Span A = (Conversion Value + Base Value)/2 (26 Sessions)
        #   ISB_26 - Senkou Span B - Leading Span B = (Conversion Value + Base Value)/2 (52 Sessions)

    
        #regular candle sticks
        #candle = go.Candlestick(x=df.index, open=df['Open'],high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")


        #generate ichimoko data points append to df
        self.df.ta.ichimoku(append=True)
        
        #the forward ichimoku span A and B 
        df_fw = df.ta.ichimoku()[1]

        #generate working copy of DF for clouds
        df, df1 = self.df.copy()
        #df1 = df.copy()


        #Create a figure object of 2 subplots        
        fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Ichimoku Plot of {self.symbol}", "RSI"), row_heights=[0.7, 0.3], shared_xaxes=True)

        #fig = go.Figure()
            

        df['label'] = np.where(df['ISA_9'] > df['ISB_26'], 1, 0)
        df['group'] = df['label'].ne(df['label'].shift()).cumsum()

        df = df.groupby('group')

        dfs = []
        for name, data in df:
            dfs.append(data)

        for df in dfs:
            fig.add_traces(go.Scatter(x=df.index, y=df.ISA_9, opacity=0.5,
            showlegend=False,
            line=dict(color='rgba(0,0,0,0)')))

            fig.add_traces(go.Scatter(x=df.index, y=df.ISB_26, opacity=0.5,
            line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty',
            showlegend=False,
            fillcolor=get_fill_color(df['label'].iloc[0])))

            #add dashed vertical line to each segment
            #fig.add_vline(x=df.index[0], line_width=1, line_dash="dash", line_color="green", row=1, col=1)



        #generate lines
        
        #get the nicer looking trending candle sticks
        hadf = ta.ha(open_= df1.Open, close = df1.Close, high=df1.High, low=df1.Low)

        #create Plot objects
        candle = go.Candlestick(x = hadf.index, open = hadf.HA_open, high= hadf.HA_high, low=hadf.HA_low, close= hadf.HA_close, name='Candles', legendrank=6)
        baseline = go.Scatter(x=df1.index, y=df1['IKS_26'],line=dict(color='rgba(0,0,0,0.5)', width=1), name="Baseline", legendrank=1)
        conversion = go.Scatter(x=df1.index, y=df1['ITS_9'], opacity=0.5, line=dict(color='rgba(255,100,0,0.5)', width=1), name="Conversion", legendrank=2)
        lagging = go.Scatter(x=df1.index, y=df1['ICS_26'], opacity=0.5, line=dict(color='rgba(0,0,255,0.5)', width=1), name="Lagging", legendrank=3)
        span_a = go.Scatter(x=df1.index, y=df1['ISA_9'], opacity=0.5, line=dict(color='rgba(0,255,0,0.5)', width=1, dash='dot'), name="Span A", legendrank=4)
        span_b = go.Scatter(x=df1.index, y=df1['ISB_26'], opacity=0.5, line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dot'), name="Span B", legendrank=5)


        #plot foward spans

        span_a_fw = go.Scatter(x=df_fw.index, y=df_fw['ISA_9'], opacity=0.5, line=dict(color='rgba(0,255,0,1)', width=1, dash='dot'), name="Span A FW", legendrank=7) 
        span_b_fw = go.Scatter(x=df_fw.index, y=df_fw['ISB_26'], opacity=0.5, line=dict(color='rgba(255,0,0,1)', width=1, dash='dot'), name="Span B FW", legendrank=8)

        #add each plot to the figure on Subplot 1 (top)
        fig.add_trace(candle, row=1, col=1)
        fig.add_trace(baseline, row=1, col=1)
        fig.add_trace(conversion, row=1, col=1)
        fig.add_trace(lagging, row=1, col=1)
        fig.add_trace(span_a, row=1, col=1)
        fig.add_trace(span_b, row=1, col=1)
        fig.add_trace(span_a_fw, row=1, col=1)
        fig.add_trace(span_b_fw, row=1, col=1)


        #add RSI to subplot 2 (bottom)
        RSI = go.Scatter(x=df1.index, y=df1['RSI_14'], line=dict(color='rgba(0,0,255,1)', width=1), name="RSI")

        fig.add_trace(RSI, row=2, col=1)
        fig.add_hline(y=70, line=dict(color='rgba(0,255,0,0.75)', width=1, dash='dot'), name="Over Bought", row=2, col=1)
        fig.add_hline(y=30, line=dict(color='rgba(255,0,0,0.75)', width=1, dash='dot'), name="Over Sold", row=2, col=1)



        #add titles and axis labels
        
        fig.update_yaxes(title="Price in $", row=1, col=1)
        fig.update_yaxes(title="RSI index", range=[0,100] , row=2, col=1)
        fig.update_xaxes(row=2, col=1)

        fig.update_layout(height=1000, width=1400, showlegend=True)

        fig.show()














####
####
####


if __name__ == '__main__':
    
    test2 = FinPlot('AMD', start='2020-01-01', end='2022-01-01')
    # print(test2.symbol)
    # print(test2.start)
    # print(test2.end)
    # print(test2.period)
    # print(test2.interval)

    #test2.get_data(default=False, file='D:/Temp/StockData/AMD.csv')
    test2.get_data(default=False, folder='D:/Temp/StockData/')
    #test2.get_data()

    #test2.plot_RSI()

    print(test2.df)

    #test2.plot_ichimoku()

    #test2.save_data('D:/Temp/StockData/')


    