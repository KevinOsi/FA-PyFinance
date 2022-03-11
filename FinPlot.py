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
import datetime

#Not using now
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
                self.df = pd.read_csv(kwargs['file'], index_col=0, parse_dates=True)
            except FileNotFoundError:
                print("File Doesn't Exist")
            else:
                return self.df
        
        #check if folder name has been supplied and try read file
        elif 'folder' in kwargs:
            print('Getting data from CSV')
            # Try to get the file and if it doesn't exist issue a warning
            try:
                self.df = pd.read_csv(kwargs['folder'] + self.symbol + '.csv' , index_col=0, parse_dates=True)
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
                
                self.df = self.df.ta.ticker(self.symbol, period=self.period)
                
            else:
                self.df = self.df.ta.ticker(self.symbol, start=self.start, end=self.end)
        
        #append cummulitive gains to df
        #self.df.ta.log_return(cumulative=True, append=True)
        
    def save_data (self, folder, **kwargs):
        """
        Saves the data to a CSV files at a folder location 
        
        keys:
            all=True  save all data or just the     

        Args:
            (foldername) - saves all data to <Folder>/symbol.csv

            (foldername, all=False) - saves only core data to <Folder>/symbol.csv
         
        
        """ 
        
        
        try:
            file = folder + self.symbol.replace(".", "_") + '.csv'
            self.df.to_csv(file)
        except Exception as ex:
            print("Couldn't save file")


    def get_plot_func(self, name):
        """
        returns the function name for the requested chart
        
        """
        if name.lower() == "rsi":
            return self.plot_RSI
        elif name.lower() == "ichimoku":
            return self.plot_ichimoku
        elif name.lower() == "sma":
            return self.plot_sma
        elif name.lower() == "volume":
            return self.plot_volume
        elif name.lower() == "macd":
            return self.plot_macd    
        elif name.lower() == "bollinger":
            return self.plot_bollinger   
        else: 
            return self.plot_RSI


    def plot_RSI(self, fig, row, col):
        """
        Retrieves RSI (14) and plots the data based upon 
        
        Upper limit of 70% indicated Over Bought

        lower limit of 30% indicates Over Sold

        Args:
            fig = a figure object to add our subplot to

            row = sepcific row to plot to

            col = sepecific col to plot to (typicaly 1)

        Uses:
            (fig, row, col) - plots all data in data frame

        Returns:
            fig = existing figure object is returned with updates
        """ 

        
        #add RSI (14) data to the dataframe
        self.df.ta.rsi(append=True)

        #create plot object with RSI data
        RSI = go.Scatter(x=self.df.index, y=self.df['RSI_14'], line=dict(color='rgba(0,0,255,1)', width=1), name="RSI", legendgroup = "RSI", legendgrouptitle_text="RSI Plot")
        
        #plot RSI and upper and lower bands
        fig.add_trace(RSI, row=row, col=col)
        fig.add_hline(y=70, line=dict(color='rgba(0,255,0,0.75)', width=1, dash='dot'), name="Over Bought", row=row, col=col)
        fig.add_hline(y=30, line=dict(color='rgba(255,0,0,0.75)', width=1, dash='dot'), name="Over Sold", row=row, col=col)

        #add titles and axis labels
      
        fig.update_yaxes(title="RSI index", range=[0,100], row=row, col=col)
      

        return fig


    def get_fill_color(self, label):
        ''' Create fill colors of green and red for ichimoku plots '''


        if label >= 1:
            #returns Green, alpha 20%
            return 'rgba(0,250,0,0.2)'
        else:
            #returns Red, alpha 20%
            return 'rgba(250,0,0,0.2)'

  
    def plot_ichimoku(self, fig, row, col):
        """
        Retrieves Ichimoku Data and plots
        
        keys:
            forward=True  plot the projected spans [NOT yet implemented]
                       

        Args:
            fig - subplot figure to plot to

            row - row to plot to on fig

            col - col to plot to on fig, usually 1
            
        
        """ 
             
        ### TODO implement check to disable forward span if requested

    
    
        #regular candle sticks
        #candle = go.Candlestick(x=df.index, open=df['Open'],high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")


        #generate ichimoko data points append to df
        self.df.ta.ichimoku(append=True)
        #   ICS_26 - Chikou span -Lagging Span = Price shifted back 26 periods
        #   IKS_26 - Kijun-sen - Base Line = (Highest Value in period + Lowest value in period)/2 (26 Sessions)
        #   ITS_9 - Tenkan-sen - Conversion Line = (Highest Value in period + Lowest value in period)/2 (9 Sessions)
        #   ISA_9 - Senkou span A - Leading Span A = (Conversion Value + Base Value)/2 (26 Sessions)
        #   ISB_26 - Senkou Span B - Leading Span B = (Conversion Value + Base Value)/2 (52 Sessions)


        #the forward ichimoku span A and B 
        df_fw = pd.DataFrame()
        df_fw = self.df.ta.ichimoku()[1]
      

        #generate working copy of DF for clouds & df1 copy for plotting
        df = self.df.copy()
        df1 = self.df.copy()
       
                

        df['label'] = np.where(df['ISA_9'] > df['ISB_26'], 1, 0)
        df['group'] = df['label'].ne(df['label'].shift()).cumsum()

        df = df.groupby('group')

        dfs = []
        for name, data in df:
            dfs.append(data)

      
       

        for df in dfs:
            fig.add_trace(go.Scatter(x=df.index, y=df.ISA_9, opacity=0.5,
            showlegend=False,
            line=dict(color='rgba(0,0,0,0)')), row=row, col=col)

            fig.add_trace(go.Scatter(x=df.index, y=df.ISB_26, opacity=0.5,
            line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty',
            showlegend=False,
            fillcolor= self.get_fill_color(df['label'].iloc[0])), row=row, col=col)
            



        #generate lines
        
        #get the nicer looking trending candle sticks
        hadf = ta.ha(open_= df1.Open, close = df1.Close, high=df1.High, low=df1.Low)

        #create Plot objects
        candle = go.Candlestick(x = hadf.index, open = hadf.HA_open, high= hadf.HA_high, low=hadf.HA_low, close= hadf.HA_close, name='Candles', legendgroup = "Ichimoku", legendgrouptitle_text="Ichiomoku Plot")
        baseline = go.Scatter(x=df1.index, y=df1['IKS_26'],line=dict(color='rgba(0,0,0,0.5)', width=1), name="Baseline", legendgroup = "Ichimoku")
        conversion = go.Scatter(x=df1.index, y=df1['ITS_9'], opacity=0.5, line=dict(color='rgba(255,100,0,0.5)', width=1), name="Conversion", legendgroup = "Ichimoku")
        lagging = go.Scatter(x=df1.index, y=df1['ICS_26'], opacity=0.5, line=dict(color='rgba(0,0,255,0.5)', width=1), name="Lagging", legendgroup = "Ichimoku")
        span_a = go.Scatter(x=df1.index, y=df1['ISA_9'], opacity=0.5, line=dict(color='rgba(0,255,0,0.5)', width=1, dash='dot'), name="Span A", legendgroup = "Ichimoku")
        span_b = go.Scatter(x=df1.index, y=df1['ISB_26'], opacity=0.5, line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dot'), name="Span B", legendgroup = "Ichimoku")


        #plot foward spans

        span_a_fw = go.Scatter(x=df_fw.index, y=df_fw['ISA_9'], opacity=0.5, line=dict(color='rgba(0,255,0,1)', width=1, dash='dot'), name="Span A FW", legendgroup = "Ichimoku") 
        span_b_fw = go.Scatter(x=df_fw.index, y=df_fw['ISB_26'], opacity=0.5, line=dict(color='rgba(255,0,0,1)', width=1, dash='dot'), name="Span B FW", legendgroup = "Ichimoku")

        #add each plot to the figure on Subplot 1 (top)
        fig.add_trace(candle, row=row, col=col)
        fig.add_trace(baseline, row=row, col=col)
        fig.add_trace(conversion, row=row, col=col)
        fig.add_trace(lagging, row=row, col=col)
        fig.add_trace(span_a, row=row, col=col)
        fig.add_trace(span_b, row=row, col=col)
        fig.add_trace(span_a_fw, row=row, col=col)
        fig.add_trace(span_b_fw, row=row, col=col)


        #add titles and axis labels
        
        fig.update_yaxes(title="Price in $", row=row, col=col)
        fig.update_layout(xaxis_rangeslider_visible=False)
        

        return fig

    
    def plot_volume(self, fig, row, col):
        """
        Retrieves Volume Data and plots it
        
        keys:
            forward=True  plot the projected spans [NOT yet implemented]
            
            timespan=True  plot all data vs limits given [NOT yet implemented]

        Args:
            fig - subplot figure to plot to

            row - row to plot to 
            
        
        """ 
        #add  data to the dataframe
        self.df.ta.obv(append=True)

        #generate plots for OBV and volume
        obv = go.Scatter(x=self.df.index, y=self.df['OBV'], line=dict(color='rgba(0,0,255,1)', width=1), name="OBV", legendgroup = "Volume", legendgrouptitle_text="Volume Plot")
        #vol = go.Scatter(x=self.df.index, y=self.df['Volume'], line=dict(color='rgba(0,255,1)', width=1), name="Vol")

        #plot volume and OBV
        #fig.add_trace(vol, row=row, col=col)
        fig.add_trace(obv, row=row, col=col)

        #add titles and axis labels
      
        fig.update_yaxes(title="OBV", row=row, col=col)


        return fig

    
    def plot_sma(self, fig, row, col):
        """
        generate simple moving averages plot with 50 and 200 day intervals
        """
       

        #SMA Plot for 50 and 200 day SMA

        self.df.ta.sma(50, append=True)
        self.df.ta.sma(200, append=True)

        #get the nicer looking trending candle sticks
        hadf = ta.ha(open_= self.df.Open, close = self.df.Close, high=self.df.High, low=self.df.Low)

        #create markers for crosses
       
        signals = pd.DataFrame(index=self.df.index)
        signals['signal'] = np.where(self.df['SMA_50'] > self.df['SMA_200'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
    


        #create plot objects
        SMA50 = go.Scatter(x=self.df.index, y=self.df['SMA_50'], line=dict(color='rgba(0,0,255,1)', width=1), name="SMA 50 day", legendgroup = "SMA" , legendgrouptitle_text="SMA Plot")
        SMA200 = go.Scatter(x=self.df.index, y=self.df['SMA_200'], line=dict(color='rgba(255,0,0,1)', width=1), name="SMA 200 day", legendgroup = "SMA")
        candles = go.Candlestick(x = hadf.index, open = hadf.HA_open, high= hadf.HA_high, low=hadf.HA_low, close= hadf.HA_close, name='Candles', legendgroup = "SMA")
        buys = go.Scatter(x=signals.loc[signals.positions == 1.0].index, y=self.df.SMA_50.loc[signals.positions == 1.0], mode='markers', marker=dict(color='rgba(0,255,0,.8)', size=20), name='Buys - Golden', legendgroup='SMA') 
        sells= go.Scatter(x=signals.loc[signals.positions == -1.0].index, y=self.df.SMA_50.loc[signals.positions == -1.0], mode='markers', marker=dict(color='rgba(255,0,0,.8)', size=20), name='Sells - Death', legendgroup='SMA') 


        #Add traces to plot
        fig.add_trace(SMA50, row=row, col=col)
        fig.add_trace(SMA200, row=row, col=col)
        fig.add_trace(candles, row=row, col=col)
        fig.add_trace(buys, row=row, col=col)
        fig.add_trace(sells, row=row, col=col)
        
       
        #formatting, nuke candlestick's range slider
        fig.update_yaxes(title="$", row=row, col=col)
        fig.update_layout(xaxis_rangeslider_visible=False)


        #return fig
        return fig


    def plot_macd(self, fig, row, col):
        """
        generate MACD plot with historgram

        Default Inputs: fast=12, slow=26, signal=9

        EMA = Exponential Moving Average

        MACD = EMA(close, fast) - EMA(close, slow)

        Signal = EMA(MACD, signal)

        Histogram = MACD - Signal

        """

        #Generate data for MACD

        self.df.ta.macd(append=True)

        barcol = np.where(self.df['MACDh_12_26_9'] >= 0, 'green' , 'red')


        #create plot objects
        MACD = go.Scatter(x=self.df.index, y=self.df['MACD_12_26_9'], line=dict(color='rgba(0,0,255,1)', width=1), name="MACD", legendgroup = "MACD", legendgrouptitle_text="MACD Plot")
        signal = go.Scatter(x=self.df.index, y=self.df['MACDs_12_26_9'], line=dict(color='rgba(255,165,0,1)', width=1), name="Signal", legendgroup = "MACD")
        histogram = go.Bar(x=self.df.index, y=self.df['MACDh_12_26_9'], marker={"color": barcol }, name="Histogram", legendgroup = "MACD")



        #Add traces to plot
        fig.add_trace(MACD, row=row, col=col)
        fig.add_trace(signal, row=row, col=col)
        fig.add_trace(histogram, row=row, col=col)
        fig.add_hline(y=0, line=dict(color='rgba(0,0,0,200.75)', width=1, dash='dot'), name="baseline", row=row, col=col)


        #formatting, nuke candlestick's range slider
        fig.update_yaxes(title="MACD", row=row, col=col)
        


        #return fig
        return fig


    def plot_bollinger(self, fig, row, col):
        """
        generate Bollinger Bands 

        Default Inputs:

        length=5, std=2, mamode="sma", ddof=0

        EMA = Exponential Moving Average

        SMA = Simple Moving Average

        STDEV = Standard Deviation

        stdev = STDEV(close, length, ddof)

        if "ema":   MID = EMA(close, length)

        else:   MID = SMA(close, length)

        LOWER = MID - std * stdev

        UPPER = MID + std * stdev


        generates: lower, mid, upper, bandwidth, and percent columns.
        """

        #Generate data for Bollinger Bands
        self.df.ta.bbands(append=True)

        #get the nicer looking trending candle sticks
        hadf = ta.ha(open_= self.df.Open, close = self.df.Close, high=self.df.High, low=self.df.Low)


        #create plot objects
        BBM = go.Scatter(x=self.df.index, y=self.df['BBM_5_2.0'], line=dict(color='rgba(0,0,255,.50)', width=1), name="Mid", legendgroup = "Bollinger", legendgrouptitle_text="Bollinger Bands")

        BBL = go.Scatter(x=self.df.index, y=self.df['BBL_5_2.0'], line=dict(color='rgba(255,0,0,1)', width=1, dash='dot'), name="Lower", legendgroup = "Bollinger")
        BBU = go.Scatter(x=self.df.index, y=self.df['BBU_5_2.0'], line=dict(color='rgba(0,255,0,1)', width=1, dash='dot'), name="Upper", fill='tonexty', fillcolor='rgba(0,0,255,.05)', legendgroup = "Bollinger")


        candles = go.Candlestick(x = hadf.index, open = hadf.HA_open, high= hadf.HA_high, low=hadf.HA_low, close= hadf.HA_close, name='Candles', legendgroup = "Bollinger")



        #Add traces to plot
        fig.add_trace(candles, row=row, col=col)
        fig.add_trace(BBL, row=row, col=col)
        fig.add_trace(BBU, row=row, col=col)
        fig.add_trace(BBM, row=row, col=col)


        #formatting, nuke candlestick's range slider
        fig.update_yaxes(title="$", row=row, col=col)
        fig.update_layout(xaxis_rangeslider_visible=False)



        #return fig
        return fig


    def multi_plot (self, *args, **kwargs):
        '''
            Plot list of charts based on args given

            OPTIONS:

            "SMA" - a 50 and 200 day SMA

            "MACD" - MACD plot with histogram

            "Ichimoku" - ichimoku plot

            "RSI" - an RSI plot with windows at 70/30

            "bollinger" - Bollinger bands with 20 day SMA

            "volume" = volume plot using OBV

            Usage:

            multi_plot("ichimoku", "sma", "RSI", heights=[1, .5, 1])

        '''
       

        print(f"generating plot of {len(args)} items")

        #PARSE OUT chart names
        chart_names = list()
        for i in range(0 , len(args)):
            #print(i + 1 , "   ", args[i])
            chart_names.append(f"{args[i].capitalize()} Plot") 

        #get row hights
        if 'heights' in kwargs:
            heights = kwargs['heights']
        else:
            #if blank
            heights = np.ones(len(args), dtype=int ).tolist()
            
        
        #create plot object of subplots of specificed size etc...
        fig = make_subplots(rows=len(args), cols=1, subplot_titles=chart_names, row_heights=heights, shared_xaxes=True, vertical_spacing= 0.05)

        #generate plots, get the plot name from args, look up function
        for i in range(0 , len(args)):
            plot_type = self.get_plot_func(args[i])
            fig = plot_type(fig, i+1, 1)
        

        #layout 
        fig.update_layout(height=1000, width=1400, showlegend=True, title=f"Plots for {self.symbol}")

     

        #range slider position and buttons
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(count=3,
                            label="3y",
                            step="year",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=False
                )
            )
        )



        #display Plots
        fig.show()




####
####
####


if __name__ == '__main__':
    
    #test_df = FinPlot('AMD', start='2020-01-01', end='2022-01-01')
    test_df = FinPlot('TSLA')
    
    #refresh data
    #test_df.get_data()
    #test_df.save_data('D:/Temp/StockData/') 
   
    #test_df.get_data(default=False, file='D:/Temp/StockData/AMD.csv')
    test_df.get_data(default=False, folder='D:/Temp/StockData/')
       
      
    test_df.multi_plot("bollinger", "rsi")
        


    