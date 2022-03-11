##################################################
####        FinPorfolio.py
####        Kevin Osicki
####
####
##################################################


from matplotlib.pyplot import annotate
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

import os
from os import listdir
from os.path import isfile, join

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FinPortfolio():

    def __init__(self, symbols, **kwargs):
        self.symbols = symbols
        self.df = pd.DataFrame()
        self.period = "5y"

        self.position = ''  #positions taken in each company for analysis



    def getPortfolio(self):
        """
        Get portfolio from given symbols group, downloads data from Yahoo finance

        Calculates cummulative daily returns for analysis per each stock and saves in dataframe

        """
        print('Getting data from yfinance')
        data = pd.DataFrame
        data = yf.download(self.symbols, group_by='ticker', period=self.period)

        for symbol in self.symbols:
            #self.df[symbol] = data[symbol]['Adj Close']
            #self.df[symbol + "_ret"] = (data[symbol]['Adj Close'] / data[symbol]['Adj Close'].shift(1)) - 1
            
            #return cummulative return for each symbol
            self.df[symbol] = (data[symbol]['Adj Close'] / data[symbol]['Adj Close'].shift(1)) - 1


    def PlotCorrelation(self):
        """
        Calculates and plots the correlation matrix 

        """

        #calulate correlation matrix
        correlation_matrix = self.df.corr(method='pearson')
        
        #new graph object
        fig = go.Figure()

        #create Heatmap of correlation data
        corr = go.Heatmap(z=correlation_matrix, y=correlation_matrix.index, x=correlation_matrix.index, colorscale='RdBu_r')
        
        fig.add_trace(corr)
        
        #annotate cells with data
        for i in range(0, len(correlation_matrix)):
            for j in range(0, len(correlation_matrix)):
                fig.add_annotation(x=i, y=j, text=f"{correlation_matrix.iloc[i,j]: .2f}" ,showarrow=False, font=dict(family="sans serif", size=24,color="Grey"))

        #add plot information
        fig.update_layout(height=1000, width=1400, showlegend=True, title=f"Correlation Matrix for selected stocks over {self.period}")

        #plot object
        fig.show()



#savePortfolio

#performance view

#correlation, some sort of optimizer?

#risk returns



if __name__ == '__main__':

    mySymbols = ["TSLA", "AAPL", "FB", "AMD", "MSFT", "COST", "AMZN", "GOOG"]
    myPortfolio = FinPortfolio(mySymbols)

    myPortfolio.getPortfolio()
    myPortfolio.PlotCorrelation()
    



