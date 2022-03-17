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

    def __init__(self, **kwargs):
        
        self.symbols = kwargs.get("symbols", '')
        self.df = pd.DataFrame()
        self.period = "5y"

        #start and end dates for plotting
        self.start = kwargs.get("start", '')
        self.end = kwargs.get("end", '')

        #positions taken in each company for analysis
        self.weights = kwargs.get("weights", [])



    def get_portfolio(self, **kwargs):
        """
        Get portfolio from given symbols group, downloads data from Yahoo finance

        Calculates cummulative daily returns for analysis per each stock and saves in dataframe

        """

        """
        Retrieves data from file or yfinance call based on supplied criteria
        
        keys:
            default = Boolean, use default settings or not
           
            file = string, full file name to read

        Args:
            () returns default values from yfinance, 5Y, 1D

            (default=False) pulls from supplied time frame or interval

            (file='') reads csv data from file name directly
            

        """ 
        
        
        #check if file location given to CSV file, else load from ta's ticker function from yfinance
        if 'file' in kwargs:
            print('Getting data from CSV')
            # Try to get the file and if it doesn't exist issue a warning
            try:
                self.df = pd.read_csv(kwargs['file'], index_col=0, parse_dates=True)

                #save symbols to self.symbols
                self.symbols = list(self.df.filter(regex='daily_').columns.str.replace("daily_",""))
               

            except FileNotFoundError:
                print("File Doesn't Exist")
            else:
                return self.df
       
        #default is to read directly from yahoo finance via pandas_ta
        else:
            print('Getting data from yfinance')
            
            #check for symbols are given
            if len(self.symbols) == 0:
                print('No symbols given, check portfolio creation or self.symbols')
            
            else:
                    
                #default values check, will use 5y / 1d values, ELSE use given date time frame
                if kwargs.get('default', True) is True:
                    print('Using default values, 5Y and 1d')
                    
                    data = yf.download(self.symbols, group_by='ticker', period=self.period)
                    
                else:
                    data = yf.download(self.symbols, group_by='ticker', start=self.start, end=self.end)
                

                for symbol in self.symbols:
                
                    #return cummulative return for each symbol
                    self.df["price_" + symbol] = data[symbol]['Adj Close']
                    self.df["daily_" + symbol] = (data[symbol]['Adj Close'] / data[symbol]['Adj Close'].shift(1)) - 1



    def save_portfolio (self, folder, **kwargs):
        """
        Saves the porfolio data to a CSV files at a folder location 
        
        keys:
            all=True  save all data or just the     

        Args:
            (foldername) - saves all data to <Folder>/symbol.csv

            (foldername, all=False) - saves only core data to <Folder>/symbol.csv
         
        
        """ 
        
        
        try:
            file = folder + "Portfolio" + '.csv'
            self.df.to_csv(file)
        except Exception as ex:
            print("Couldn't save file")



    def plot_correlation(self):
        """
        Calculates and plots the correlation matrix 

        """
        
        #calulate correlation matrix, scrub daily prefix
        correlation_matrix = self.df.filter(regex='daily_').corr(method='pearson')
        correlation_matrix.columns = correlation_matrix.columns.str.replace("daily_","")
        correlation_matrix.index = correlation_matrix.index.str.replace("daily_","")


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

    def plot_riskbox(self):
        """
        Calculates and plots the box risk approximation via box plots for all stocks' cummulative return

        
        """
                
        #calulate correlation matrix, scrub daily prefix
        riskboxdf = pd.DataFrame()
        riskboxdf = self.df.filter(regex='daily_')
        riskboxdf.columns = riskboxdf.columns.str.replace("daily_","")
        
        
        #new graph object
        fig = go.Figure() 
        
        #generate box plots
        for column in riskboxdf:
            boxret = go.Box(y=riskboxdf[column], name=column) 
            fig.add_trace(boxret)

        #add plot information
        fig.update_layout(height=1000, width=1400, showlegend=True, title=f"risk box plot for stocks over {self.period}")


        fig.show()


    def plot_cummulative(self):
        """
        Calculates and plots the cummulative gains for all stocks

        """

        cummulativedf = (1 + self.df.filter(regex='daily_')).cumprod()
        cummulativedf.columns = cummulativedf.columns.str.replace("daily_","")

        #new graph object
        fig = go.Figure() 
        
        #generate box plots
        for column in cummulativedf:
            cummret = go.Scatter(x=cummulativedf.index, y=cummulativedf[column], name=column) 
            fig.add_trace(cummret)

        #add plot information
        fig.update_layout(height=1000, width=1400, showlegend=True, title=f"cummulative returns for stocks over {self.period}")


        fig.show()
        

    def PortfolioPerformance(self, **kwargs):
        """
        WORKING ON THIS SECTION! 

        Calculates and plots the cummulative gains for all stocks
       

        """
       

        returns = self.df.filter(regex='price_').pct_change()
        returns.columns = returns.columns.str.replace('price_', '')
        meanreturns = returns.mean()
        covMatrix = returns.cov()

        #get weights (just making an even distro for now)
        self.weights = np.ones(len(returns.columns)) * 1/len(returns.columns)
        print(self.weights)


        #applies returns vs weights over trading period
        weightedreturns = np.sum(meanreturns*self.weights)*252
        
        #calc porfolio variance
        std = np.sqrt( np.dot(self.weights.T, np.dot(covMatrix, self.weights))) * np.sqrt(252)

        print(std)




#performance view

#some sort of optimizer?

#risk returns



if __name__ == '__main__':

    #mySymbols = ["TSLA", "AAPL", "FB", "AMD", "MSFT", "COST", "AMZN", "GOOG"]
    #myPortfolio = FinPortfolio(symbols=mySymbols, start='2020-01-01', end='2022-01-01')
    myPortfolio = FinPortfolio()

    #myPortfolio.get_portfolio(default=True)  
    #myPortfolio.save_portfolio('D:/Temp/StockData/')    

    myPortfolio.get_portfolio(file="D:/Temp/StockData/Portfolio.csv")
    #myPortfolio.plot_correlation()
    myPortfolio.plot_riskbox()
    #myPortfolio.plot_cummulative()

    #myPortfolio.PortfolioPerformance()
