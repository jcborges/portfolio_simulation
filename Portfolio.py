__author__ = "Julio Borges"
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math


class Portfolio():
    def __init__(self, tickers: list, weights: list, start_investment: float, start=None, end=None, period = "10y"):
        assert(len(tickers) == len(weights))
        assert(np.sum(weights) == 1)
        self.assets = []
        self.weights = weights
        self.start_investment = start_investment
        self.tickers = tickers
        self.n = len(tickers)
        self.dividends = dict()

        for ticker in tickers:
            if start and end:
                asset = yf.Ticker(ticker).history(start=start, end=end)
            else:
                asset = yf.Ticker(ticker).history(period=period)
            self.assets.append(asset)
        self._calibrate_dates()
        self._calculate_values()


    def _calibrate_dates(self):
        common_dates = set.intersection(*[set(asset.index.unique()) for asset in self.assets])
        self.assets = [asset[asset.index.map(lambda x: x in common_dates)] for asset in self.assets]


    def _calculate_asset_development(self, idx, weighted = True):
        values = []
        if weighted:
            stocks = self.start_investment * self.weights[idx] / self.assets[idx]["Close"].iloc[0]
        else:
            stocks = self.start_investment / self.assets[idx]["Close"].iloc[0]
        for j, row in self.assets[idx].iterrows():
            value = row["Close"] * stocks
            if row["Dividends"] > 0:
                dividends = stocks * row["Dividends"]
                stocks = stocks + (dividends / row["Close"])
                try:
                    self.dividends[j] += dividends
                except:
                    self.dividends[j] = dividends
            values.append(value)
        return pd.Series(values, index=self.assets[idx].index)


    def _calculate_values(self):
        all_values = []
        for i in range(self.n):
            ss = self._calculate_asset_development(i)
            all_values.append(ss)
        valuesmat = np.array(all_values).sum(axis=0)
        self.hist = pd.Series(valuesmat, index=ss.index)
        self.values = all_values


    def plot(self, hline=True):
        """
        Plot portfolio development
        :param hline: Show horizontal line with initial investment
        """
        self._set_heat_params()
        self.hist.plot(label = "Portfolio", title = "Portfolio Development")
        plt.hlines(self.start_investment, self.hist.index.min(), self.hist.index.max(), colors="r")
        plt.grid()


    def plot_comparison(self):
        """
        Plot portfolio development against development of single stocks
        """
        for i in range(self.n):
            ss = self._calculate_asset_development(i, False)
            ss.plot(label = self.tickers[i])
        self.hist.plot(label="Portfolio", title = "Simulation of Portfolio vs. 100%% Initial Investment on individual stocks\n Simulated Investment = %.2f" % self.start_investment)
        plt.grid()
        plt.legend()


    def summary(self):
        """
        Print summary information about portfolio development
        """
        win = self.hist.iloc[-1] - self.hist.iloc[0]
        years = (self.hist.index[-1] - self.hist.index[0]).days / 365
        yy = math.pow(self.hist.iloc[-1]/self.hist.iloc[0] , 1/years) -1
        dividends = pd.Series(self.dividends).sum()

        t2 = "Capital Development: from %.2f to %.2f" % (self.hist.iloc[0], self.hist.iloc[-1]) + "\n"
        t3 = "Yield: %.4f" % yy + "\n"
        t4 = "Gain: %.4f; From which Dividends: %.4f; and Asset Price Development: %.4f" % (win, dividends, win-dividends)

        print(t2, t3, t4)


    def generate_saving_plan(self, monthly_investment):
        sparlist = []
        for i in range(self.n):
            df_asset = self.assets[i]
            weight = self.weights[i]
            investment = monthly_investment * weight
            ss = self._calculate_sparplan(df_asset, investment)
            sparlist.append(ss)
        valuesmat = np.array(sparlist).sum(axis=0)
        self.sparhist = pd.Series(valuesmat, index=ss.index)
        mindex = ss.groupby(pd.Grouper(freq='M')).mean().index
        self.spar_dev = pd.Series([monthly_investment*i for i in range(1, len(mindex-1)+1)], index=mindex)


    def _calculate_sparplan(self, df_asset, investment):
        lastmonth = df_asset.index[0].month
        stocks = investment / df_asset["Close"].iloc[0]
        values = []
        for i, row in df_asset.iterrows():
            value = row["Close"] * stocks
            if row["Dividends"] > 0:
                stocks = stocks + ((stocks * row["Dividends"]) / row["Close"])
            if lastmonth != i.month:
                stocks = stocks + (investment / row["Close"])
            lastmonth = i.month
            values.append(value)
        return pd.Series(values, index=df_asset.index)

    def plot_sparplan(self):
        """
        Plots development of a saving plan with a given portfolio. Also shows simulation for a single payment
        porfolio development and total amount invested over time.
        """
        self.sparhist.plot(label="Portfolio Development")
        self.spar_dev.plot(label="Investment")
        self.hist.plot(label="Single Payment Curve")
        plt.grid()
        plt.legend()
