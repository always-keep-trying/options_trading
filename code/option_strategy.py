import os
import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yahoo.code.support_functions as sf
import yahoo.code.ETL as ETL
from yahoo.code.ETL import bcolors


class Strategy:
    """Option Strategy with single underlying"""

    def __init__(self, ticker, trade_date):
        self.ticker = ticker
        self.trade_date = trade_date

        self.options_df = None

        self.positions = None

        self.theo_PnL = None
        self.premium = None
        self.max_loss = None
        self.max_profit = None
        self.break_even_points = None

    def load_options_data(self):
        raise NotImplementedError("To be implemented by child class")

    def select_positions(self):
        raise NotImplementedError("To be implemented by child class")

    def calculate_theo_pnl(self):
        raise NotImplementedError("To be implemented by child class")

    def plot_theo_pnl(self):
        raise NotImplementedError("To be implemented by child class")


class Straddle(Strategy):
    def __init__(self, ticker, trade_date, quantity, target_dtox):
        super().__init__(ticker, trade_date)
        self.strike = None
        self.quantity = quantity
        self.dtox = None
        self.expiration = None
        self.target_dtox = target_dtox
        self.load_options_data()
        self.set_dtox()

    def load_options_data(self):
        self.options_df = ETL.fetch_data(ticker=self.ticker, is_option=True, ref_date=self.trade_date)

    def set_dtox(self):
        all_dtox = self.options_df.loc[:, 'DTOX'].drop_duplicates().to_list()
        all_dtox.sort()

        self.dtox = min(all_dtox, key=lambda x: abs(x - self.target_dtox))
        self.expiration = self.options_df.query(f"DTOX == {self.dtox}")['Expiration'].to_list()[0]

    def set_strike(self, strike):
        # we are assuming that the dtox is already set
        all_strikes = self.options_df.query(f"DTOX == {self.dtox}")['strike'].drop_duplicates().to_list()
        all_strikes.sort()
        if strike in all_strikes:
            self.strike = strike
        else:
            # select the closest strike
            self.strike = min(all_strikes, key=lambda x: abs(x-strike))
            print(f"{bcolors.WARNING}The requested {strike} strike is invalid, "
                  f"updating it to {self.strike}{bcolors.ENDC}\n")

    def set_ATM_strike(self, underlying_close=None):
        atm_set = set()

        all_dtox = self.options_df.loc[:, 'DTOX'].drop_duplicates().to_list()
        all_dtox.sort()
        # use the first 2 tenors to find the ATM strike
        for selected_dtox in all_dtox[0:2]:
            # iterate through both call and put
            for pc_type in ['C', 'P']:
                contract = self.options_df.query(f"DTOX == {str(selected_dtox)} and put_call_code == '{pc_type}'",
                                                 inplace=False).sort_values('strike')
                ind = contract.loc[:, 'inTheMoney'].diff(1).to_list().index(True)
                atm_set.add(contract.iloc[ind, :]['strike'])
                atm_set.add(contract.iloc[ind - 1, :]['strike'])

        if underlying_close:
            self.strike = min(atm_set, key=lambda x: abs(x - underlying_close))
        else:
            self.strike = min(atm_set)
        print(f"ATM Strike of {self.strike} selected \n")

    def select_positions(self):

        opt_subset = self.options_df.query(f"DTOX == {self.dtox}")
        all_strikes_dtox = opt_subset.loc[:, 'strike'].to_list()
        all_strikes_dtox.sort()
        strike_counts = opt_subset.groupby('strike').count()['put_call_code'].reset_index(drop=False).query(
            ' put_call_code == 2 ')
        strike_counts_list = strike_counts.strike.to_list()

        if self.strike in strike_counts_list:
            opt_subset = opt_subset.query(f"strike == {self.strike}")
        else:
            # atm strike selection is modified
            new_ATM_Strike = min(strike_counts_list, key=lambda x: abs(x - self.strike))
            print(
                f"Original ATM strike was {self.strike}, due to lack of listed contracts the new ATM strike is {new_ATM_Strike}")
            opt_subset = opt_subset.query(f"strike == {new_ATM_Strike}")
            self.strike = new_ATM_Strike

        opt_subset = opt_subset.drop(
            columns=['lastTradeDate', 'change', 'percentChange', 'contractSize', 'currency']).reset_index(drop=True)

        opt_subset.loc[opt_subset.put_call_code == 'C', 'Netquantity'] = self.quantity
        opt_subset.loc[opt_subset.put_call_code == 'P', 'Netquantity'] = self.quantity
        self.positions = opt_subset
        # sign of premium is same as net quantity
        self.premium = np.float64(sum(opt_subset.Netquantity * opt_subset.lastPrice))

    def calculate_theo_pnl(self):
        opt_subset = self.options_df.query(f"DTOX == {self.dtox}")
        all_strikes_dtox = opt_subset.loc[:, 'strike'].to_list()
        # theo PnL
        theo_PnL = pd.DataFrame({'strike': all_strikes_dtox, 'PnL': None})

        pnl_list = np.zeros(len(all_strikes_dtox))
        for index, row in self.positions.iterrows():
            strike = row['strike']
            net_quantity = row['Netquantity']
            if row['put_call_code'] == 'C':
                pnl_list = pnl_list + np.array(
                    list(map(lambda x: net_quantity * np.max([x - strike, 0]), all_strikes_dtox)))
            elif row['put_call_code'] == 'P':
                pnl_list = pnl_list + np.array(
                    list(map(lambda x: net_quantity * np.max([strike - x, 0]), all_strikes_dtox)))

        theo_PnL.loc[:, 'PnL'] = np.subtract(pnl_list, self.premium)
        self.theo_PnL = theo_PnL.sort_values('strike', ignore_index=True)

        if self.quantity > 0:
            self.max_profit = np.inf
            self.max_loss = -1*self.premium
        elif self.quantity < 0:
            self.max_profit = -1*self.premium
            self.max_loss = -np.inf

        self.break_even_points = [self.strike - self.premium/abs(net_quantity), self.strike + self.premium/abs(net_quantity)]

    def plot_theo_pnl(self):
        fig_h, ax = plt.subplots(figsize=(8, 5))

        plt.plot(self.theo_PnL.strike, self.theo_PnL.PnL, marker='.', label='Theo PnL')
        y_tick_size = abs(np.subtract(*list(ax.yaxis.get_majorticklocs()[0:2])))
        x_tick_size = abs(np.subtract(*list(ax.xaxis.get_majorticklocs()[0:2])))

        title_string = f"{self.quantity} "
        if self.quantity > 0:
            plt.scatter(self.strike, self.max_loss, marker='x', color='r', label='Max Loss')
            ax.annotate(f"${self.max_loss:,.1f}", xy=(self.strike, self.max_loss),
                        xytext=(self.strike - x_tick_size, self.max_loss),
                        arrowprops=dict(facecolor='red', shrink=0.0001))

            ax.annotate(f"{self.break_even_points[1]:,.1f}", xy=(self.break_even_points[1], 0),
                        xytext=(self.break_even_points[1], y_tick_size),
                        arrowprops=dict(facecolor='black', shrink=0.0001))

            title_string += "Long " + self.__class__.__name__
        elif self.quantity < 0:
            plt.scatter(self.strike, self.max_profit, marker='x', color='r', label='Max Profit')
            ax.annotate(f"${self.max_profit:,.1f}", xy=(self.strike, self.max_profit),
                        xytext=(self.strike - x_tick_size, self.max_profit),
                        arrowprops=dict(facecolor='red', shrink=0.0001))

            ax.annotate(f"{self.break_even_points[1]:,.1f}", xy=(self.break_even_points[1], 0),
                        xytext=(self.break_even_points[1], -1*y_tick_size),
                        arrowprops=dict(facecolor='black', shrink=0.0001))

            title_string += "Short " + self.__class__.__name__

        plt.scatter(self.break_even_points, [0, 0], marker='^', color='k', label='Break-even')



        ax.yaxis.set_major_formatter('${x:1.1f}')

        plt.ylabel("PnL")
        plt.xlabel('Strike')

        plt.title(title_string + f", ticker: '{self.ticker}', {self.expiration}(DTOX:{self.dtox})")
        plt.legend()
        plt.grid()
        plt.show()

    def build(self):
        if self.strike is None:
            # if no strike was selected, use ATM
            self.set_ATM_strike()
        self.select_positions()
        self.calculate_theo_pnl()
        print(self)

    def __str__(self):
        p_string = f"Ticker: '{self.ticker}', Exp: {self.expiration}, Dtox: {self.dtox}, Premium: {self.premium: .2f} \n"
        p_string += f"Strike: {self.strike}, Break even points: {self.break_even_points} \n"
        p_string += f"Max Profit: {self.max_profit: .2f}, Max Loss: {self.max_loss: .2f} \n"
        return p_string

    def __repr__(self):
        r_string = f"Straddle(ticker='{self.ticker}', trade_date='{self.trade_date}', "
        r_string += f"quantity={self.quantity}, target_dtox={self.target_dtox})"
        return r_string
