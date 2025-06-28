import os
import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import yahoo.src.support_functions as sf
import yahoo.src.ETL as ETL
from yahoo.src.ETL import bcolors

r_arrow_p = dict(edgecolor='red', shrink=0.005, headwidth=4, width=1, fill=False)
k_arrow_p = dict(edgecolor='black', shrink=0.005, headwidth=4, width=1, fill=False)


class Strategy:
    """Option Strategy with single underlying"""

    def __init__(self, ticker, trade_date, quantity, target_dtox):
        self.ticker = ticker

        if isinstance(trade_date, str):
            trade_date = datetime.datetime.strptime(trade_date, '%Y-%m-%d')
        self.trade_date = trade_date
        self.quantity = quantity
        self.options_df = None

        self.target_dtox = target_dtox
        self.dtox = None
        self.expiration = None

        self.positions = None

        self.theo_PnL = None
        self.premium = None
        self.max_loss = None
        self.max_profit = None
        self.break_even_points = None

        self.data_source = None

    def load_options_data(self):
        self.options_df = ETL.fetch_data(ticker=self.ticker, is_option=True, ref_date=self.trade_date)

        if "dte" in self.options_df.columns:
            self.data_source = "ORATS"
        elif "DTOX" in self.options_df.columns:
            self.data_source = "yfinance"
            self.convert_format()
        else:
            raise NotImplementedError("This class is only implimented to fit ORATS data")

    def convert_format(self):
        """ Convert the data fetched from 'yfinance' to match the format of 'ORATS' data """
        self.options_df = self.options_df.rename(
            columns={'DTOX':'dte','lastPrice':'Value','impliedVolatility':'smvVol',
                     'Expiration':'expirDate'}
        )
        self.options_df.loc[:,'ticker'] = self.ticker
        self.option_df.loc[:,'tradeDate'] = self.trade_date
        # Note: yfinance data would need stockPrice column


    def set_dtox(self):
        all_dtox = self.options_df.loc[:, 'dte'].drop_duplicates().to_list()
        all_dtox.sort()
        self.dtox = min(all_dtox, key=lambda x: abs(x - self.target_dtox))
        self.expiration = self.options_df.query(f"dte == {self.dtox}")['expirDate'].to_list()[0]


    def select_positions(self):
        raise NotImplementedError("To be implemented by child class")

    def calculate_theo_pnl(self):
        opt_subset = self.options_df.query(f"dte == {self.dtox}")

        all_strikes_dtox = opt_subset.loc[:, 'strike'].drop_duplicates().to_list()
        all_strikes_dtox.sort()
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

    def plot_theo_pnl(self):
        fig_h, ax = plt.subplots(figsize=(8, 5))
        plt.plot(self.theo_PnL.strike, self.theo_PnL.PnL, marker='.', label='Theo PnL')
        plt.scatter(self.break_even_points, [0, 0], marker='^', color='k', label='Break-even')
        ax.yaxis.set_major_formatter('${x:1.1f}')
        title_string = f"{self.quantity}"
        if self.quantity > 0:
            title_string += "(Long) "
        elif self.quantity < 0:
            title_string += "(Short) "

        title_string += self.__class__.__name__ + \
                        f", ticker: '{self.ticker}', {self.expiration}(DTOX:{self.dtox})"

        if hasattr(self,'strike') and not(hasattr(self,'strikes')):
            title_string += f" Strike:{self.strike}"
        elif hasattr(self,'strikes') and not(hasattr(self,'strike')):
            title_string += f" Strikes:{self.strikes}"

        plt.title(title_string)
        plt.ylabel("PnL")
        plt.xlabel('Strike')
        return fig_h, ax

    def __str__(self):
        p_string = f"Data Source:{self.data_source}, Trade Date: {self.trade_date.date()}, Quantity:{self.quantity} \n"
        p_string += f"Ticker: '{self.ticker}', Exp: {self.expiration}, Dtox: {self.dtox}, Premium: {self.premium: .2f} \n"
        p_string += f"Break even points: {self.break_even_points} \n"
        p_string += f"Max Profit: {self.max_profit: .2f}, Max Loss: {self.max_loss: .2f} \n"
        return p_string

    def __repr__(self):
        r_string = f"{self.__class__.__name__}(ticker='{self.ticker}', trade_date='{self.trade_date}', "
        r_string += f"quantity={self.quantity}, target_dtox={self.target_dtox})"
        return r_string


class Straddle(Strategy):
    def __init__(self, ticker, trade_date, quantity, target_dtox):
        super().__init__(ticker, trade_date, quantity, target_dtox)
        self.strike = None
        self.load_options_data()
        self.set_dtox()

    def set_strike(self, strike):
        # we are assuming that the dtox is already set
        all_strikes = self.options_df.query(f"dte == {self.dtox}")['strike'].drop_duplicates().to_list()
        all_strikes.sort()
        if strike in all_strikes:
            self.strike = strike
        else:
            # select the closest strike
            self.strike = min(all_strikes, key=lambda x: abs(x - strike))
            print(f"{bcolors.WARNING}The requested {strike} strike is invalid, "
                  f"updating it to {self.strike}{bcolors.ENDC}\n")

    def set_ATM_strike(self, underlying_close=None):
        contracts = self.options_df.query(f"dte == {self.dtox}")
        ind = np.argmin(np.abs(contracts['strike'].to_numpy()-contracts['stockPrice'].to_numpy()))
        self.strike = contracts.iloc[ind, :]['strike']

    def select_positions(self):
        try:
            opt_subset = self.options_df.query(f"dte == {self.dtox}")
            opt_subset = opt_subset.query(f"strike == {self.strike}")
        except Exception as e:
            opt_subset = self.options_df.query(f"dte == {self.dtox}")
            strike_counts_list = opt_subset.strike.to_list()
            new_ATM_Strike = min(strike_counts_list, key=lambda x: abs(x - self.strike))
            print(f"Original ATM strike was {self.strike}, due to lack of listed "
                  f"contracts the new ATM strike is {new_ATM_Strike}")
            self.strike = new_ATM_Strike
            opt_subset = opt_subset.query(f"strike == {self.strike}")
        finally:
            assert opt_subset.shape[0] == 2, 'Need 2 contracts to make a straddle. ' \
                                             'num of contracts selected:'+str(opt_subset.shape[0])

        opt_subset.loc[opt_subset.put_call_code == 'C', 'Netquantity'] = self.quantity
        opt_subset.loc[opt_subset.put_call_code == 'P', 'Netquantity'] = self.quantity
        # sign of premium is same as net quantity
        self.premium = 0
        for _, value in opt_subset.iterrows():
            if value['Netquantity'] > 0:
                self.premium += value['Netquantity']*value['AskPrice']
            elif value['Netquantity'] < 0:
                self.premium += value['Netquantity'] * value['BidPrice']

        self.positions = opt_subset.reset_index(drop=True)
        assert self.positions.shape[0] == 2, "Straddle has 2 positions"


    def calculate_theo_pnl(self):
        super().calculate_theo_pnl()

        if self.quantity > 0:
            self.max_profit = np.inf
            self.max_loss = -1 * self.premium
        elif self.quantity < 0:
            self.max_profit = -1 * self.premium
            self.max_loss = -np.inf

        self.break_even_points = [self.strike - self.premium / abs(self.quantity),
                                  self.strike + self.premium / abs(self.quantity)]

    def plot_theo_pnl(self,show = True):
        fig_h, ax = super().plot_theo_pnl()

        y_tick_size = abs(np.subtract(*list(ax.yaxis.get_majorticklocs()[0:2])))
        x_tick_size = abs(np.subtract(*list(ax.xaxis.get_majorticklocs()[0:2])))

        title_string = f"{self.quantity} "
        if self.quantity > 0:
            plt.scatter(self.strike, self.max_loss, marker='x', color='r', label='Max Loss')
            ax.annotate(f"${self.max_loss:,.1f}",
                        xy=(self.strike, self.max_loss),
                        xytext=(self.strike - x_tick_size/2, self.max_loss),
                        arrowprops=r_arrow_p)

        elif self.quantity < 0:
            plt.scatter(self.strike, self.max_profit, marker='x', color='r', label='Max Profit')
            ax.annotate(f"${self.max_profit:,.1f}",
                        xy=(self.strike, self.max_profit),
                        xytext=(self.strike - x_tick_size/2, self.max_profit),
                        arrowprops=r_arrow_p)
        # Annotate break even points
        for be_point in self.break_even_points:
            ax.annotate(f"{be_point:,.1f}",
                        xy=(be_point, 0),
                        xytext=(be_point, np.sign(self.quantity) * y_tick_size),
                        arrowprops=k_arrow_p, horizontalalignment='center')

        plt.legend()
        plt.grid()
        if show:
            plt.show()
        else:
            plt.close()
        return fig_h, ax

    def build(self):
        if self.strike is None:
            # if no strike was selected, use ATM
            self.set_ATM_strike()
        self.select_positions()
        self.calculate_theo_pnl()
        print(self)

    def __str__(self):
        return super().__str__() + f"Strike: {self.strike}"


class Strangle(Strategy):
    def __init__(self, ticker, trade_date, quantity, target_dtox):
        super().__init__(ticker, trade_date, quantity, target_dtox)
        self.strikes = None
        self.load_options_data()
        self.set_dtox()

    def set_strikes(self, strikes):
        # we are assuming that the dtox is already set
        all_strikes = self.options_df.query(f"dte == {self.dtox}")['strike'].drop_duplicates().to_list()
        all_strikes.sort()
        if set(strikes).issubset(set(all_strikes)):
            self.strikes = strikes
        else:
            # select the closest strike
            self.strikes = [min(all_strikes, key=lambda x: abs(x - a)) for a in strikes]
            print(f"{bcolors.WARNING}The requested {strikes} strikes were invalid, "
                  f"updating it to {self.strikes}{bcolors.ENDC}\n")

    def select_positions(self):
        opt_subset = self.options_df.query(f"dte == {self.dtox}")

        assert set(self.strikes).issubset(set(opt_subset['strike'].to_list())), "Strike selected are not avaiable!"
        opt_subset = opt_subset.loc[opt_subset['strike'].isin(self.strikes)]

        # upper strike is the call, shorter strike is the put
        opt_subset = pd.concat(
            [opt_subset.query(f"put_call_code == 'P' and strike == {min(self.strikes)}"),
             opt_subset.query(f"put_call_code == 'C' and strike == {max(self.strikes)}")],
            ignore_index=True)

        opt_subset.loc[:, 'Netquantity'] = self.quantity
        self.positions = opt_subset
        # sign of premium is same as net quantity
        self.premium = np.float64(sum(opt_subset.Netquantity * opt_subset.Value))
        assert self.positions.shape[0] == 2, "Strangle has 2 positions"

    def calculate_theo_pnl(self):
        super().calculate_theo_pnl()

        if self.quantity > 0:
            self.max_profit = np.inf
            self.max_loss = (-1 * self.premium)
        elif self.quantity < 0:
            self.max_profit = (-1 * self.premium)
            self.max_loss = -np.inf

        self.break_even_points = [min(self.strikes) - abs(self.premium/self.quantity),
                                  max(self.strikes) + abs(self.premium / self.quantity)]

    def plot_theo_pnl(self, show=True):
        fig_h, ax = super().plot_theo_pnl()

        y_tick_size = abs(np.subtract(*list(ax.yaxis.get_majorticklocs()[0:2])))
        x_tick_size = abs(np.subtract(*list(ax.xaxis.get_majorticklocs()[0:2])))

        if self.quantity > 0:
            x_point = np.min(self.strikes)
            plt.scatter(x_point, self.max_loss, marker='x', color='r', label='Max Loss')
            ax.annotate(f"${self.max_loss:,.1f}",
                        xy=(x_point, self.max_loss),
                        xytext=(x_point - x_tick_size/2, self.max_loss),
                        arrowprops=r_arrow_p)
        elif self.quantity < 0:
            x_point = np.min(self.strikes)
            plt.scatter(x_point, self.max_profit, marker='x', color='r', label='Max Profit')
            ax.annotate(f"${self.max_profit:,.1f}",
                        xy=(x_point, self.max_profit),
                        xytext=(x_point + x_tick_size/2, self.max_profit),
                        arrowprops=r_arrow_p)

        for be_point in self.break_even_points:
            ax.annotate(f"{be_point:,.1f}",
                        xy=(be_point, 0),
                        xytext=(be_point, np.sign(self.quantity) * y_tick_size),
                        arrowprops=k_arrow_p, horizontalalignment='center')
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        else:
            plt.close()
        return fig_h, ax

    def build(self):
        assert self.strikes is not None, "Please use set_strikes() to set strikes"
        self.select_positions()
        self.calculate_theo_pnl()
        print(self)

    def __str__(self):
        return super().__str__() + f"Strikes: {self.strikes}"


# Notes: There are other ways to implement the option strategies. One way is to start with the definitions
# of just call and put and use the __add__() or __sub__() magic method to construct any strategies.
# However, since I am currently exploring Straddle and Strangle, the current implementation works for my purpose.
# I may revisit this and re-implementation down the line




