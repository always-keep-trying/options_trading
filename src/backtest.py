import os
import pandas as pd
import numpy as np

import yahoo.src.ORATS_data as od


BACKTEST_PATH = os.path.join(os.environ['PYTHONPATH'],'yahoo','backtest')
assert os.path.exists(BACKTEST_PATH), 'backtest path does not exist!'


def backtest_file_name(ticker: str, exp_dt: str, strike_val: float | int) -> str:
    """
    Method that defines path of a file containing one set(put and call) of contract.
    Data source: ORATS
    Args:
        ticker: Symbol
        exp_dt: Expiration date
        strike_val: Strike

    Returns:
        path of the parquet file
    """
    ticker_folder = os.path.join(BACKTEST_PATH, ticker)
    if not os.path.exists(ticker_folder):
        os.mkdir(ticker_folder)

    strike_str = str(strike_val)
    if '.' in strike_str:
        # decimal strike
        whole_num, decimal_val = strike_str.split('.')

        if decimal_val == '0':
            # drop the trailing zero
            strike_str = whole_num
        else:
            strike_str = whole_num + "-" + decimal_val

    return os.path.join(ticker_folder, f"{ticker}_{exp_dt}_{strike_str}.parquet")


def extract_contract_data(ticker:str , exp_dt: str, strike_val: float | int) -> pd.DataFrame:
    """
    If the data file exists, reads the data from 'backtest' folder.
    If the data file doens't exist, fetch the data and save it in the 'backtest' folder and read the data.
    Args:
        ticker: Symbol
        exp_dt: Expiration date
        strike_val: Strike

    Returns:
        DataFrame of the contract data (both put and call).
    """
    file_name = backtest_file_name(ticker, exp_dt, strike_val)

    if not os.path.exists(file_name):
        df = od.get_his_strikes_OPRA(ticker=ticker, exp_date=exp_dt, strike=strike_val)
        df_fmt = od.format_ORATS_data(df)
        df_fmt.to_parquet(file_name)
        print(f"File Saved: {file_name}")
    return pd.read_parquet(file_name)


def extract_positions(position_df: pd.DataFrame) -> None:
    """
    Get contracts relavant to positions
    Args:
        position_df: Position DataFrame

    Returns:
        datafroame of all contracts history
    """
    posn_df = position_df.loc[:, ['ticker', 'expirDate', 'strike']].drop_duplicates()
    posn_list = posn_df.values.tolist()

    for one_posn in posn_list:
        extract_contract_data(*one_posn)

    print(posn_df)
    print("Positions processed")

def calculat_payoff(df, is_call=True):
    if is_call:
        # call payoff
        payoff = list(map(lambda x: np.max([x,0]), df['stockPrice'].to_numpy()-df['strike'].to_numpy()))
    else:
        # put payoff
        payoff = list(map(lambda x: np.max([x,0]), df['strike'].to_numpy()-df['stockPrice'].to_numpy()))
    return payoff


class Backtest:
    def __init__(self, positon_df):
        self.position_df = positon_df
        assert 'Netquantity' in self.position_df.columns, 'Please define Netquantity column'
        # one business date
        self.trade_date = self.position_df['tradeDate'].drop_duplicates().squeeze()

        self.symbols_list = self.position_df['ticker'].drop_duplicates().to_list()
        self.contract_hist = {}
        self.get_contract_hist()

        self.portfolio_payoff = None
        self.portfolio_value = None
        self.underlying_price_hist = None


    def get_contract_hist(self):
        # run ETL through all position
        extract_positions(self.position_df)

        for symb in self.symbols_list:

            symb_posn_df = self.position_df.query(f"ticker == '{symb}'").drop(columns='put_call_code')
            symb_posn_df = symb_posn_df.loc[:,['ticker', 'tradeDate', 'expirDate', 'dte', 'strike', 'stockPrice']].drop_duplicates()
            assert symb_posn_df.shape[0] == 1, 'Processing one contract at a time'
            expiration, strike = symb_posn_df[['expirDate', 'strike']].squeeze().values
            contract_hist = extract_contract_data(ticker=symb, exp_dt=expiration, strike_val=strike)
            # each contract_hist will contain both put and call data
            self.contract_hist[symb] = contract_hist.loc[contract_hist.tradeDate>=self.trade_date,:].reset_index(drop=True)


    def backtest(self):
        portfolio_payoff = pd.DataFrame()
        portfolio_value = pd.DataFrame()

        underlying_price_hist = pd.DataFrame()
        for symb in self.symbols_list:
            contract_hist = self.contract_hist[symb]
            for _, row in self.position_df.query(f"ticker == '{symb}'").iterrows():
                # looking forward, does not include the date the trade was entered
                curr_posn = contract_hist.loc[contract_hist.tradeDate > self.trade_date, :].copy()
                curr_posn = curr_posn.merge(
                    row.to_frame().T.loc[:, ['ticker', 'expirDate', 'strike', 'put_call_code', 'Netquantity']])
                curr_posn = curr_posn.sort_values('dte', ascending=False).reset_index(drop=True)
                curr_posn.loc[:, 'Payoff'] = calculat_payoff(curr_posn, is_call=(row['put_call_code'] == 'C'))
                curr_posn.loc[:, 'Net_Payoff'] = curr_posn['Payoff'] * np.float64(curr_posn.Netquantity)
                curr_posn.loc[:, 'Net_Value'] = curr_posn['Value'] * np.float64(curr_posn.Netquantity)
                #curr_posn = curr_posn.set_index('tradeDate')
                contract_detail = "_".join(map(str, row[['ticker', 'expirDate', 'put_call_code']].values)) + '_' + str(
                    row['strike']).replace('.0', '').replace('.', '-')
                print(contract_detail)

                if portfolio_payoff.empty:
                    portfolio_payoff = curr_posn.loc[:, ['tradeDate','Net_Payoff']].rename(columns={'Net_Payoff': contract_detail})
                    portfolio_value = curr_posn.loc[:, ['tradeDate','Net_Value']].rename(columns={'Net_Value': contract_detail})
                else:
                    # import pdb
                    # pdb.set_trace()

                    portfolio_payoff = portfolio_payoff.merge(
                        curr_posn.loc[:, ['tradeDate','Net_Payoff']].rename(columns={'Net_Payoff': contract_detail}),
                        how='inner', on='tradeDate')
                    portfolio_value = portfolio_value.merge(
                        curr_posn.loc[:, ['tradeDate','Net_Value']].rename(columns={'Net_Value': contract_detail}),
                        how='inner')


            if underlying_price_hist.empty:
                underlying_price_hist = contract_hist.loc[contract_hist.put_call_code == 'C',['tradeDate','stockPrice']].rename(columns={'stockPrice':symb})
            else:
                underlying_price_hist = underlying_price_hist.merge(
                    contract_hist.loc[contract_hist.put_call_code == 'C',['tradeDate','stockPrice']].rename(columns={'stockPrice':symb}),
                    how='outer', on='tradeDate'
                )


        portfolio_payoff = portfolio_payoff.set_index('tradeDate')
        portfolio_value = portfolio_value.set_index('tradeDate')
        underlying_price_hist = underlying_price_hist.set_index('tradeDate')

        portfolio_payoff.loc[:, 'Payoff'] = portfolio_payoff.values.sum(axis=1)
        portfolio_value.loc[:, 'Value'] = portfolio_value.values.sum(axis=1)

        portfolio_payoff = portfolio_payoff.merge(underlying_price_hist, how='outer', left_index=True, right_index=True)
        portfolio_value = portfolio_value.merge(underlying_price_hist, how='outer',  left_index=True, right_index=True)

        portfolio_payoff.index = pd.to_datetime(portfolio_payoff.index)
        portfolio_value.index = pd.to_datetime(portfolio_value.index)
        underlying_price_hist.index = pd.to_datetime(underlying_price_hist.index)


        for symb in self.symbols_list:
            portfolio_payoff.loc[:, symb+'_Payoff'] = \
                portfolio_payoff.loc[:, [x for x in portfolio_payoff.columns if x.startswith(symb+'_')]].values.sum(1)
            portfolio_value.loc[:, symb + '_Value'] = \
                portfolio_value.loc[:, [x for x in portfolio_value.columns if x.startswith(symb + '_')]].values.sum(1)

        self.portfolio_payoff = portfolio_payoff
        self.portfolio_value = portfolio_value
        self.underlying_price_hist = underlying_price_hist


    def calculate_time_series(self, symbol, column_name):

        contract_hist = self.contract_hist[symbol]
        contract_hist.loc[:,'Calculate'] = contract_hist.loc[:,column_name]*contract_hist.loc[:, 'Netquantity']

        return contract_hist.groupby(['tradeDate'])['Calculate'].sum().rename(column={'Calculate':'Agg_'+column_name})



