import os
import pandas as pd
import numpy as np

import yahoo.src.ORATS_data as od
import yahoo.src.option_strategy as OptS


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
                curr_posn = contract_hist.loc[contract_hist.tradeDate >= self.trade_date, :].copy()
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

        # aggregate per symbol level results
        if len(self.symbols_list)>1:
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


def process_order(order_dataframe: pd.DataFrame, include_contracts=True)->tuple:
    """
    Baed on the dataframe of orders, backtest the strategy
    Args:
        order_dataframe: Dataframe of order that is to be backtested
        include_contracts: optionally to include contract history as part of the output

    Returns:
        tuple of at most 3 objects in the following order
        1) order results in a dataframe
        2) Value historty in a dataframe
        3) [Optional] All Contract history in a dataframe

    """
    # TODO: add limit order functionality

    order_results = order_dataframe.copy()
    order_results.loc[:, 'premium'] = np.nan
    order_results.loc[:, 'dtox'] = np.nan
    order_results.loc[:, 'expiration'] = ''

    order_results.loc[:,'Contracts'] = np.nan
    order_results = order_results.astype({'Contracts':'object'})

    all_contracts = []
    all_values = []
    all_underlying = []
    for i, row_val in order_dataframe.iterrows():
        trade_date = row_val['trade_date']
        ticker = row_val['ticker']
        strat_type = row_val['type']
        quantity = row_val['Quantity']
        target_dtox = row_val['target_dtox']

        strategy_obj = getattr(OptS, strat_type)(ticker=ticker, trade_date=trade_date, quantity=quantity,
                                                 target_dtox=target_dtox)
        strategy_obj.build()
        #fig_h, ax_h = strategy_obj.plot_theo_pnl(show=False)

        order_results.loc[i, 'premium'] = strategy_obj.premium
        order_results.loc[i, 'dtox'] = strategy_obj.dtox
        order_results.loc[i, 'expiration'] = strategy_obj.expiration

        backtst_obj = Backtest(strategy_obj.positions)
        backtst_obj.backtest()
        value_df = backtst_obj.portfolio_value
        # underlying
        all_underlying.append(backtst_obj.underlying_price_hist)
        # contracts
        contract_cols = [x for x in value_df.columns if x.startswith(ticker + '_')]
        if include_contracts:
            all_contracts.append(value_df.loc[:, contract_cols])
        order_results.loc[i, 'Contracts'] = [[x] for x in contract_cols]
        # Value
        all_values.append(value_df.loc[:, ['Value']].rename(columns={'Value': ticker+'_'+trade_date}))

    # format for output dataframes
    if len(all_values)>1:
        value_hist = pd.merge(*all_values, how='outer', left_index=True, right_index=True)
    else:
        value_hist = all_values[0]
    value_hist.loc[:, 'Total'] = np.nansum(value_hist.values, axis=1)

    # add PnL
    value_hist.loc[:, 'DoD_Change'] = value_hist['Total'].diff(1)
    value_hist.loc[:, 'PnL'] = value_hist['DoD_Change'].cumsum()


    if len(all_underlying)>1:
        underling_df = pd.merge(*all_underlying, how='outer', left_index=True, right_index=True)
    else:
        underling_df = all_underlying[0]
    value_hist = pd.merge(value_hist,underling_df, how='outer', left_index=True, right_index=True)

    if len(all_contracts) > 1:
        contract_hist = pd.merge(*all_contracts, how='outer', left_index=True, right_index=True)
    else:
        contract_hist = all_contracts[0]
    contract_hist.loc[:, 'Total'] = np.nansum(contract_hist.values, axis=1)
    contract_hist = pd.merge(contract_hist, underling_df, how='outer', left_index=True, right_index=True)

    # cutt off the data till the minimum expiration
    exp_date_df = order_results.groupby('trade_date')['dtox'].min().to_frame().reset_index(drop=False).merge(
        order_results[['trade_date', 'dtox', 'expiration']]).drop_duplicates()
    contract_hist = contract_hist.loc[:exp_date_df['expiration'].squeeze(), :]
    value_hist = value_hist.loc[:exp_date_df['expiration'].squeeze(), :]

    if include_contracts:
        return order_results, value_hist, contract_hist
    else:
        return order_results, value_hist


def summarize_trade(df_order, df_value, df_contract, plot_ratio):
    # due to data issues some data are missing, needs to drop them
    na_vals = df_contract.isna().any(axis=1).to_frame().rename(columns={0: 'Null'})
    na_vals = na_vals.loc[na_vals['Null'], :]
    if not na_vals.empty:
        reduction = na_vals.shape[0]/df_value.shape[0]
        df_value = df_value.loc[~df_value.index.isin(na_vals.index),:]
        df_contract = df_contract.loc[~df_contract.index.isin(na_vals.index),:]
        plot_ratio = plot_ratio.loc[~plot_ratio.index.isin(na_vals.index),:]
        print(f"NA removal reduction: {np.round(reduction*100,1)}%")

    dt_start = df_value.index.min()
    dt_end = df_value.index.max()
    subset_plot_df = plot_ratio.loc[(plot_ratio.index >= dt_start) & (plot_ratio.index <= dt_end), :]


    # first date the z-score goes below zero or to the min
    min_z_score = np.min(subset_plot_df['Z_Score'].to_numpy())
    if min_z_score>0:
        zero_zscore_dt = subset_plot_df.index[subset_plot_df['Z_Score'].to_numpy() == min_z_score][0]
    else:
        zero_zscore_dt = subset_plot_df.index[np.argmax(subset_plot_df['Z_Score'].to_numpy() < 0)]

    result_dict = {'PnL_exp': df_value.iloc[-1, :]['PnL']}
    all_ticker = df_order['ticker'].drop_duplicates().to_list()
    for sym in all_ticker:
        result_dict[f"pct_{sym}_exp"] = (df_value.iloc[-1, :][sym] / df_value.iloc[0, :][sym]) - 1
    result_dict['zero_zscore_dt'] = zero_zscore_dt
    result_dict['PnL_zero_zscore'] = df_value.loc[zero_zscore_dt, 'PnL']
    for sym in all_ticker:
        result_dict[f"pct_{sym}_zero_zscore"] = (df_value.loc[zero_zscore_dt, sym] / df_value.iloc[0, :][sym]) - 1,
    result_df = pd.DataFrame(result_dict, index=[df_order.trade_date.to_list()[0]])
    result_df.index.name = 'trade_date'

    assert not(result_df.isna().values.any()), 'Summary df can not have any NA values'

    return result_df