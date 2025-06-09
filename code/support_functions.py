import pandas as pd
import numpy as np

DAYS_IN_YEAR = 365
period_dic = {
    '1d': 1,
    '5d': 5,
    '1mo': round(DAYS_IN_YEAR / 12),
    '3mo': round(3 * DAYS_IN_YEAR / 12),
    '6mo': round(6 * DAYS_IN_YEAR / 12),
    '1y': DAYS_IN_YEAR,
    '2y': 2 * DAYS_IN_YEAR,
    '5y': 5 * DAYS_IN_YEAR,
    '10y': 10 * DAYS_IN_YEAR
}
IVOL_FLOOR = np.float64(1e-4)


def clean_data(function):
    """ Decorator used to clean data, only applied when the function rertuns a DataFrame"""
    def filter_data(data_frame):
        # remove un-wanted data
        data_frame = data_frame.loc[data_frame.openInterest != 0, :]
        data_frame = data_frame.loc[data_frame.ask > data_frame.bid, :]
        data_frame = data_frame.loc[data_frame.impliedVolatility > IVOL_FLOOR, :]
        cols_to_sort = ['strike']
        if 'Expiration' in data_frame.columns:
            data_frame.loc[:, 'dtox'] = (
                    data_frame.loc[:, 'Expiration'].map(lambda x: pd.to_datetime(x).date())
                    - data_frame.loc[:, 'lastTradeDate'].map(lambda x: x.date())
            ).map(lambda x: x.days)
            print('decorator')
            data_frame = data_frame.loc[data_frame.dtox > 0, :]
            cols_to_sort.append('Expiration')
        data_frame = data_frame.sort_values(cols_to_sort).reset_index(drop=True)
        return data_frame

    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            result = filter_data(result)
        return result

    return wrapper


def select_expiration(period_str, options, last_date):
    """
    Using the yfinance's Options data, identify the options expiration date that is closest to the selected period.

    Args:
        period_str (str): 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
        options (yfinance.ticker.Options): Options data from yfinance
        last_date (Timestamp): Last businessdate

    Returns:
        exp_selected (str): String of expiration date using the "YYYY-MM-DD" format.
        dtox_selected (int): Days to expiration.
    """
    ref_dtox = period_dic[period_str]

    dtox = list(map(lambda x: (pd.to_datetime(x).date() - last_date.to_pydatetime().date()).days, options))
    # find the dtox closest to the selected days to exp
    dtox_selected = min(dtox, key=lambda x: abs(x - ref_dtox))
    exp_selected = options[dtox == dtox_selected]
    return exp_selected, dtox_selected


def get_option_chain_tenor(period_str, ticker, last_date, underlying_last_close, IS_CALL=True):
    """
    Using the yfinance's Options data, fetch options data with the expiration date that is closest to the selected period.
    The function can add a log moneyness column if the underlying_last_cose is provided; ln(S/K)
    The following data points are modified
        - zero open interest data are removed
        - Ask prices should be higher than bid price.

    Args:
        period_str (str): 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y
        ticker (yfinance.ticker.Ticker): Ticker from yfinance
        last_date (Timestamp): Last businessdate
        [Optional] underlying_last_close (np.float): Close price of the underlying on the last_date, used for calculation of the log moneyness
        IS_CALL (boolean): if the returned data is call or not (returns put data if set to False)

    Returns:
        options_selected: Selected Options data.
    """
    # TODO: refactor using get_option_chain
    exp_selected, dtox_selected = select_expiration(period_str, ticker.options, last_date)
    print(ticker.info['longName'])
    option_chain_1m = ticker.option_chain(date=exp_selected)
    print_message = ""
    if IS_CALL:
        options_selected = option_chain_1m.calls
        print_message += "Call"
    else:
        options_selected = option_chain_1m.puts
        print_message += "Put"
    print_message += f" Option {period_str} expiration selected: {exp_selected}, dtox of {dtox_selected}"
    print(print_message)
    # remove un-wanted data
    options_selected = options_selected.loc[options_selected.openInterest != 0, :]
    options_selected = options_selected.loc[options_selected.ask > options_selected.bid, :]
    options_selected = options_selected.sort_values('strike').reset_index(drop=True)

    if underlying_last_close:
        options_selected.loc[:, 'Log_Moneyness'] = np.log(underlying_last_close / options_selected.strike)

    return options_selected


def get_strikes(ticker, last_date):
    """
    Using the 1-month call options, select the 3 tenors representing In The Money(ITM), At The Money(ATM),
    and Out The Money(OTM).

    Args:
        ticker (yfinance.ticker.Ticker): Ticker from yfinance
        last_date (Timestamp): Last businessdate

    Returns:
        strike_dict: A dictionary where the keys are the type of strikes and the values are the corresponding strikes (float)
    """
    option_1mo_call = get_option_chain_tenor(period_str='1mo', ticker=ticker,
                                             last_date=last_date, underlying_last_close=None,
                                             IS_CALL=True)
    # select the strike that is ATM
    ATM_strike = float(option_1mo_call.loc[option_1mo_call.loc[:, 'inTheMoney'].diff(1) == True, 'strike'].squeeze())
    # lis of strikes for ITM and OTM
    ITM_strike_list = option_1mo_call.loc[option_1mo_call.inTheMoney, 'strike'].to_list()
    OTM_strike_list = option_1mo_call.loc[~option_1mo_call.inTheMoney, 'strike'].to_list()
    # select the center element
    ITM_strike = ITM_strike_list[len(ITM_strike_list) // 2]
    OTM_strike = OTM_strike_list[len(OTM_strike_list) // 2]
    # result in a dictionary
    strike_dict = {'ITM': ITM_strike, 'ATM': ATM_strike, 'OTM': OTM_strike}
    print(strike_dict)
    return strike_dict


def get_option_chain_strikes(ticker, last_date):
    """
    For all the available tenors, fetch the data with the tenors that are ITM, ATM, and OTM.

    Args:
        ticker (yfinance.ticker.Ticker): Ticker from yfinance
        last_date (Timestamp): Last businessdate

    Returns:
        call_data (pd.DataFrame): DataFrame of Call data
        put_data (pd.DataFrame):  DataFrame of Put data
    """
    # TODO: refactor using get_option_chain
    strike_dict = get_strikes(ticker, last_date)
    strikes = list(strike_dict.values())

    call_data = pd.DataFrame()
    put_data = pd.DataFrame()
    for x in ticker.options:
        opt_tenor = ticker.option_chain(date=x)
        # CALL
        opt_calls = opt_tenor.calls
        opt_calls.loc[:, 'Expiration'] = x
        opt_calls = opt_calls.loc[opt_calls.strike.isin(strikes), :]
        call_data = pd.concat([call_data, opt_calls], ignore_index=True)
        # PUT
        opt_puts = opt_tenor.puts
        opt_puts.loc[:, 'Expiration'] = x
        opt_puts = opt_puts.loc[opt_puts.strike.isin(strikes), :]
        put_data = pd.concat([put_data, opt_puts], ignore_index=True)
    call_data.loc[:, 'put_call_code'] = 'C'
    put_data.loc[:, 'put_call_code'] = 'P'
    # Add strike's corresponding moneyness as a column
    ref_df = pd.DataFrame({'strike': list(strike_dict.values()), 'moneyness': list(strike_dict.keys())})
    call_data = pd.merge(call_data, ref_df, how='left')
    put_data = pd.merge(put_data, ref_df, how='left')
    # Add Days to expiration
    call_data.loc[:, 'dtox'] = list(map(lambda x: (pd.to_datetime(x).date() - last_date.to_pydatetime().date()).days,
                                        call_data.loc[:, 'Expiration']))
    put_data.loc[:, 'dtox'] = list(map(lambda x: (pd.to_datetime(x).date() - last_date.to_pydatetime().date()).days,
                                       put_data.loc[:, 'Expiration']))
    return call_data, put_data


@clean_data
def get_option_chain(ticker, strikes_selected=None, tenors_selected=None):
    """
    For all the available tenors, fetch the data with the tenors that are ITM, ATM, and OTM.

    Args:
        ticker (yfinance.ticker.Ticker): Ticker from yfinance
        strikes_selected (list): [Optional] List of strikes to filter for from the original data
        tenors_selected (list): [Optional] List of tenors (in 'YYYY-MM-DD' format) to filter for from the original data

    Returns:
        call_data (pd.DataFrame): DataFrame of Call data
        put_data (pd.DataFrame):  DataFrame of Put data
    """

    all_tenors = ticker.options
    if tenors_selected:
        all_tenors = (x for x in all_tenors if x in tenors_selected)
        print(f"Tenors reduced from {len(ticker.options)} to {len(all_tenors)}")

    df = pd.DataFrame()
    for x in all_tenors:
        opt_tenor = ticker.option_chain(date=x)
        # CALL
        opt_calls = opt_tenor.calls
        opt_calls.loc[:, 'put_call_code'] = 'C'

        # PUT
        opt_puts = opt_tenor.puts
        opt_puts.loc[:, 'put_call_code'] = 'P'
        tenor_data = pd.concat([opt_calls, opt_puts], ignore_index=True)
        tenor_data.loc[:, 'Expiration'] = x
        df = pd.concat([df, tenor_data], ignore_index=True)

    if strikes_selected:
        df = df.loc[df.strike.isin(strikes_selected), :]
        print(f"Strikes selected :{strikes_selected}")

    return df
