import pandas as pd
import numpy as np

days_in_year = 365
period_dic = {
    '1d': 1,
    '5d': 5,
    '1mo': round(days_in_year / 12),
    '3mo': round(3 * days_in_year / 12),
    '6mo': round(6 * days_in_year / 12),
    '1y': days_in_year,
    '2y': 2 * days_in_year,
    '5y': 5 * days_in_year,
    '10y': 10 * days_in_year
}


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

    strike_dict = get_strikes(ticker, last_date)

    call_data = pd.DataFrame()
    put_data = pd.DataFrame()
    for x in ticker.options:
        opt_tenor = ticker.option_chain(date=x)
        # CALL
        opt_calls = opt_tenor.calls
        opt_calls.loc[:, 'Expiration'] = x
        opt_calls = opt_calls.loc[opt_calls.strike.isin(strike_dict.values), :]
        call_data = pd.concat([call_data, opt_calls], ignore_index=True)
        # PUT
        opt_puts = opt_tenor.puts
        opt_puts.loc[:, 'Expiration'] = x
        opt_puts = opt_puts.loc[opt_puts.strike.isin(strike_dict.values), :]
        put_data = pd.concat([put_data, opt_puts], ignore_index=True)
    call_data.loc[:, 'put_call_code'] = 'C'
    put_data.loc[:, 'put_call_code'] = 'P'
    return call_data, put_data


