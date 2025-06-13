import os
import datetime

import pandas as pd
import numpy as np

import yfinance as yf
from fredapi import Fred

import matplotlib.pyplot as plt

try:
    fred_api = Fred(api_key=os.environ['FRED_API_KEY'])
except Exception as e:
    raise Exception("Please set your environment variable FRED_API_KEY. " + str(e))


def time_series(function):
    """Decorator used to get info regarding the time series data"""

    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        # display information regarding the time series
        print(result.iloc[:, 0:4].tail(3))

        # check if the last value is from today and check if the market is open or not
        # during trading hours, the yahoo finance api fetches the current value and gives a false "close" value
        # we will drop today's value if it is before market close
        now_dt = datetime.datetime.now()
        market_close = datetime.time(12 + 3, 15)  # 3:15 pm CST
        if (now_dt.date() == pd.to_datetime(result.index.max()).date()) & (now_dt.time() < market_close):
            result = result.drop(result.index.max(), inplace=False)
            print(f"Removing today's value as market is still open (now: {now_dt})")

        start_date = result.index.min()
        end_date = result.index.max()
        n = result.shape[0]
        print("\n")
        fmt = lambda x: pd.to_datetime(x).date()
        print(f"Time Series data from {fmt(start_date)} to {fmt(end_date)}. Total of {n} observation")

        return result

    return wrapper


@time_series
def extract_hist_yf(symbol: str, period: str = "max") -> pd.DataFrame:
    """Extract time series from yahoo finance
    Args:
        symbol (str): Ticker name.
        period (str): Length of time series to fetch from yahoo finance.

    Returns:
        Time series dataframe.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period)
    try:
        Long_Name = ticker.info['longName']
        Symbol = ticker.info['symbol']
        print(f"{Long_Name=}")
        print(f"{Symbol=}")
    except Exception as e:
        print("Error in getting ticker information \n " + str(e))
    return hist


@time_series
def extract_hist_fred(symbol: str) -> pd.DataFrame:
    """Extract time series from FRED
    Args:
        symbol (str): Ticker name.

    Returns:
        Time series dataframe.

    """
    fred_data = pd.DataFrame(fred_api.get_series(symbol), columns=['Close'])
    try:
        info = fred_api.get_series_info(symbol)
        Long_Name = info['title']
        Symbol = info['id']
        print(f"{Long_Name=}")
        print(f"{Symbol=}")
    except Exception as e:
        print("Error in getting ticker information \n " + str(e))
    return fred_data


def calculate_ratio(ts_numer: pd.DataFrame, ts_denom: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the ratio of the 2 time series given(e.g. VXN / VIX). Computes "Rolling30" column which
    is a 30-day rolling average of the Ratio. It also computs "Diff" column, defined as "Ratio" - "Rolling30".

    Args:
        ts_numer: Time series used as the numerator of the Ratio.
        ts_denom: Time series used as the denomicator of the Ratio.

    Returns:
        Time series of the ratio analysis.
    """
    # compare ratio: merge data, drop na to get same hist, calculate ratio, compute stats
    # use date as index (i.e. remove time information)
    ts_numer.index = ts_numer.index.map(lambda x: pd.to_datetime(x).date())
    ts_denom.index = ts_denom.index.map(lambda x: pd.to_datetime(x).date())

    ts_data = ts_numer.loc[:, ['Close']].join(ts_denom.loc[:, ['Close']], how='inner', lsuffix='Numerator',
                                              rsuffix='Denominator')
    ts_data.loc[:, 'Ratio'] = ts_data.CloseNumerator / ts_data.CloseDenominator

    ts_data.loc[:, 'Rolling30'] = ts_data.Ratio.rolling(30).mean()
    ts_data.loc[:, 'Diff'] = ts_data.Ratio - ts_data.Rolling30
    return ts_data


def plot_diff_hist(ts_data: pd.DataFrame, quantile: float = 0.005) -> plt.Figure:
    """
    Plot a histogram of the "Diff" column generated from calculate_ratio method.
    Calculate 2 quantiles of this histogram.
    Args:
        ts_data: DataFrame generated from calculate_ratio method.
        quantile: Define which quantile of the different you would like.

    Returns:

    """
    # observe the dislocation using the diff
    threshold_left = np.quantile(ts_data.Diff.dropna().to_list(), quantile)
    threshold_right = np.quantile(ts_data.Diff.dropna().to_list(), 1 - quantile)

    print(f"The {np.round(quantile * 100, 2)}% quantile of the difference is {np.round(threshold_left, 3)}")
    print(f"The {np.round((1 - quantile) * 100, 2)}% quantile of the difference is {np.round(threshold_right, 3)}")

    hist_h = plt.figure()
    plt.hist(ts_data.Diff, 30)
    ylim_val = plt.ylim()
    plt.vlines(threshold_left, ylim_val[0], ylim_val[1], 'r')
    plt.vlines(threshold_right, ylim_val[0], ylim_val[1], 'r')
    plt.title('Histogram of Diff')
    plt.grid()
    plt.show()
    return hist_h


def plot_dislocation_time_series(ts_data: pd.DataFrame, query: str = None,
                                 start_date: datetime.date = datetime.date(2018, 1, 1)) \
        -> tuple[plt.Figure, pd.DataFrame]:
    """
    Plot the time series of the Ratio and Rolling30. The query is the condition on which we define the dislocation.
    If a query is defined, it will mark the observations that meets those criteria with a red x.
    Args:
        ts_data: DataFrame generated from calculate_ratio method.
        query: [Optional] A query of conditions that selects when the dislocation occurs.
        start_date: Start date of the analysis, to limit the lookback period.

    Returns:
        A figure handle from the time series plot and a DataFrame selected based on the query.
    """
    fig_h = plt.figure(figsize=(10, 5))
    plot_data = ts_data.loc[ts_data.index >= start_date, :]
    plt.plot(plot_data.index, plot_data.Ratio, marker='o', ms=1.5, label='Ratio')
    # rational: use last month of observations to calculate an average
    plt.plot(plot_data.index, plot_data.Rolling30, marker='.', ls=None, c='m', ms=1.5, alpha=0.3, label='MV(30)')
    plt.legend()
    plt.grid()
    if query:
        identifier = plot_data.query(query)
        plt.scatter(identifier.index, identifier.Ratio, marker='x', c='r', label='Dislocation')
    else:
        identifier = None
    plt.close(fig_h)
    return fig_h, identifier


def main_analysis(analysis_dictionary: dict, query=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to calculate the Ratio, generate a histogram of "Diff", and plot a time series plot regarding the Ratio.

    Args:
        analysis_dictionary: A dictionary used to calculate the ratio. Please define the keys as the ticker and the
        value as the time series. The first item in the dictionary will be the numerator and the second item will
        be the denominator.
        query: [Optional] A query of conditions that selects when the dislocation occurs.

    Returns:
        1) DataFrame selected based on the query, it will be empty if query has not been defined.
        2) DataFrame calculated from the calculate_ratio method
    """
    # 1st element will be the numerator and the 2nd element will be the denominator
    key_vals = list(analysis_dictionary.keys())
    ratio_definition_str = f"Ratio = {key_vals[0]}/{key_vals[1]}"
    print(f"Performing analysis of {key_vals[0]} over {key_vals[1]}, " + ratio_definition_str)
    ts_data = calculate_ratio(ts_numer=analysis_dictionary[key_vals[0]], ts_denom=analysis_dictionary[key_vals[1]])
    # summary stat of last 1 year
    summary_data = ts_data.loc[:, ['Ratio', 'Rolling30', 'Diff']].tail(252)
    summary = summary_data.describe()
    summary = summary.loc[['min', 'mean', 'max', 'std'], :]
    print("Summary Statistics")
    print(summary)

    plot_diff_hist(ts_data, quantile=0.005)

    # plot the time series, the query will provide the red x markers for dislocation
    plt_h, id_df = plot_dislocation_time_series(ts_data, query)
    plt.figure(plt_h)
    plt.title(ratio_definition_str)
    plt.show()
    return id_df, ts_data
