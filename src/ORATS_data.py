import requests
import os
import pandas as pd
from urllib.parse import urljoin

ORATS_API_KEY = os.environ['ORATS_API_KEY']
data_url = "https://api.orats.io/datav2/"
# REF: https://orats.com/docs/historical-data-api


def get_request(url: str, query_parms: dict) -> pd.DataFrame:
    request_url = urljoin(data_url, url)
    try:
        response = requests.request("GET", request_url, params=query_parms)
        response.raise_for_status()  # Raise exception for bad status codes

        # Convert response to DataFrame
        data = response.json()
        df = pd.DataFrame(data['data'])
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def get_ticker(ticker: str) -> pd.DataFrame:
    """
    Get Ticker information from ORATS API
    Args:
        ticker (str): The ticker to retrieve. Ex: AAPL

    Returns:

    """
    query_dict = {"token": ORATS_API_KEY, "ticker": ticker}
    return get_request('ticker', query_dict)


def get_hist_strikes(tickers: str | list, trade_date: str, dte=None, delta=None) -> pd.DataFrame:
    """
    Fetches option chain of one date using the ORATS API.
    Args:
        tickers (str or list): The ticker to retrieve (multiple tickers should be comma delimited - max of 10 allowed). Ex: AAPL,TSLA
        trade_date (str): YYYY-MM-DD
        dte (str): Filter by DTE range. Ex: 30,45
        delta (str): Filter by delta range. Ex: .30,.45
    Returns:

    """

    if isinstance(tickers, list):
        assert len(tickers) <= 10, "Maximum of 10 tickers can be processed"
        tickers = ",".join(tickers)

    querystring = {"token": ORATS_API_KEY, "ticker": tickers, "tradeDate": trade_date}
    if dte:
        querystring['dte'] = dte

    if delta:
        querystring['delta'] = delta

    return get_request("hist/strikes", querystring)


def get_his_strikes_OPRA(ticker: str, exp_date: str, strike: int | str, trade_date=None) -> pd.DataFrame:
    """
    Gets the full history of one option contract using the ORATS API.
    Args:
        ticker: Option contract's underlying symbol.
        exp_date: Expiration date of the option contract.
        strike: Strike of the option contract
        trade_date: [Optional] reference date of the option contract

    Returns:

    """
    if isinstance(strike, int):
        strike = str(strike)

    querystring = {"token": ORATS_API_KEY, "ticker": ticker, "expirDate": exp_date,
                   "strike": strike}

    if trade_date:
        querystring["tradeDate"] = trade_date

    return get_request("hist/strikes/options", querystring)


def format_ORATS_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format dataframe fetched from ORATS API. Each contract originally have both call and put in a single row,
    this function splits those into 2 rows (one for call one for put).
    Args:
        df: Data fetched frome ORATS API

    Returns:
        formatted datafrmae
    """
    opt_contract_cols = ['ticker', 'tradeDate', 'expirDate', 'dte', 'strike', 'stockPrice']
    contract_specific_columns = ['BidPrice', 'Value', 'AskPrice', 'BidIv', 'MidIv', 'AskIv']
    greeks_cols = ['delta', 'gamma', 'theta', 'vega', 'rho', 'phi']
    other_cols = ['smvVol', 'residualRate']

    new_df = pd.DataFrame()

    for pc_type in ['call', 'put']:
        col_rename = {}
        for x in contract_specific_columns:
            col_rename[pc_type + x] = x
        curr_df = df.loc[:, opt_contract_cols + list(map(lambda x: pc_type + x, contract_specific_columns)) +
                            greeks_cols + other_cols].rename(columns=col_rename).copy()
        curr_df.loc[:, 'put_call_code'] = pc_type[0].upper()
        curr_df = curr_df.loc[:, list(curr_df.columns[0:6]) + ['put_call_code'] + list(curr_df.columns[6:-1])]
        new_df = pd.concat([new_df, curr_df], ignore_index=True)

    return new_df.sort_values(['dte', 'strike']).reset_index(drop=True)

