import os
import pandas as pd

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

    """
    posn_df = position_df.loc[:, ['ticker', 'expirDate', 'strike']].drop_duplicates()
    posn_list = posn_df.values.tolist()

    for one_posn in posn_list:
        extract_contract_data(*one_posn)
    print("Positions processed")
    print(posn_df)




