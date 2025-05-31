import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_1m_exp(options, last_date):
    """
    Using the yfinance's Options data, identify the options expiration date that is closest to 30 days.

    Args:
      options (yfinance.ticker.Options): Options data from yfinance
      last_date (Timestamp): Last businessdate

    Returns:
      exp_1m (str): String of expiration date using the "YYYY-MM-DD" format.
      dtox_1m (int): Days to expiration.
    """
    dtox = list(map(lambda x: (pd.to_datetime(x).date() - last_date.to_pydatetime().date()).days, options))
    # find the dtox closest to 30 days
    dtox_1m = min(dtox, key=lambda x: abs(x - 30))
    exp_1m = options[dtox == dtox_1m]
    return exp_1m, dtox_1m


def get_1m_option_chain(options, last_date, underlying_last_close, IS_CALL=True):
    """
    Using the yfinance's Options data, fetch options data with the expiration date that is closest to 30 days.
    The function also adds a log moneyness column; ln(S/K)
    The following data points are modified
        - zero open interest data are removed
        - Ask prices should be higher than bid price.

    Args:
      options (yfinance.ticker.Options): Options data from yfinance
      last_date (Timestamp): Last businessdate
      underlying_last_close (np.float): Close price of the underlying on the last_date
      IS_CALL (boolean): if the returned data is call or not (returns put data if set to False)

    Returns:
      options_selected: Selected Options data.
    """
    opt_1m_exp, opt_1m_exp_dtox = get_1m_exp(options.options, last_date)
    print(options.info['longName'])
    option_chain_1m = options.option_chain(date=opt_1m_exp)
    print_message = ""
    if IS_CALL:
        options_selected = option_chain_1m.calls
        print_message += "Call"
    else:
        options_selected = option_chain_1m.puts
        print_message += "Put"
    print_message += f" Option 1 month expiration selected: {opt_1m_exp}, dtox of {opt_1m_exp_dtox}"
    print(print_message)
    options_selected.loc[:, 'Log_Moneyness'] = np.log(underlying_last_close / options_selected.strike)
    # remove un-wanted data
    options_selected = options_selected.loc[options_selected.openInterest != 0, :]
    options_selected = options_selected.loc[options_selected.ask > options_selected.bid, :]

    options_selected = options_selected.sort_values('strike').reset_index(drop=True)
    return options_selected


def plot_ivol_and_price(options, underlying_last_close):
    """
    Plot 4 suboplots of options data based on the combination of the following variables
        x-axis: Log-moneyness, Strike
        y-axis: Implied Volatility, Price

    Args:
      options (yfinance.ticker.Options): Options data from yfinance
      underlying_last_close (np.float): Close price of the underlying on the last_date

    Returns:
      fig: Figure object
      ax: Tuple of axis, each element represents a subplot
    """

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(221)
    plt.scatter(options.Log_Moneyness, options.impliedVolatility)
    plt.axvline(x=0, color='r', linestyle='--')

    plt.grid()
    plt.xlabel('Log Moneyness')
    plt.ylabel('Ivol')
    plt.title('Log Moneyness vs Implied Vol')

    ax2 = plt.subplot(222)
    plt.scatter(options.strike, options.impliedVolatility)
    plt.axvline(x=underlying_last_close, color='r', linestyle='--')

    plt.grid()
    plt.xlabel('Strike')
    plt.ylabel('Ivol')
    plt.title('Strike vs Implied Vol')

    ax3 = plt.subplot(223)
    plt.scatter(options.Log_Moneyness, options.lastPrice)
    plt.axvline(x=0, color='r', linestyle='--')

    plt.grid()
    plt.xlabel('Log Moneyness')
    plt.ylabel('Price')
    plt.title('Log Moneyness vs Price')

    ax4 = plt.subplot(224)
    plt.scatter(options.strike, options.lastPrice)
    plt.axvline(x=underlying_last_close, color='r', linestyle='--')

    plt.grid()
    plt.xlabel('Strike')
    plt.ylabel('Price')
    plt.title('Strike vs Price')

    ax = (ax1, ax2, ax3, ax4)
    return fig, ax
