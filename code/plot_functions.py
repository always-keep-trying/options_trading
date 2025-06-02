import matplotlib.pyplot as plt


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