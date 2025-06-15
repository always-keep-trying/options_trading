# Options, Signals, and Strategies

This repository uses open source packages to source data, including index and options data. 
The aim of this repo is to generate signals for trading, construct strategies using the signals, and backtest these strategies. 


## Options Data

Time series data of index and options data can be fetched from the [yfinance](https://ranaroussi.github.io/yfinance/index.html) python package.
To simplify the process of fetching these data I've written a [support functions module](./code/support_functions.py) and a separate [module for visualizing](./code/plot_functions.py).

As an example I've fetched data related to "SPX" Index and their options in the [option data visualization notebook](./code/option_data_visualiztion.ipynb). I've plotted the Implied volatility Skew/Smile in the notebook.

Few challenge I've noticed right away regarding the option data source from yfnance were...
- The options data is the current day's data only. I do not have access to historical data. 
- No greeks are available (Delta, Vega, Rho, and Gamma).
- Data quality issues. For example some implied volatity values seems to be set to a near zero value. In order to remove such observations I've created a decorator to remove such observations. (see [support_functions.py](./code/support_functions.py) for details). 

## Signals

I've decided to focus on relationship between volatility indices. Full disclosure here, this idea was introduced to me by Euan Sinclair from a webinar hosted by CBOE: "Finding Edge: A Key Part of Trading Process". During this webinar the relationship that was explored was between VXN(Cboe NASDAQ-100 Volatility Index) to VIX(30 day volatility of SPX). 

However, I take this idea further by comparing other volatility indices (Volatility index of NDX, DJX, RUT, Gold, and Oil). I've named the module to explore these relationships and dislocation as [dislocation analysis](./code/dislocation_analysis.py). You can take a look at the results of these analysis and dislocation in [this notebook](./code/Finding_an_Edge.ipynb).

*A new data source FRED has been added in this step. Some indices were not available in yfinance but I was able to find them in the [fredapi](https://pypi.org/project/fredapi/).*

## Strategies

Once we use the Signals to indentify opportunities(i.e. dislocation in relationship) we can take action by constructing a trade. For example a ATM Straddle of the indices would be a great option to explore.

In order to overcome the challenge of not having access to historical options data, we will collect daily options data. To accomplish this I've written the [Extract Transform and Load(ETL)](./code/ETL.py) module which can be run daily.

- **More to come on this section**


## Data sources

Python packages used
- [yfinance](https://ranaroussi.github.io/yfinance/index.html): An open-source tool that uses Yahoo's publicly available APIs to fetch data.
- [fredapi](https://pypi.org/project/fredapi/): Fetches data from FRED, a trusted data source provided by the Federal Reserve Bank of St.Louis. You need to apply for a API key through their [website](https://fred.stlouisfed.org/)


## Set up process

Few notes on the setup
- Please set your environment variable PYTHONPATH so that you can import the modules in this repo.
- Please set your environment variable FRED_API_KEY so that you can utilize the fredapi package.

Python virtual environment used in this repo can be replicated using [requirements.txt](./requirements.txt)