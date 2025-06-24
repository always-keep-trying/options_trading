# Options, Signals, and Strategies

This repository uses open source packages to source index and options data. 
The aim of this repo is to generate signals for trading, construct strategies using the signals, and backtest these strategies. 


## Options Data

Time series data of index and options data can be fetched from the [yfinance](https://ranaroussi.github.io/yfinance/index.html) python package.
To simplify the process of fetching these data, the [support functions module](./src/support_functions.py) and a separate [module for visualizing](./src/plot_functions.py) were created.

As an example I've fetched data related to "SPX" Index and their options in the [option data visualization notebook](./src/option_data_visualiztion.ipynb). The Implied volatility Skew/Smile has been plotted in the notebook.

Few challenge that came up during exploration of the data source were...
- The options data is the current day's data only. It does not give access to historical data. 
- No greeks available (Delta, Vega, Rho, Gamma, and Theta).
- Data quality issues. For example some implied volatility values seems to be set to a near zero value. In order to remove such observations the @clean_data decorator will be used to remove such observations. (see [support_functions.py](./src/support_functions.py)). 

## Signals

The signals focused on this repository would be on relationship between volatility indices. Full disclosure here, this idea was introduced to me by Euan Sinclair from a webinar hosted by CBOE: "Finding Edge: A Key Part of Trading Process". During this webinar the relationship that was explored was between VXN(Cboe NASDAQ-100 Volatility Index) to VIX(30 day volatility of SPX). 

However, expanding on this idea this repository takes it further by comparing other volatility indices (Volatility index of NDX, DJX, RUT, Gold, and Oil). To explore these relationships and dislocations is the [dislocation analysis](./src/dislocation_analysis.py) module was created.  
**The results of these analysis and dislocations can be seen in [this notebook](./src/notebooks/Finding_an_Edge.ipynb).**

*A new data source FRED has been added in this step. Some indices were not available in yfinance, but they were available in the [fredapi](https://pypi.org/project/fredapi/).*

## Strategies

Once the Signals indentify opportunities(i.e. dislocation in relationship) to invest, actions can be taken to profit by constructing a trade. For example a ATM Straddle of the indices (options on the SPX for VIX, options on the NDX for VXN etc...) would be a great option.

In order to overcome the challenge of not having access to historical options data, we will collect daily options data. To accomplish this I've written the [Extract Transform and Load(ETL)](./src/ETL.py) module which can be run daily.

- **More to come on this section. Specifically, the constuction of the straddles and their performances**


## Data sources

Python packages used
- [yfinance](https://ranaroussi.github.io/yfinance/index.html): An open-source tool that uses Yahoo's publicly available APIs to fetch data.
- [fredapi](https://pypi.org/project/fredapi/): Fetches data from FRED, a trusted data source provided by the Federal Reserve Bank of St.Louis. An API key is needed to use this module, please visit [website](https://fred.stlouisfed.org/) to get one.

[ORATS Delayed Data API](https://orats.com/docs/historical-data-api) is a paid service that provices options historical data.


## Set up 

Few notes on the setup
- Please set your environment variable PYTHONPATH so that you can import the modules in this repo.
- Please set your environment variable FRED_API_KEY so that you can utilize the fredapi package.
- Please set your environment variable ORATS_API_KEY so that you can fetch data through their API(paid service).


Python virtual environment used in this repo can be replicated using [requirements.txt](./requirements.txt)