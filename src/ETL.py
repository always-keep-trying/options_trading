import os
import datetime

import pandas as pd
import math

import yahoo.src.support_functions as sf
import yahoo.src.dislocation_analysis as da
import yahoo.src.ORATS_data as od

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


IS_VERBOSE = False


class ETL:
    """Parent class for Extract Transform and Load process"""

    def __init__(self, data_dir: str, ticker: str | list):
        self.data = None
        self.data_dir = data_dir
        self.count = None

        self.ticker = ticker

        self.today_dt = datetime.datetime.now().date()
        self.today_dir = None
        if IS_VERBOSE:
            print(f"Today's date is {self.today_dt.strftime('%Y-%m-%d')}")
        self.file_name = None
        self.create_dir()

    def create_dir(self):
        assert os.path.exists(self.data_dir), "Data directory does not exist!"
        self.today_dir = os.path.join(self.data_dir, self.today_dt.strftime('%Y-%m-%d'))
        if not os.path.exists(self.today_dir):
            os.mkdir(self.today_dir)
            print("Created new folder: " + self.today_dir)

    def extract(self):
        raise NotImplementedError("To be implemented by child class")

    def transform(self):
        raise NotImplementedError("To be implemented by child class")

    def load(self):
        raise NotImplementedError("To be implemented by child class")

    def validate(self):
        raise NotImplementedError("To be implemented by child class")

    def revert_load(self):
        raise NotImplementedError("To be implemented by child class")

    def run(self):
        if isinstance(self.ticker, str) and os.path.exists(self.file_name):
            if IS_VERBOSE:
                print(f"File already exists: {self.file_name}")
            self.count = pd.read_parquet(self.file_name).shape[0]
        elif isinstance(self.ticker, list) and all([os.path.exists(self.file_name(x)) for x in self.ticker]):
            self.count = 0
            for symbol in self.ticker:
                self.count += pd.read_parquet(self.file_name(symbol)).shape[0]
        elif self.today_dt.weekday() >= 5:  # check if it is a weekend
            print("today is a weekend and the ETL will not be executed")
        else:
            self.extract()
            if not self.data.empty:
                #try:
                self.transform()
                self.load()
                self.validate()
                print(f"{self.__class__.__name__} complete for {self.ticker}")
                # except:
                #     self.revert_load()
            else:
                print(f"{bcolors.WARNING}Warning: "
                      f"No data was extracted {self.ticker} during use of "
                      f"{self.__class__.__name__}{bcolors.ENDC}")
        print(self)

    def __str__(self):
        p_str = f"{self.__class__.__name__} for ticker:{self.ticker} \n"
        if isinstance(self.file_name, str):
            p_str += f"File: {self.file_name}\n"
        p_str += f"Count: {self.count}"
        return p_str

    def fetch(self) -> pd.DataFrame:
        if os.path.exists(self.file_name):
            return pd.read_parquet(self.file_name)
        else:
            return pd.DataFrame()


class UnderlyingETL(ETL):
    """Child class specific to Underlying ETL using the yfinance"""

    def __init__(self, data_dir: str, ticker: str):
        super().__init__(data_dir, ticker)
        underlying_dir = os.path.join(self.today_dir, 'Underlying')
        if not os.path.exists(underlying_dir):
            os.mkdir(underlying_dir)
            print("Created new folder: " + underlying_dir)
        self.file_name = os.path.join(underlying_dir, self.ticker + '.parquet')

    def extract(self):
        self.data = da.extract_hist_yf(self.ticker)

    def transform(self):
        # no transform needed
        assert pd.to_datetime(
            self.data.index.max()).date() == self.today_dt, \
            f"Data is missing today's data! Today: {self.today_dt}"

    def load(self):
        self.data.to_parquet(self.file_name)
        self.count = self.data.shape[0]

    def validate(self):
        # very simple validation
        read_df = pd.read_parquet(self.file_name)
        df_size = read_df.shape
        assert df_size[0] > 0, "The file is empty!"

        assert {'Open', 'High', 'Low', 'Close'}.issubset(set(read_df.columns)), "there are missing columns (from OHLC)"

        value_closest = min(read_df.iloc[-1, :][['Open', 'High', 'Low', 'Close']].to_list(), key=lambda x: abs(x - 0))
        assert not math.isclose(value_closest,0), "There is a zero value in the OHLC data!"


class OptionETL(ETL):
    """Child class specific to Option ETL using the yfinance"""

    def __init__(self, data_dir: str, ticker: str):
        super().__init__(data_dir, ticker)
        option_dir = os.path.join(self.today_dir, 'Option')
        if not os.path.exists(option_dir):
            os.mkdir(option_dir)
            print("Created new folder: " + option_dir)
        self.file_name = os.path.join(option_dir, "Option" + self.ticker + '.parquet')

    def extract(self):
        self.data = sf.get_option_chain_asis(self.ticker)

    def transform(self):
        self.data = self.data.astype({'inTheMoney': 'bool'})
        self.data.loc[:, 'DTOX'] = self.data.Expiration.apply(lambda x: (pd.to_datetime(x).date() - self.today_dt).days)

    def load(self):
        self.data.to_parquet(self.file_name)
        self.count = self.data.shape[0]

    def validate(self):
        # very simple validation
        read_df = pd.read_parquet(self.file_name)
        df_size = read_df.shape
        assert df_size[0] > 0, "The file is empty!"

        assert {'strike', 'lastPrice', 'impliedVolatility', 'DTOX'}.issubset(
            set(read_df.columns)), "there are missing columns (from OHLC)"


class ORATS_Options(ETL):
    """ get historical data """
    def __init__(self, data_dir: str, ticker: list, trade_date: str):
        self.trade_date = trade_date
        super().__init__(data_dir, ticker)
        self.today_dir = None

    def create_dir(self):
        option_dir = os.path.join(self.data_dir, self.trade_date, 'Option')
        if not os.path.exists(option_dir):
            os.makedirs(option_dir)
            print("Created new folder: " + option_dir)
        self.file_name = lambda x: os.path.join(option_dir, "Option" + x + '.parquet')

    def extract(self):
        self.data = od.get_hist_strikes(tickers=self.ticker, trade_date=self.trade_date)

    @staticmethod
    def check_any_duplicate_data(data_frame: pd.DataFrame) -> bool:
        """
        Checkes for duplciate contract data. Returns True if there is a duplciate data and
        retuns False when there is no duplicates data
        Args:
            data_frame: Dataframe of option contracts for a given date.
        Returns:
            boolean
        """
        count_df = data_frame.groupby(
            ['ticker', 'tradeDate', 'expirDate', 'dte', 'strike', 'put_call_code']
        )['stockPrice'].count().reset_index(drop=False).rename(columns={'stockPrice': 'Count'})
        drop_df = count_df.query('Count > 1')
        is_duplicate_data = not(drop_df.empty)
        return is_duplicate_data

    @staticmethod
    def clean_up_duplocate_data(data_frame: pd.DataFrame) -> pd.DataFrame:
        ind_df = data_frame.loc[:, ['ticker', 'tradeDate', 'expirDate', 'dte', 'strike', 'put_call_code']].drop_duplicates()
        data_frame = data_frame.loc[ind_df.index, :].reset_index(drop=True)
        return data_frame

    def transform(self):
        # subset data by ticker
        all_tickers = self.data['ticker'].drop_duplicates().to_list()
        data_dict = {}
        for symb in all_tickers:

            df = od.format_ORATS_data(
                df = self.data.loc[self.data['ticker'] == symb, :].reset_index(drop=False).copy()
            )
            duplicate_data = self.check_any_duplicate_data(df)
            if duplicate_data:
                df = self.clean_up_duplocate_data(df)
                duplicate_data_2nd = self.check_any_duplicate_data(df)
                assert not(duplicate_data_2nd), 'There should be no longer duplciat data'

            data_dict[symb] = df
        # convert the pd.DataFrame to dict
        self.data = data_dict

    def load(self):
        self.count = 0
        for key, value in self.data.items():
            value.to_parquet(self.file_name(key))
            self.count += value.shape[0]

    def validate(self):
        # very simple validation
        for key, value in self.data.items():
            read_df = pd.read_parquet(self.file_name(key))
            df_size = read_df.shape
            assert df_size[0] > 0, f"The file for {key} is empty!"

            assert {'ticker', 'tradeDate', 'expirDate', 'dte', 'strike', 'stockPrice'}.issubset(
                set(read_df.columns)), "Contract info is missing!"

            assert {'delta', 'gamma', 'theta', 'vega', 'rho'}.issubset(
                set(read_df.columns)), "Greeks are missing!"

            count_df = read_df.groupby(
                ['ticker','tradeDate','expirDate','dte','strike','put_call_code']
            )['stockPrice'].count().reset_index(drop=False)

            assert count_df['stockPrice'].max() == 1, 'Duplicate data exists!'

    def revert_load(self):
        """
        Method used to delete the data when validation fails
        """
        for key, _ in self.data.items():
            file_dir = self.file_name(key)
            if os.path.exists(file_dir):
                os.remove(file_dir)
                print(f"File removed: {file_dir}")


    def clean_up(self):
        """
        Clean up data by removing duplciate contract data stemming from ORATs datasource.
        """
        for sym in self.ticker:
            curr_file = self.file_name(sym)
            if os.path.exists(curr_file):
                df = pd.read_parquet(curr_file)
                if self.check_any_duplicate_data(df):
                    df = self.clean_up_duplocate_data(df)
                    os.remove(curr_file)
                    df.to_parquet(curr_file)
                    print(f"Dropped duplcaite contract data.\n New file: {curr_file}")


    def __str__(self):
        p_str = super().__str__()
        p_str += "\nFiles: " + "\n ".join([self.file_name(x) for x in self.ticker])
        return p_str




def main(data_dir: str, index_opt_dict: dict):
    """ main function to run the ETL"""
    now_dt = datetime.datetime.now()
    market_close = datetime.time(12 + 3, 15)  # 3:15 pm CST
    if now_dt.time() < market_close:
        print("Market is still open, skipping ETL from yfinance")
    else:
        ETL_summary = pd.DataFrame()
        today_dir = None
        for index_sym, opt_sym in index_opt_dict.items():
            und = UnderlyingETL(data_dir=data_dir, ticker=index_sym)
            und.run()

            if isinstance(opt_sym, str):
                opt_sym = [opt_sym]

            for x in opt_sym:
                opt = OptionETL(data_dir=data_dir, ticker=x)
                opt.run()

                summary = pd.DataFrame(
                    {'Vol_index': index_sym,
                     'Vol_index_data': und.count,
                     'Option': x,
                     'Option_data': opt.count
                     }, index=[0]
                )
                ETL_summary = pd.concat([ETL_summary, summary], ignore_index=True)

                today_dir = today_dir or und.today_dir or opt.today_dir
        print(ETL_summary)
        ETL_summary.to_csv(
            os.path.join(today_dir, "ET_Summary_" + os.path.basename(today_dir) + ".csv"),
            index=False
        )


def main_ORATS_hist(data_dir: str, tickers: list, ref_date: str):
    """ Fetch historical data of options using ORATS API"""

    opt_hist = ORATS_Options(data_dir=data_dir, ticker=tickers, trade_date=ref_date)
    opt_hist.run()
    print(f"ETL completed for symbols:{tickers}, total count of data: {opt_hist.count}")


def fetch_data(ticker: str, is_option: bool = True,
               ref_date: str | datetime.date = datetime.datetime.now().date()) -> pd.DataFrame:
    """
    Fetch existing data generated from the ETL process
    Args:
        ticker: String ticker used in yfinance
        is_option: if the data is option data or not, if set to False the underlying data will be fetched
        ref_date: the date of the data to be fetched

    Returns:

    """
    if isinstance(ref_date,str):
        ref_date = pd.to_datetime(ref_date, format="%Y-%m-%d").date()

    data_path = os.path.join(os.environ['PYTHONPATH'], "yahoo", "data", ref_date.strftime('%Y-%m-%d'))
    assert os.path.exists(data_path), f"Requested date: {ref_date} data does not exist. (Path: {data_path})"

    if is_option:
        file_path = os.path.join(data_path, 'Option', f"Option{ticker}.parquet")
    else:
        file_path = os.path.join(data_path, 'Underlying', f"{ticker}.parquet")

    return pd.read_parquet(file_path)


if __name__ == "__main__":
    data_dir = os.path.join(os.environ['PYTHONPATH'], "yahoo", "data")

    index_opt_dict = {
        "^VIX": "^SPX",  # replication/examples, VIX
        "^VXN": "^NDX",  # replication, Nasdaq-100
        "^VXD": "^DJX",  # example 1, Dow Jones Industrial Average
        "^OVX": ["UCO", "BNO"],  # example 2, Crude Oil
        "^RVX": "^RUT",  # example 3, Russell-2000
        "^GVZ": "GLD"  # example 4 and 5, Gold
    }

    if datetime.datetime.now().date().weekday() >= 5:
        print("skipping ETL during weekend")
    else:
        main(data_dir, index_opt_dict)

    main_ORATS_hist(data_dir=data_dir, tickers=["SPX", "BNO", "UCO", "DJX"], ref_date='2020-04-21')
