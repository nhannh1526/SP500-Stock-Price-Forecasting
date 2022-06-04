import datetime
import os
import pickle
import time
import traceback

import pandas as pd
import yfinance
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm

FUNCTIONS = ["sma",
             "ema",
             "wma",
             "dema",
             "tema",
             "trima",
             "kama",
             "mama",
             "macd",
             "stoch",
             "rsi",
             "adx",
             "ppo",
             "cci",
             "aroon",
             "trix",
             "ultosc",
             "bbands",
             "sar",
             "ad",
             "obv"]
APIKEY = "YOUR_API_KEY "
ts = TimeSeries(key=APIKEY, output_format='pandas')
ti = TechIndicators(key=APIKEY, output_format="pandas")

if not os.path.exists("data"):
    os.makedirs("data")


def get_sp500_tickers():
    """Get S&P500 ticker symbols

    Returns:
        list: list of S&P500 ticker symbols
    """
    payload = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

    tickers = sorted(payload[0]["Symbol"].values.tolist())
    with open("data/sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def get_ts(function,
           symbol,
           **kwargs):
    """Get timeseries data from AlphaVantage

    Args:
        function (string): The time series of your choice
        symbol (string): The name of the equity of your choice

    Returns:
        DataFrame: Returns function time series of the equity specified
    """
    args = ""
    for key, value in kwargs.items():
        if type(value) == str:
            args += "," + key + "=" + "'" + value + "'"
        else:
            args += "," + key + "=" + str(value)
    command = "get_" + function + "(" + "'" + symbol + "'" + args + ")"
    data, meta_data = eval("ts." + command)
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    return data


def get_ti(function,
           symbol,
           **kwargs):
    """Get Technical Indicators from AlphaVantage

    Args:
        function (string): The technical indicator of your choice
        symbol (string): The name of the token of your choice

    Returns:
        DataFrame: Returns the function values
    """
    args = ""
    for key, value in kwargs.items():
        if type(value) == str:
            args += "," + key + "=" + "'" + value + "'"
        else:
            args += "," + key + "=" + str(value)
    command = "get_" + function + "(" + "'" + symbol + "'" + args + ")"
    data, meta_data = eval("ti." + command)
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
    return data


def join_df(df1,
            df2):
    """Merge two DataFrame

    Args:
        df1 (DataFrame): First DataFrame
        df2 (DataFrame): Second DataFrame

    Returns:
        DataFrame: Merged DataFrame
    """
    return df1.merge(df2, how="left", left_index=True, right_index=True)


def get_data(ticker,
             dataset_path,
             start_date=None,
             end_date=None):
    """Get data from Yahoo Finance and AlphaVantage

    Args:
        ticker (string): A ticker symbol or stock symbol
        dataset_path (string): Path to the data directory
        start_date (datetime, optional): Download start date. Defaults to None.
        end_date (datetime, optional): Download end date. Defaults to None.
    """
    file_path = os.path.join(dataset_path, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"[INFO] Getting {file_path}")

        """Get data from Yahoo Finance
        """
        df = yfinance.download(ticker, start=start_date,
                               end=end_date, progress=False)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

        """Get Daily Adjusted data from AlphaVantage
        """
        try:
            daily_adjusted = get_ts(
                "daily_adjusted", ticker, outputsize="full")
            daily_adjusted.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend Amount',
                                      'Split Coefficient']
            daily_adjusted.drop(df.columns, axis=1, inplace=True)

            df = join_df(df, daily_adjusted)
        except Exception:
            traceback.print_exc()

        """Get Technical Indicators from AlphaVantage
        """
        for function in FUNCTIONS:
            time.sleep(0.5)
            try:
                df = join_df(df, get_ti(function, ticker))
            except Exception:
                traceback.print_exc()
        df.to_csv(os.path.join(dataset_path, f"{ticker}.csv"))
    else:
        print(
            f"[INFO] Already have {file_path}")
    # time.sleep(1)


def prepare_dataset(reload_sp500=False,
                    start_date=None,
                    end_date=datetime.datetime.now()):
    if reload_sp500:
        tickers = get_sp500_tickers()
    else:
        with open("data/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists("data/dataset"):
        os.makedirs("data/dataset")

    # tickers = ["AAPL", "AMZN", "FB", "GOOGL", "MSFT", "NFLX"]
    for ticker in tqdm(tickers, desc="Getting stocks and financial data from Yahoo Finance and Alpha Vantage"):
        get_data(ticker, "data/dataset", start_date, end_date)


def main():
    prepare_dataset(True)


if __name__ == "__main__":
    main()
