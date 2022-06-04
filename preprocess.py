import os
from tkinter.tix import COLUMN
from matplotlib.pyplot import axis
import pandas as pd
from tqdm import tqdm
from datetime import datetime

DATASET_PATH = "data/dataset/"
SUBDATASET_PATH = "data/subdataset"
COLUMN_NAMES = [
    'Open',
    'High',
    'Low',
    'Close',
    'Adj Close',
    'Volume',
    'Dividend Amount',
    'Split Coefficient',
    'SMA',
    'EMA',
    'WMA',
    'DEMA',
    'TEMA',
    'TRIMA',
    'KAMA',
    'MAMA',
    'FAMA',
    'MACD',
    'MACD_Hist',
    'MACD_Signal',
    'SlowK',
    'SlowD',
    'RSI',
    'ADX',
    'PPO',
    'CCI',
    'Aroon Down',
    'Aroon Up',
    'TRIX',
    'ULTOSC',
    'Real Upper Band',
    'Real Middle Band',
    'Real Lower Band',
    'SAR',
    'Chaikin A/D',
    'OBV'
]
DROP_COLUMNS = ["Volume",
                "Dividend Amount",
                "MACD",
                "MACD_Hist",
                "MACD_Signal",
                "SlowK",
                "SlowD",
                "Aroon Down",
                "Aroon Up",
                "RSI",
                "ADX",
                "PPO",
                "CCI",
                "TRIX",
                "ULTOSC"]


def main():
    if not os.path.exists(SUBDATASET_PATH):
        os.makedirs(SUBDATASET_PATH)

    for ticker in tqdm(os.listdir(DATASET_PATH), desc="Filter data from 01-01-2018 to 03-15-2022, Fill nan values and Drop unnecessary features"):
        df = pd.read_csv(DATASET_PATH+ticker, index_col="Date")
        df = df.reindex(COLUMN_NAMES, axis=1)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        if df.index.min() < datetime(2018, 1, 1):
            df = df.loc[datetime(2018, 1, 1):datetime(2022, 3, 15)]
            df.interpolate(inplace=True)
            df = df.apply(lambda col: col.fillna(col.median()))
            df.drop(DROP_COLUMNS, axis=1, inplace=True)
            df.to_csv(os.path.join(SUBDATASET_PATH, ticker))


if __name__ == "__main__":
    main()
