import requests
import json
import pandas as pd
import time
import numpy as np
import logging
import os
from datetime import datetime as dt

try:
    import invest.dateutils as dateutils
except ModuleNotFoundError:
    import dateutils

logger = logging.getLogger("main")
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class Security:
    STOCK_DIR = 'data'

    def __init__(self, symbol: str):
        self.API_URL = 'https://www.alphavantage.co/query'
        self.symbol = symbol
        self.filename = os.path.join(Security.STOCK_DIR, f'{symbol}.json')
        if os.path.exists(self.filename):
            self.data = pd.read_json(self.filename)
            self.data.sort_values('date', inplace=True)
            self.latest_date = dt.strftime(self.data.tail(1)['date'].tolist()[0], '%Y-%m-%d')
        else:
            logger.warning(f"{symbol} not found in cache")
            self.data = None
            self.latest_date = '1970-01-01'

    def get_interday_gain_loss(self):
        """
        Calculates the gain and loss between CLOSEs on two consecutive days.
        Returns: A dataframe keyed on the date. Has one column with both gains and losses reported, a column to report
        only gains, and a column to report only losses.
        """
        df = self.data[['date', 'close']].copy()
        df['interday_gain_loss'] = df['close'].diff(periods=1).fillna(0.00)
        df['interday_gain'] = df['interday_gain_loss'].where(df['interday_gain_loss'] > 0)
        df['interday_loss'] = -1 * df['interday_gain_loss'].where(df['interday_gain_loss'] < 0)
        df[['interday_gain', 'interday_loss']] = df[['interday_gain', 'interday_loss']].fillna(0.00)

        df['interday_gain_loss_pct'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
        df['interday_gain_pct'] = df['interday_gain_loss_pct'].where(df['interday_gain_loss_pct'] > 0)
        df['interday_loss_pct'] = -1 * df['interday_gain_loss_pct'].where(df['interday_gain_loss_pct'] < 0)
        df[['interday_gain_pct', 'interday_loss_pct']] = df[['interday_gain_pct', 'interday_loss_pct']].fillna(0.00)
        df[['interday_gain_loss_pct', 'interday_gain_pct',
            'interday_loss_pct']] = df[['interday_gain_loss_pct', 'interday_gain_pct', 'interday_loss_pct']].fillna(
            0.00)
        df.drop(columns=['close'], inplace=True)
        return df

    def get_intraday_gain_loss(self):
        """
        Calculates the gain and loss between OPEN and CLOSE on a single day.
        Returns: A dataframe keyed on the date. Has one column with both gains and losses reported, a column to report
        only gains, and a column to report only losses.
        """
        df = self.data[['date', 'open', 'close']].copy()
        df['intraday_gain_loss'] = df['close'] - df['open']
        df['intraday_gain'] = df['intraday_gain_loss'].where(df['intraday_gain_loss'] > 0)
        df['intraday_loss'] = -1 * df['intraday_gain_loss'].where(df['intraday_gain_loss'] < 0)
        df[['intraday_gain', 'intraday_loss']] = df[['intraday_gain', 'intraday_loss']].fillna(0.00)

        df['intraday_gain_loss_pct'] = ((df['close'] / df['open']) - 1) * 100
        df['intraday_gain_pct'] = df['intraday_gain_loss_pct'].where(df['intraday_gain_loss_pct'] > 0)
        df['intraday_loss_pct'] = -1 * df['intraday_gain_loss_pct'].where(df['intraday_gain_loss_pct'] < 0)
        df[['intraday_gain_pct', 'intraday_loss_pct']] = df[['intraday_gain_pct', 'intraday_loss_pct']].fillna(0.00)
        df.drop(columns=['open', 'close'], inplace=True)
        return df

    def get_sma(self, period=10):
        """
        Calculates a Simple Moving Average (SMA) using interday gain/loss data.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and a column with an SMA value.
        """
        # TODO Make a closing-price variant (Not just gain/loss)
        df = self.get_interday_gain_loss()
        df[f'sma_{period}dy'] = df['interday_gain_loss'].rolling(window=period, min_periods=0).mean()
        df[f'sma_{period}dy_pct'] = df['interday_gain_loss_pct'].rolling(window=period, min_periods=0).mean()
        df.drop(columns=[c for c in df.columns.values if 'interday_' in c], inplace=True)
        return df

    def get_sma_price(self, period=10):
        """
        Calculates a Simple Moving Average (SMA) using price data.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and a column with an SMA value.
        """
        df = self.data[['date', 'close']].copy()
        df[f'sma_price_{period}dy'] = df['close'].rolling(window=period, min_periods=0).mean()
        df.drop(columns=['close'], inplace=True)
        return df

    def get_ema(self, period=10):
        """
        Calculates an Exponential Moving Average (EMA) using interday gain/loss data.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and a column with an EMA value.
        """
        # TODO Make a closing-price variant (Not just gain/loss)
        df = self.get_interday_gain_loss()
        df[f'ema_{period}dy'] = df['interday_gain_loss'].ewm(span=period).mean()
        df[f'ema_{period}dy_pct'] = df['interday_gain_loss_pct'].ewm(span=period).mean()
        df.drop(columns=[c for c in df.columns.values if 'interday_' in c], inplace=True)
        return df

    def get_bollinger_bands(self, period=20, stddev_factor=1):
        """
        Calculates an upper and lower Bollinger Band. It uses either an SMA or an EMA, and adds/subtracts a windowed
        standard deviation from it.
        Param period: An integer value describing the window of the moving average and standard deviation
        Param stddev_factor: Use a variable sized standard deviation in the Bollinger Bands
        Param method: Use SMA or EMA weightings
        Returns: A dataframe keyed on the date and columns for an upper and lower Bollinger Band.
        """
        to_merge = self.get_sma_price(period)
        df = self.data[['date', 'close']].copy().merge(to_merge, on='date')
        df[f'std_{period}dy'] = df['close'].rolling(window=period, min_periods=1).std()

        df[f'bollinger_upper_{period}dy_{stddev_factor}std'] = \
            stddev_factor * df[f'std_{period}dy'] + df[f'sma_price_{period}dy']
        df[f'bollinger_lower_{period}dy_{stddev_factor}std'] = \
            df[f'sma_price_{period}dy'] - stddev_factor * df[f'std_{period}dy']

        df.drop(columns=['close', f'std_{period}dy', f'sma_price_{period}dy'], inplace=True)
        return df

    def get_rsi(self, period=20):
        """
        Calculates the Relative Strength Index (RSI). A value lower than 30 means the stock is under-bought, and 70 is
        over-bought.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and an RSI column
        """
        df = self.get_interday_gain_loss()
        df[['sma_gain', 'sma_loss']] = df[['interday_gain', 'interday_loss']] \
            .rolling(window=period, min_periods=1).mean()
        df[f'rsi_{period}dy'] = 100 - (100 / (1 + df[f'sma_gain'] / df[f'sma_loss']))
        df.drop(columns=['interday_gain', 'interday_loss', 'interday_gain_loss', 'sma_gain', 'sma_loss'],
                inplace=True)
        return df

    def signals_rsi_bollinger(self, bollinger_period=20, rsi_period=14, stddev_factor=1):
        try:
            df = self.data.copy()
        except AttributeError:
            print(f"No cache to load for {self.symbol}. Returning None")
            return None
        df = pd.merge(self.get_bollinger_bands(period=bollinger_period, stddev_factor=stddev_factor), df,
                      on='date')
        df = pd.merge(self.get_rsi(rsi_period), df, on='date')
        df = pd.merge(self.get_sma_price(bollinger_period), df, on='date')

        df['signal'] = 'NO-SIGNAL'
        df.loc[(df[f'rsi_{rsi_period}dy'] > 70) & (
                df[f'bollinger_upper_{bollinger_period}dy_{stddev_factor}std'] < df['close']), 'signal'] = 'OVER-BOUGHT'
        df.loc[(df[f'rsi_{rsi_period}dy'] < 30) & (
                df[f'bollinger_lower_{bollinger_period}dy_{stddev_factor}std'] > df['close']), 'signal'] = \
            'UNDER-BOUGHT'
        return df

    def update(self, force_full_output=False):
        """
        Updates a security's open/close data. Note, the environment 'ALPHAVANTAGE_API_KEY' must be set in order for
        an update to occur.
        Param force_full_output: An AlphaVantage API parameter, which when set to true will force AlphaVantage to give
            back all records pertaining to a given security.
            Returns: An integer value:
                0: The update function had an error
                1: The update was successfully performed
                2: There was nothing to update for the security
        """
        today = dt.now()
        if self.latest_date == dt.strftime(dateutils.last_open_date(today), '%Y-%m-%d'):
            logger.info(f"NOP. {self.symbol} up to date")
            return 2

        if (today - dt.strptime(self.latest_date, '%Y-%m-%d')).days >= 100 or force_full_output:
            output_switch = 'full'
        else:
            output_switch = 'compact'

        api_params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.symbol,
            'outputsize': 'full',
            'apikey': os.environ.get("ALPHAVANTAGE_API_KEY")
        }

        try:
            data = requests.get(url=self.API_URL, params=api_params)
            data = data.json()
        except requests.HTTPError as e:
            logger.error(f"Failed to update {self.symbol}", e)
            return 0

        # Format the data: Rename Columns, and store in a records-based format
        try:
            time_series = data['Time Series (Daily)']
        except KeyError as e:
            logger.error(f"Rejected API call due to AlphaVantage rate-limiting: {data}")
            return 0

        df = pd.DataFrame.from_dict(time_series, orient='index').reset_index()

        df.rename(
            columns={'index': 'date', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                     '5. volume': 'volume'}, inplace=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': np.int64})
        df.sort_values('date', ascending=True, inplace=True)

        if output_switch == 'full':
            self.data = df.to_dict(orient='records')
        elif output_switch == 'compact':
            curr_data = []
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    curr_data = json.load(f)
            last_date = dt.strptime(curr_data[-1]['date'], '%Y-%m-%d')
            for new_date in dateutils.daterange(last_date, dt.now(), as_generator=True):
                try:
                    new_data = \
                        df.loc[df['date'] == dt.strftime(new_date, '%Y-%m-%d')].to_dict(orient='records')[0]
                    curr_data.append(new_data)
                except IndexError:
                    # Found a date that the stock was not recorded (NYSE holiday or Weekend)
                    pass
            self.data = curr_data

        with open(self.filename, 'w+') as f:
            f.write(json.dumps(self.data, indent=4))
        self.latest_date = dt.strftime(dt.now(), '%Y-%m-%d')
        return 1
