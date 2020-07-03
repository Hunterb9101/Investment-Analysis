import requests
import json
import pandas as pd
import utils
import time
import numpy as np
import logging
import os
from glob import glob
from datetime import datetime as dt

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
logger.addHandler(sh)
logger.setLevel(logging.INFO)

STOCK_DIR = 'data'


class Security:
    def __init__(self, symbol: str):
        self.API_URL = 'https://www.alphavantage.co/query'
        self.symbol = symbol
        self.filename = os.path.join(STOCK_DIR, f'{symbol}.json')
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                try:
                    self.data = sorted(json.load(f), key=lambda x: x['date'])
                    self.latest_date = self.data[-1]['date']
                except json.decoder.JSONDecodeError:
                    self.data = None
                    self.latest_date='1970-01-01'
                    logger.warning(f"{symbol} cache file is empty. Was it overwritten?")

            logger.info(f"Loaded {symbol} from cache")
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
        df['interday_gain_close'] = df['interday_gain_loss'].where(df['interday_gain_loss'] > 0)
        # Loss values reported as postive for RSI
        df['interday_loss_close'] = -1 * df['interday_gain_loss'].where(df['interday_gain_loss'] < 0)
        df[['interday_gain_close', 'interday_loss_close']] = df[['interday_gain_close', 'interday_loss_close']].fillna(0.00)
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
        df['intraday_gain_close'] = df['intraday_gain_loss'].where(df['intraday_gain_loss'] > 0)
        df['intraday_loss_close'] = -1 * df['intraday_gain_loss'].where(df['intraday_gain_loss'] < 0)
        df[['intraday_gain_close', 'intraday_loss_close']] = df[['intraday_gain_close', 'intraday_loss_close']].fillna(0.00)
        df.drop(columns=['open', 'close'], inplace=True)
        return df

    def get_sma(self, period=10):
        """
        Calculates a Simple Moving Average (SMA) using interday gain/loss data.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and a column with an SMA value.
        """
        df = self.get_interday_gain_loss()
        df[f'sma_close_{period}dy'] = df['interday_gain_loss'].rolling(window=period, min_periods=1).mean()
        df.drop(columns=[c for c in df.columns.values if 'interday_' in c], inplace=True)
        return df

    def get_ema(self, period=10):
        """
        Calculates an Exponential Moving Average (EMA) using interday gain/loss data.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and a column with an EMA value.
        """
        df = self.data[['date', 'close']].copy()
        df[f'ema_close_{period}dy'] = df['close'].ewm(span=period).mean()
        df.drop('close', inplace=True)
        return df

    def get_bollinger_bands(self, period=20, stddev_factor=1, method='SMA'):
        """
        Calculates an upper and lower Bollinger Band. It uses either an SMA or an EMA, and adds/subtracts a windowed
        standard deviation from it.
        Param period: An integer value describing the window of the moving average and standard deviation
        Param stddev_factor: Use a variable sized standard deviation in the Bollinger Bands
        Param method: Use SMA or EMA weightings
        Returns: A dataframe keyed on the date and columns for an upper and lower Bollinger Band.
        """
        method = method.lower()
        if method == 'sma':
            to_merge = self.get_sma(period)
        elif method == 'ema':
            to_merge = self.get_ema(period)
        else:
            raise ValueError("Invalid method! Use SMA or EMA.")
        df = self.data[['date', 'close']].copy().merge(to_merge, on='date')
        df[f'std_close_{period}dy'] = df['close'].rolling(window=period, min_periods=1).std()

        df[f'bollinger_upper_{method.lower()}_{period}dy_{stddev_factor}std'] = \
            stddev_factor * df[f'std_close_{period}dy'] + df[f'{method}_close_{period}dy']
        df[f'bollinger_lower_{method.lower()}_{period}dy_{stddev_factor}std'] = \
            stddev_factor * df[f'std_close_{period}dy'] - df[f'{method}_close_{period}dy']

        df.drop(columns=['close', f'std_close_{period}dy', f'sma_close_{period}dy'], inplace=True)
        return df

    def get_rsi(self, period=20):
        """
        Calculates the Relative Strength Index (RSI). A value lower than 30 means the stock is under-bought, and 70 is
        over-bought.
        Param period: An integer value describing the window of the moving average
        Returns: A dataframe keyed on the date and an RSI column
        """
        df = self.get_interday_gain_loss()
        df[['sma_gain', 'sma_loss']] = df[['interday_gain_close', 'interday_loss_close']]\
            .rolling(window=period, min_periods=1).mean()
        df[f'rsi_{period}dy'] = 100 - (100 / (1 + df[f'sma_gain'] / df[f'sma_loss']))
        df.drop(columns=['interday_gain_close', 'interday_loss_close', 'interday_gain_loss', 'sma_gain', 'sma_loss'],
                inplace=True)
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

        if self.latest_date == dt.strftime(today, '%Y-%m-%d'):
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
        except KeyError:
            logger.error("Rejected API call due to AlphaVantage rate-limiting.")
            return 0

        df = pd.DataFrame.from_dict(time_series, orient='index').reset_index()

        df.rename(
            columns={'index': 'date', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                     '5. volume': 'volume'}, inplace=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': np.int64})
        df.sort_values('date', ascending=True, inplace=True)

        # TODO: Move SMA/EMA/Bollinger Bands/RSI calculations downstream
        # df = df.merge(self.get_intraday_gain_loss(), on='date')
        # for period in [10, 20, 50]:
        #     df = df.merge(self.get_sma(period), on='date')
        #     df = df.merge(self.get_bollinger_bands(period), on='date')
        #     df = df.merge(self.get_rsi(period), on='date')

        if output_switch == 'full':
            self.data = df.to_dict(orient='records')
        elif output_switch == 'compact':
            curr_data = []
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    curr_data = json.load(f)
            last_date = dt.strptime(curr_data[-1]['date'], '%Y-%m-%d')
            for new_date in utils.daterange(last_date, dt.now(), as_generator=True):
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

    @staticmethod
    def update_from_list(securities: list, force_full_output=False, max_tries=5):
        if len(securities) > 500:
            logger.warning("Too many securities for API Request. Truncating to 500 oldest-updated reports")
            updated = []
            for symbol in securities:
                if os.path.exists(os.path.join(STOCK_DIR, f'{symbol}.json')):
                    with open(os.path.join(STOCK_DIR, f'{symbol}.json')) as f:
                        updated.append({"symbol": symbol, "updated_at": json.load(f)[-1]['date']})
                else:
                    updated.append({"symbol": symbol, "updated_at": '1970-01-01'})
            updated = sorted(updated, key=lambda x: x['updated_at'])
            securities = [u['symbol'] for u in updated][:500]

        for symbol in securities:
            s = Security(symbol)
            failed = 0
            request = 0
            while request == 0 and failed < max_tries:
                if failed != 0:
                    logger.warning(f"Failed {failed} out of {max_tries} times for {symbol}")
                request = s.update(force_full_output)
                failed += 1
                if request == 0:
                    logger.info("Waiting 6 seconds...")
                    time.sleep(6)


class WatchList:
    watchlist_dir = 'watchlists/'

    def __init__(self, name, security_list):
        self.file = os.path.join(WatchList.watchlist_dir, f'{name}.json')
        if os.path.exists(self.file):
            with open(self.file) as f:
                self._watchlist = json.load(f)
        else:
            self._watchlist = {}

        for symbol in security_list:
            if symbol not in self._watchlist.keys():
                self.add(symbol)

    def _save(self):
        with open(self.file, 'w+') as f:
            f.write(json.dumps(self._watchlist, indent=4))

    def add(self, security):
        if security in self._watchlist.keys():
            raise ValueError(f"{security} already exists in this watchlist!")
        self._watchlist[security] = {"update_priority": "FILL"}
        self._save()

    def remove(self, security):
        if security not in self._watchlist.keys():
            raise ValueError(f"{security} doesn't exist in this watchlist!")
        del self._watchlist[security]
        self._save()

    def update(self, security, update_priority):
        update_priority_opts = ['ALWAYS', 'FILL']
        if update_priority not in update_priority_opts:
            raise ValueError(f"Invalid parameter for update_priority: {update_priority}. "
                             f"Use a value in [{','.join(update_priority_opts)}]")
        self._watchlist[security]['update_priority'] = update_priority
        self._save()


def get_sec_tickers():
    """Get list of tickers from SEC. (As of 6/30/20)"""
    return sorted(pd.read_json("https://www.sec.gov/files/company_tickers.json", orient='index')['ticker'].tolist())


def get_sp500_tickers():
    """Get list of S&P500 index from Wikipedia. (As of 6/30/20)"""
    return sorted(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist())


if __name__ == '__main__':
    with open('config.json') as f:
        os.environ['ALPHAVANTAGE_API_KEY'] = json.load(f)['ALPHAVANTAGE_API_KEY']

    # Create a watchlist
    stakes = ["JPM", "XOM", "INTC", "QQQ", "XLE", "XLV", "ARKF", "SCHA"]
    #data = get_sp500_tickers()
    #data.extend(stakes)

    # TODO - Make prioritization what gets updated for WatchList (ALWAYS / FILL).
    to_watch = WatchList('ExampleWatchList', list(set(stakes)))
    for s in stakes:
        to_watch.update(s, 'ALWAYS')

    Security.update_from_list(stakes, force_full_output=True)
