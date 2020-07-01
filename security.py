import requests
import json
import pandas as pd
import utils
import time
import numpy as np
import logging
import os
from datetime import datetime as dt

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s: %(message)s'))
logger.addHandler(sh)
logger.setLevel(logging.INFO)


class Security:
    def __init__(self, symbol: str):
        self.API_URL = 'https://www.alphavantage.co/query'
        self.symbol = symbol
        if os.path.exists(f'data/{symbol}.json'):
            with open(f'data/{symbol}.json') as f:
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

    def update(self, force_full_output=False):
        """
        Updates a security's open/close data. Note, the environment 'ALPHAVANTAGE_API_KEY' must be set in order for
        an update to occur.
        """
        today = dt.now()

        if (today - dt.strptime(self.latest_date, '%Y-%m-%d')).days >= 100 or force_full_output:
            output_switch = 'full'
        elif self.latest_date == dt.strftime(today, '%Y-%m-%d'):
            logger.info(f"NOP. {self.symbol} up to date")
            return 1
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
            logger.error("Failed to update {self.symbol}", e)
            return 0

        # Format the data: Rename Columns, and store in a records-based format
        try:
            time_series = data['Time Series (Daily)']
        except:
            logger.error("Bad Request")
            return 0

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.reset_index(inplace=True)

        df.rename(
            columns={'index': 'date', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                     '5. volume': 'volume'}, inplace=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': np.int64})

        # Get gain/loss statistics for Exponential/Simple Moving average (EMA/SMA) statistics
        df['gain/loss'] = df['close'].diff(periods=1).fillna(0.00)
        df['gain_close'] = df['gain/loss'].where(df['gain/loss'] > 0)
        # Loss values reported as postive for RSI
        df['loss_close'] = -1 * df['gain/loss'].where(df['gain/loss'] < 0)
        df[['gain_close', 'loss_close']] = df[['gain_close', 'loss_close']].fillna(0.00)

        # Get Short/Medium/Long values for SMA and RSI.
        periods = [
            {'name': 'short', 'period': 10},
            {'name': 'medium', 'period': 20},
            {'name': 'long', 'period': 50}
        ]
        for period in periods:
            # SMA: Needed for Relative Strength Index (RSI) calculations
            df[['sma_gain_{}'.format(period['name']), 'sma_loss_{}'.format(period['name'])]] = \
                df[['gain_close', 'loss_close']].rolling(window=period['period'], min_periods=1).mean()
            df['std_close_{}'.format(period['name'])] = df['close'].rolling(
                window=period['period'], min_periods=1).std()

            # EMA: Used to calculate Bollinger Bands
            df['ema_close_{}'.format(period['name'])] = df['close'].ewm(
                span=period['period']).mean()

            # Relative Strength Index (RSI) Calculation
            df[f'rsi_{period["name"]}'] = 100 - (100 / (1 + df[f'sma_gain_{period["name"]}'] /
                                                        df[f'sma_loss_{period["name"]}']))

        # Put in dictionary form to put into JSON format
        time_series_dict = df.to_dict(orient='records')

        if output_switch == 'full':
            self.data = time_series_dict
        elif output_switch == 'compact':
            curr_data = []
            with open(f"data/data_{self.symbol}.json") as f:
                curr_data = json.load(f)
            last_date = dt.strptime(curr_data[-1]['date'], '%Y-%m-%d')

            for new_date in utils.daterange(last_date, dt.now(), as_generator=True):
                try:
                    new_data = \
                        df.loc[df['date'] == dt.strftime(new_date, '%Y-%m-%d')].to_dict(orient='records')[0]
                    curr_data.append(new_data)
                except IndexError:
                    pass  # Found a date that the stock was not recorded (NYSE holiday or Weekend)
            self.data = curr_data

        with open(f"data/{self.symbol}.json", 'w+') as f:
            f.write(json.dumps(self.data, indent=4))
        self.latest_date = dt.strftime(dt.now(), '%Y-%m-%d')
        return 1

    @staticmethod
    def update_from_list(securities: list, force_full_output=False, max_tries=5):
        for symbol in securities:
            s = Security(symbol)
            failed = 0
            request = 0
            while request == 0 and failed < max_tries:
                if failed != 0:
                    logger.warning(f"Failed {failed} out of {max_tries} times for {symbol}")
                request = s.update(force_full_output)
                failed += 1
                logger.info("Waiting 12 seconds...")
                time.sleep(12)


def get_sec_tickers():
    """Get list of tickers from SEC. (As of 6/30/20)"""
    return sorted(pd.read_json("https://www.sec.gov/files/company_tickers.json", orient='index')['ticker'].tolist())


def get_sp500_tickers():
    """Get list of S&P500 index from Wikipedia. (As of 6/30/20)"""
    return sorted(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist())


if __name__ == '__main__':
    with open('config.json') as f:
        os.environ['ALPHAVANTAGE_API_KEY'] = json.load(f)['ALPHAVANTAGE_API_KEY']
    Security.update_from_list(['MSFT', 'QQQ'], force_full_output=True)
