import json
import os
import pandas as pd
from security import Security
import logging
import time

logger = logging.getLogger("main")


class WatchList:
    watchlist_dir = 'watchlists/'

    def __init__(self, name, security_list=None):
        self.file = os.path.join(WatchList.watchlist_dir, f'{name}.json')
        empty = True
        if os.path.exists(self.file):
            with open(self.file) as f:
                self._watchlist = json.load(f)
                logger.info(f"Loading watchlist {name}")
                empty = False
        else:
            self._watchlist = {}

        if security_list is not None:
            for symbol in security_list:
                if symbol not in self._watchlist.keys():
                    self.add(symbol)

        if empty and security_list is None:
            logger.error(f"Attempting to create an empty watchlist {name}!")
            raise ValueError(f"Attempting to create an empty watchlist {name}!")

    def _save(self):
        with open(self.file, 'w+') as f:
            f.write(json.dumps(self._watchlist, indent=4))

    def list(self):
        return sorted(list(self._watchlist.keys()))

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

    def update_values(self, security, update_priority):
        update_priority_opts = ['ALWAYS', 'FILL']
        if update_priority not in update_priority_opts:
            raise ValueError(f"Invalid parameter for update_priority: {update_priority}. "
                             f"Use a value in [{','.join(update_priority_opts)}]")
        self._watchlist[security]['update_priority'] = update_priority
        self._save()

    def update_securities(self, force_full_output=False, max_tries=6):
        priority = ['ALWAYS', 'FILL']
        if len(self._watchlist.keys()) > 500:
            logger.warning("Too many securities for API Request. Truncating to 500 oldest-updated reports")
            updated = []
            for symbol in self._watchlist.keys():
                if os.path.exists(os.path.join(Security.STOCK_DIR, f'{symbol}.json')):
                    with open(os.path.join(Security.STOCK_DIR, f'{symbol}.json')) as f:
                        updated.append({"symbol": symbol, "updated_at": json.load(f)[-1]['date'],
                                        'priority': self._watchlist[symbol]['update_priority']})
                else:
                    updated.append({"symbol": symbol, "updated_at": '1970-01-01',
                                    'priority': self._watchlist[symbol]['update_priority']})
            updated = sorted(updated, key=lambda x: (priority.index(x['priority']), x['updated_at']))
            securities = [u['symbol'] for u in updated][:500]

        logger.info(f"Collected securities list: {', '.join(securities)}")
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
                    logger.info("Waiting 12 seconds...")
                    time.sleep(12)

    def rsi_bollinger_report(self):
        df = pd.DataFrame()
        for w in self.list():
            df_signals = Security(w).signals_rsi_bollinger()
            if df_signals is None:
                continue
            row = {'security': w,
                   'last_date_overbought': df_signals[df_signals['signal'] == 'OVER-BOUGHT']['date'].max(),
                   'last_date_underbought': df_signals[df_signals['signal'] == 'UNDER-BOUGHT']['date'].max(),
                   'current_sma_price': df_signals[df_signals['date'].max() == df_signals['date']]
                   ['sma_price_20dy'].tolist()[0],
                   'current_price': df_signals[df_signals['date'].max() == df_signals['date']]['close'].tolist()[0],
                   'last_updated': df_signals['date'].max()}
            df_new = pd.DataFrame.from_dict([row])
            df = pd.concat([df, df_new])
        return df.reset_index(drop=True)


def get_sec_tickers():
    """Get list of tickers from SEC. (As of 6/30/20)"""
    return sorted(pd.read_json("https://www.sec.gov/files/company_tickers.json", orient='index')['ticker'].tolist())


def get_sp500_tickers():
    """Get list of S&P500 index from Wikipedia. (As of 6/30/20)"""
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    tickers.remove('BF.B')
    tickers.append('BFB')
    return sorted(tickers)


if __name__ == '__main__':
    with open('config.json') as f:
        os.environ['ALPHAVANTAGE_API_KEY'] = json.load(f)['ALPHAVANTAGE_API_KEY']

    # Create a watchlist
    stakes = ["JPM", "XOM", "INTC", "QQQ", "XLE", "XLV", "ARKF", "SCHA"]
    data = get_sp500_tickers()
    data.extend(stakes)

    # TODO - Make prioritization what gets updated for WatchList (ALWAYS / FILL).
    to_watch = WatchList('EquityWatchList', list(set(data)))
    for s in stakes:
        to_watch.update_values(s, 'ALWAYS')

    to_watch.update_securities(force_full_output=True)
