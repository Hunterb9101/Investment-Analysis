import pytest
from invest.security import Security
import pandas as pd
from pandas.testing import assert_frame_equal


class TestMetrics:
    def test_intraday_gain_loss(self):
        """
        Checks correctness of gain/loss intraday values/percentages of an equity.
        """
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        intraday = s.get_intraday_gain_loss().round(3)
        assert s.data[['intraday_gain_loss', 'intraday_gain_loss_pct']].equals(
            intraday[['intraday_gain_loss', 'intraday_gain_loss_pct']])

    def test_interday_gain_loss(self):
        """
        Checks correctness of gain/loss interday values/percentages of an equity.
        """
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        interday = s.get_interday_gain_loss().round(3)
        assert_frame_equal(s.data[['interday_gain_loss', 'interday_gain_loss_pct']],
            interday[['interday_gain_loss', 'interday_gain_loss_pct']], check_dtype=False)

    def test_sma_pct_chg(self):
        """
        Checks the functionality of the 5-day SMA rolling average on interday percentage price change of an equity.
        """
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        sma = s.get_sma(period=5).round(3)
        assert_frame_equal(pd.DataFrame(s.data['sma_5dy_pct'], dtype='float64'), pd.DataFrame(sma['sma_5dy_pct']),
                           check_dtype=False)

    def test_sma_price(self):
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        sma = s.get_sma_price(period=5).round(3)
        assert_frame_equal(pd.DataFrame(s.data['sma_price_5dy'], dtype='float64'), pd.DataFrame(sma['sma_price_5dy']),
                           check_dtype=False)


    def test_ema(self):
        pass

    def test_bollinger_bands(self):
        pass

    def test_rsi(self):
        pass