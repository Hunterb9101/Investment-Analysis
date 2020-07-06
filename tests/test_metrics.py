import pytest
from invest.security import Security
import pandas as pd
from pandas.testing import assert_frame_equal

class TestMetrics:
    def test_intraday_gain_loss(self):
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        # Round to 3 digits of precision
        intraday = s.get_intraday_gain_loss().round(3)
        assert s.data[['intraday_gain_loss', 'intraday_gain_loss_pct']].equals(
            intraday[['intraday_gain_loss', 'intraday_gain_loss_pct']])

    def test_interday_gain_loss(self):
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        # Round to 3 digits of precision
        interday = s.get_interday_gain_loss().round(3)
        assert_frame_equal(s.data[['interday_gain_loss', 'interday_gain_loss_pct']],
            interday[['interday_gain_loss', 'interday_gain_loss_pct']], check_dtype=False)

    def test_sma(self):
        s = Security('NULL')
        s.data = pd.read_json('test_data/example_data_1.json')
        sma = s.get_sma(period=5)
        assert_frame_equal(pd.DataFrame(s.data['sma_5dy_pct']), pd.DataFrame(sma['sma_5dy_pct']))

    def test_ema(self):
        pass

    def test_bollinger_bands(self):
        pass

    def test_rsi(self):
        pass