import invest.dateutil
import pytest
from datetime import datetime as dt


class TestUtils:
    @pytest.mark.parametrize(
        'date,expected',
        [('20200606', False),  # Weekend
         ('20200101', False),  # Holiday
         ('20200715', True)  # Weekday
         ]
    )
    def test_nyse_open(self, date, expected):
        assert invest.dateutil.nyse_open(dt.strptime(date, '%Y%m%d')) == expected