import invest.dateutils
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
        assert invest.dateutils.nyse_open(dt.strptime(date, '%Y%m%d')) == expected

    @pytest.mark.parametrize(
        'date,expected',
        [
            ('20200606', '20200605'),  # Saturday
            ('20200607', '20200605'),  # Sunday
            ('20200101', '20191231'),  # Holiday
            ('20200715', '20200715')  # Weekday
        ]
    )
    def test_last_open_date(self, date, expected):
        assert invest.dateutils.last_open_date(dt.strptime(date, '%Y%m%d')) == dt.strptime(expected, '%Y%m%d')
