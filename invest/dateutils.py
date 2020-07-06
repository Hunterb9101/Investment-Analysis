from datetime import timedelta, datetime


def daterange(start_date: datetime, end_date: datetime, as_generator=False):
    arr = []
    for n in range(int ((end_date - start_date).days)):
        if as_generator:
            yield start_date + timedelta(n+1)
        else:
            arr.append(start_date + timedelta(n+1))
    return arr


def nyse_open(dt: datetime):
    """ Returns if the NYSE is open given datetime element `dt`"""
    nyse_holidays = ['20200101', '20200120', '20200217', '20200410', '20200525', '20200703', '20200907', '20201126',
                     '20201225', '20210101', '20210118', '20210215', '20210402', '20210531', '20210705', '20210906',
                     '20211125', '20211224', '20220117', '20220221', '20220415', '20220530', '20220704', '20220905',
                     '20221124', '20221226']

    # It is the weekend
    if dt.weekday() >= 5 or dt.strftime('%Y%m%d') in nyse_holidays:
        return False
    return True


def last_open_date(dt: datetime):
    """ Returns the earliest date that the New York Stock Exchange (NYSE) was open since `dt`"""
    while True:
        if nyse_open(dt):
            return dt
        dt -= timedelta(days=1)
