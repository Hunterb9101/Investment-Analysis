from datetime import timedelta, date


def daterange(start_date, end_date, as_generator=False):
    arr = []
    for n in range(int ((end_date - start_date).days)):
        if as_generator:
            yield start_date + timedelta(n+1)
        else:
            arr.append(start_date + timedelta(n+1))
    return arr
