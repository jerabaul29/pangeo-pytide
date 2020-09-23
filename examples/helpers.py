import datetime
import pytz

from external_water_level_stations.helper.raise_assert import ras


def assert_is_utc_datetime(date_in):
    """Assert that date_in is an UTC datetime."""
    ras(isinstance(date_in, datetime.datetime))

    if not (date_in.tzinfo == pytz.utc or
            date_in.tzinfo == datetime.timezone.utc):
        raise Exception("not utc!")

    if date_in.tzinfo == pytz.utc:
        print("prefer using datetime.timezone.utc to pytz.utc")


def assert_10min_multiple(date_in):
    """Assert that date_in is a datetime that is a
    multiple of 10 minutes.
    """
    ras(isinstance(date_in, datetime.datetime))
    ras(date_in.second == 0)
    ras((date_in.minute % 10) == 0)
    ras(date_in.microsecond == 0)


def datetime_range(datetime_start, datetime_end, step_timedelta):
    """Yield a datetime range, in the range [datetime_start; datetime_end[,
    with step step_timedelta."""
    assert_is_utc_datetime(datetime_start)
    assert_is_utc_datetime(datetime_end)
    ras(isinstance(step_timedelta, datetime.timedelta))
    ras(datetime_start < datetime_end)
    ras(step_timedelta > datetime.timedelta(0))

    crrt_time = datetime_start
    yield crrt_time

    while True:
        crrt_time += step_timedelta
        if crrt_time < datetime_end:
            yield crrt_time
        else:
            break
