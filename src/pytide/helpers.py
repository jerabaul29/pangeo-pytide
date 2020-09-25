import datetime
import pytz

import numpy as np
import matplotlib.pyplot as plt

from raise_assert import ras


def assert_is_utc_datetime(date_in):
    """Assert that date_in is an UTC datetime."""
    ras(isinstance(date_in, datetime.datetime))

    if not (date_in.tzinfo == pytz.utc or
            date_in.tzinfo == datetime.timezone.utc):
        raise Exception("not utc!")

    if date_in.tzinfo == pytz.utc:
        print("prefer using datetime.timezone.utc to pytz.utc")


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

def RMSE_stats(observations,
               official_predictions, pytide_predictions,
               explanation=""):
    """Print some RMSE statistics about the comparison between observations,
    official predictions, and pytide predictions.
    Input:
        - observations: real-world observations, np float array
        - *_predictions: the predictions, np float arrays
        - explanation: str to display when printing information.
    """
    RMSE_pytide = \
        np.sqrt(np.mean((observations - pytide_predictions)**2))
    RMSE_predictions = \
        np.sqrt(np.mean((observations - official_predictions)**2))
    RMSE_pytide_predictions = \
        np.sqrt(np.mean((pytide_predictions - official_predictions)**2))

    print("RMSE pytide vs obs {}: {}".format(explanation, RMSE_pytide))
    print("RMSE official prediction vs obs {}: {}".format(explanation, RMSE_predictions))
    print("RMSE pytide vs official predictions {}: {}".format(explanation, RMSE_pytide_predictions))


def show_tides(time_as_datetimes, observations,
               pytide_predictions, official_predictions,
               print_stats=True, explanation="", max_amplitude=1e4):
    """Show the tide observation vs predictions by pytides and official predictions.
    Input:
        - time_as_datetimes: list of datetimes as time base
        - observations: np array of real-world observations
        - *_predictions: np array of predictions.
        - print_stats: whether to print RMSE stats
        - explanation: the explanation to display about the RMSE context
    """
    if print_stats:
        RMSE_stats(observations, official_predictions, pytide_predictions, explanation)

    plt.figure()

    plt.plot(time_as_datetimes, observations, label="observations")
    plt.plot(time_as_datetimes, pytide_predictions, "*", label="pytide prediction")
    plt.plot(time_as_datetimes, official_predictions, "+", label="dataset predictions")
    plt.ylabel("[cm]")

    plt.ylim([-max_amplitude, max_amplitude])

    if explanation != "":
        plt.title(explanation)

    plt.legend(loc="lower right")

    plt.show()
