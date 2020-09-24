import datetime
import pytz

import os
import time

import netCDF4

import pytide

import matplotlib.pyplot as plt

from helpers import show_tides

os.environ["TZ"] = "UTC"
time.tzset()

##############################
# predict against numerical data
if True:
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests", "dataset",
                                "fes_tide_time_series.nc")

    with netCDF4.Dataset(path_to_data) as dataset:
        time_input = dataset['time'][:] * 1e-6    # microseconds to epoch (seconds)
        observations = dataset['ocean'][:]             # TODO: report

    list_utc_datetimes = [pytz.utc.localize(datetime.datetime.fromtimestamp(crrt_timestamp)) for
                          crrt_timestamp in time_input]

    pytide_analyzer = pytide.PyTideAnalyzer(verbose=1)

    # using display=True here would illustrate the fit
    pytide_analyzer.fit_tide_data(list_utc_datetimes, observations, display=False)

    prediction = pytide_analyzer.predict_tide(list_utc_datetimes)

    plt.figure()
    plt.plot(list_utc_datetimes, observations, label="numerical input")
    plt.plot(list_utc_datetimes, prediction, "*", label="pytide prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()

##############################
# predict against real-world data
# load the dataset for the "middle"
if True:
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests", "dataset",
                                "dataset_observation_middle.nc4")

    with netCDF4.Dataset(path_to_data) as dataset:
        time_input = dataset['timestamps'][:]
        observations = dataset['observations'][:]
        official_predictions = dataset['predictions'][:]

    list_utc_datetimes = [pytz.utc.localize(datetime.datetime.fromtimestamp(crrt_timestamp)) for
                          crrt_timestamp in time_input]

    pytide_analyzer = pytide.PyTideAnalyzer(verbose=1)

    # using display=True here would illustrate the fit
    pytide_analyzer.fit_tide_data(list_utc_datetimes, observations, display=False)

    prediction = pytide_analyzer.predict_tide(list_utc_datetimes)

    show_tides(list_utc_datetimes, observations, prediction, official_predictions,
               explanation="BGO middle dataset")

##############################
# this now also works against datasets with bad data, given right flags
if True:
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests", "dataset",
                                "dataset_observations_contains_invalid.nc4")

    with netCDF4.Dataset(path_to_data) as dataset:
        time_input = dataset['timestamps'][:]
        observations = dataset['observations'][:]
        official_predictions = dataset['predictions'][:]

    list_utc_datetimes = [pytz.utc.localize(datetime.datetime.fromtimestamp(crrt_timestamp)) for
                          crrt_timestamp in time_input]

    pytide_analyzer = pytide.PyTideAnalyzer(verbose=1)

    # using display=True here would illustrate the fit
    pytide_analyzer.fit_tide_data(list_utc_datetimes, observations, display=True, clean_signals=True)

    prediction = pytide_analyzer.predict_tide(list_utc_datetimes)

    show_tides(list_utc_datetimes, observations, prediction, official_predictions,
               explanation="BGO dataset with invalid")
