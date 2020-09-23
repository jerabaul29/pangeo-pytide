import pytide
import os
import time as time_module
import matplotlib.pyplot as plt
import netCDF4
import datetime
import pytz
import numpy as np

from helpers import datetime_range

# Make sure the interpreter is in UTC in all the following
os.environ["TZ"] = "UTC"
time_module.tzset()

##################################################
##################################################
# example with simulation data

if False:
    # load the dataset
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests", "dataset",
                                "fes_tide_time_series.nc")

    with netCDF4.Dataset(path_to_data) as dataset:
        time = dataset['time'][:] * 1e-6    # microseconds to epoch (seconds)
        h = dataset['ocean'][:]             # TODO: report

    time_as_datetimes = [datetime.datetime.fromtimestamp(crrt_timestamp) for
                        crrt_timestamp in time]

    # build pytide wave table with all modes
    wt = pytide.WaveTable()

    # compute the nodal modulations corresponding to the times
    f, vu = wt.compute_nodal_modulations(time_as_datetimes)

    # build modes table from the available records
    w = wt.harmonic_analysis(h, f, vu)

    # predict over a time range TODO: why print datetimes?
    datetime_prediction = \
        list(datetime_range(pytz.utc.localize(time_as_datetimes[0]),
                            pytz.utc.localize(time_as_datetimes[-1]),
                            datetime.timedelta(minutes=10)))

    time_prediction = [crrt_datetime.timestamp() for
                    crrt_datetime in datetime_prediction]

    hp = wt.tide_from_tide_series(time_prediction, w)

    # show the results
    plt.figure()

    plt.plot(time_as_datetimes, h, label="data")
    plt.plot(datetime_prediction, hp, "*", label="model fit")

    plt.legend(loc="lower right")

    plt.show()

##################################################
##################################################
# example with real-world data, including both
# observations and some "official" predictions

if True:
    # load the dataset
    path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests", "dataset",
                                "dataset_observation.nc4")

    with netCDF4.Dataset(path_to_data) as dataset:
        time = dataset['timestamps'][:]
        h = dataset['observations'][:]
        predictions = dataset['predictions'][:]

    mean_h = np.mean(h)
    # h = h - np.mean(h)
    # predictions = predictions - np.mean(predictions)
    h = h - mean_h
    predictions = predictions - mean_h

    time_as_datetimes = [datetime.datetime.fromtimestamp(crrt_timestamp) for
                         crrt_timestamp in time]

    index_half_data = len(time) // 2

    # build pytide wave table with all modes
    wt = pytide.WaveTable()

    # compute the nodal modulations corresponding to the times
    # based on half of the data
    f, vu = wt.compute_nodal_modulations(time_as_datetimes[:index_half_data])

    # build modes table from the available records
    # based on half of the data
    w = wt.harmonic_analysis(h[:index_half_data],
                             f[:index_half_data],
                             vu[:index_half_data])

    # predict over a time range, for the full dataset
    hp = wt.tide_from_tide_series(time, w)

    # print some statistics

    # on half of the data
    RMSE_pytide = np.sqrt(np.mean((h[:index_half_data] - hp[:index_half_data])**2))
    RMSE_predictions = np.sqrt(np.mean((h[:index_half_data] - predictions[:index_half_data])**2))
    RMSE_pytide_predictions = np.sqrt(np.mean((hp[:index_half_data] - predictions[:index_half_data])**2))

    print("RMSE of pytide on half data used for training: {}".format(RMSE_pytide))
    print("RMSE predictions on half data used for training: {}".format(RMSE_predictions))
    print("RMSE pytide vs predictions on half data used for training: {}".format(RMSE_pytide_predictions))

    # on the full data
    RMSE_pytide = np.sqrt(np.mean((h - hp)**2))
    RMSE_predictions = np.sqrt(np.mean((h - predictions)**2))
    RMSE_pytide_predictions = np.sqrt(np.mean((hp - predictions)**2))

    print("RMSE of pytide on full data: {}".format(RMSE_pytide))
    print("RMSE predictions on full data: {}".format(RMSE_predictions))
    print("RMSE pytide vs predictions on full data: {}".format(RMSE_pytide_predictions))

    # show the results
    plt.figure()

    plt.plot(time_as_datetimes, h, label="data")
    plt.plot(time_as_datetimes, hp, "*", label="pytide prediction")
    plt.plot(time_as_datetimes, predictions, "+", label="dataset predictions")

    plt.legend(loc="lower right")

    plt.show()

     # TODO: add a fragment around year 2019
     # TODO: add a fragment around year 1980
     # TODO: perform fitting also when data missing etc, by taking care of missing data

# TODO: make into a jupyter notebook for ease of reading
# TODO: write a super-simple class-based API "for dummies"
# TODO: make sure manage to install locally for testing purposes
