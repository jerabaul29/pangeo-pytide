# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import datetime
import os
import time
import numpy
import numpy as np
import unittest
import netCDF4
import pytide
import pytz


class AstronomicAngle(unittest.TestCase):
    def test_init(self):
        aa = pytide.AstronomicAngle(datetime.datetime(2000, 1, 1))
        self.assertTrue(isinstance(aa, pytide.AstronomicAngle))
        self.assertAlmostEqual(aa.h, 4.886452089967941, delta=1e-6)
        self.assertAlmostEqual(aa.n, 2.182860931126595, delta=1e-6)
        self.assertAlmostEqual(aa.nu, 0.20722218671046477, delta=1e-6)
        self.assertAlmostEqual(aa.nuprim, 0.13806065629468897, delta=1e-6)
        self.assertAlmostEqual(aa.nusec, 0.13226438100551682, delta=1e-6)
        self.assertAlmostEqual(aa.p, 1.4537576754171415, delta=1e-6)
        self.assertAlmostEqual(aa.p1, 4.938242223271549, delta=1e-6)
        self.assertAlmostEqual(aa.r, 0.1010709894525481, delta=1e-6)
        self.assertAlmostEqual(aa.s, 3.6956255851976114, delta=1e-6)
        self.assertAlmostEqual(aa.t, 3.1415926536073755, delta=1e-6)
        self.assertAlmostEqual(aa.x1ra, 1.1723206438502318, delta=1e-6)
        self.assertAlmostEqual(aa.xi, 0.1920359426758722, delta=1e-6)


class Wave(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            pytide.Wave()


class WaveTable(unittest.TestCase):
    DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dataset", "fes_tide_time_series.nc")

    def test_init(self):
        wt = pytide.WaveTable()
        self.assertEqual(len(wt), 67)
        self.assertEqual(len([item for item in wt]), 67)
        self.assertEqual(wt.wave("M2"), wt.wave(pytide.Wave.Ident.kM2))
        self.assertNotEqual(wt.wave("M2"), wt.wave(pytide.Wave.Ident.kK1))
        self.assertTrue(wt.wave("__M2__") is None)
        self.assertListEqual(sorted(wt.known_constituents()),
                             sorted([item.name() for item in wt]))
        for item in wt:
            self.assertEqual(item.ident,
                             getattr(pytide.Wave.Ident, "k" + item.name()))

        wt = pytide.WaveTable(["M2", "K1", "O1", "P1", "Q1", "S1"])
        self.assertEqual(len(wt), 6)
        self.assertListEqual(sorted([item.name() for item in wt]),
                             sorted(["M2", "K1", "O1", "P1", "Q1", "S1"]))

    def test_wave(self):
        aa = pytide.AstronomicAngle(datetime.datetime(2000, 1, 1))
        wt = pytide.WaveTable(["M2"])
        wave = wt.wave("M2")
        self.assertAlmostEqual(wave.freq * 86400,
                               12.140833182614747,
                               delta=1e-6)
        self.assertEqual(wave.type, wave.TidalType.kShortPeriod)

    def test_degraded(self):
        with netCDF4.Dataset(self.DATASET) as dataset:
            time = dataset['time'][:] * 1e-6
            h = dataset['ocean'][:] * 1e-2

        wt = pytide.WaveTable()

        wt.compute_nodal_modulations(
            [datetime.datetime(2012, 1, 1),
             datetime.datetime(2012, 1, 2)])
        wt.compute_nodal_modulations(
            numpy.array([
                numpy.datetime64("2012-01-01"),
                numpy.datetime64("2012-01-02")
            ]))

        with self.assertRaises(TypeError):
            wt.compute_nodal_modulations(datetime.datetime(2012, 1, 1))

        with self.assertRaises(TypeError):
            wt.compute_nodal_modulations(time)

        with self.assertRaises(TypeError):
            wt.compute_nodal_modulations([3])

        with self.assertRaises(ValueError):
            wt.compute_nodal_corrections(3)

    def test_analysis(self):
        with netCDF4.Dataset(self.DATASET) as dataset:
            time = (dataset['time'][:] * 1e-6).astype("datetime64[s]")
            h = dataset['ocean'][:] * 1e-2

        wt = pytide.WaveTable()
        f, vu = wt.compute_nodal_modulations(time)
        w = wt.harmonic_analysis(h, f, vu)
        delta = h - wt.tide_from_tide_series(time, w)

        self.assertAlmostEqual(delta.mean(), 0, delta=1e-16)
        self.assertAlmostEqual(delta.std(), 0, delta=1e-12)

class TestPyTideAnalyzer(unittest.TestCase):
    dataset_numeric = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "dataset", "fes_tide_time_series.nc")

    dataset_real_world = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "dataset", "dataset_observation_middle.nc4")

    os.environ["TZ"] = "UTC"
    time.tzset()

    def test_against_simulation(self):
        pytide_analyzer = pytide.PyTideAnalyzer()

        with netCDF4.Dataset(self.dataset_numeric) as dataset:
            time_data = dataset['time'][:] * 1e-6    # microseconds to epoch (seconds)
            h = dataset['ocean'][:]

        list_utc_datetimes = [pytz.utc.localize(datetime.datetime.fromtimestamp(crrt_timestamp)) for
                              crrt_timestamp in time_data]

        pytide_analyzer.fit_tide_data(list_utc_datetimes, h, display=False)
        prediction = pytide_analyzer.predict_tide(list_utc_datetimes)

        tolerance = 1e-3

        all_within_tolerance = np.all(prediction - h < tolerance)
        assert all_within_tolerance

    def test_agains_real_world(self):
        with netCDF4.Dataset(self.dataset_real_world) as dataset:
            time_input = dataset['timestamps'][:]
            observations = dataset['observations'][:]
            official_predictions = dataset['predictions'][:]

        list_utc_datetimes = [pytz.utc.localize(datetime.datetime.fromtimestamp(crrt_timestamp)) for
                              crrt_timestamp in time_input]

        pytide_analyzer = pytide.PyTideAnalyzer(verbose=0)

        # using display=True here would illustrate the fit
        pytide_analyzer.fit_tide_data(list_utc_datetimes, observations, display=False, clean_signals=True)
        prediction = pytide_analyzer.predict_tide(list_utc_datetimes)

        def np_RMSE(arr_1, arr_2):
            return np.sqrt(np.mean((arr_1 - arr_2)**2))

        RMSE_prediction_official = np_RMSE(prediction, official_predictions)
        RMSE_prediction_real = np_RMSE(prediction, observations)

        assert RMSE_prediction_official < 4.0
        assert RMSE_prediction_real < 14.0

if __name__ == "__main__":
    unittest.main()
