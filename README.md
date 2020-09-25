[![conda](https://anaconda.org/conda-forge/pytide/badges/installer/conda.svg?service=github)](https://www.anaconda.com/distribution/)
[![platforms](https://anaconda.org/conda-forge/pytide/badges/platforms.svg?service=github)](https://anaconda.org/conda-forge/pytide)
[![latest-release-date](https://anaconda.org/conda-forge/pytide/badges/latest_release_date.svg?service=github)](https://github.com/CNES/pangeo-pytide/commits/master)
[![license](https://anaconda.org/conda-forge/pytide/badges/license.svg?service=github)](https://opensource.org/licenses/BSD-3-Clause)

# pangeo-pytide

## About

`pytide` allows to analyze the tidal constituents of a time series from a
[harmonic
analysis](https://pangeo-pytide.readthedocs.io/en/latest/pytide.html#pytide.WaveTable.harmonic_analysis).
The definition of tidal constants and astronomical arguments is taken from
[FES2014 tidal prediction
software](https://bitbucket.org/cnes_aviso/fes/src/master/).

It was developed to analyze the *MIT/GCM LLC4320* model. The script
["mit_gcm_detiding.py"](https://github.com/CNES/pangeo-pytide/blob/master/src/scripts/mit_gcm_detiding.py)
used to perform this analysis is distributed with this distribution.

## Try it for yourself

Try this library on [http://binder.pangeo.io/](pangeo.binder.io): <a href="https://binder.pangeo.io/v2/gh/CNES/pangeo-pytide/master?filepath=notebooks%2Fmitgcm_detiding.ipynb"><img style="float;margin:2px 2px -4px 2px" src="https://binder.pangeo.io/badge_logo.svg"></a>

## How To Install

[Anaconda](https://anaconda.org) is a free and open-source distribution of the
Python programming language for scientific computing, that aims to simplify
package management and deployment. Package versions are managed by the package
management system conda.

The first step is to install the anaconda distribution. The installation manual
for this software is detailed
[here](https://docs.anaconda.com/anaconda/install/).

To install the software using conda simply execute the following command:

    conda install pytide -c conda-forge

This command will install the software and the necessary dependencies. More
information is available on the syntax of this command on the [related
documentation](https://conda.io/projects/conda/en/latest/commands/install.html).

> If you want to build the package yourself, you can find more information on
> the [help page](https://pangeo-pytide.readthedocs.io/en/latest/setup.html) of
> the project.

### Notes

I had some problems installing due to some MKL issues even in conda, and therefore in this version I removed MKL altogether as it was not important for my use. In this case, the build script works fine on my machine (Ubuntu 20.04), and the result available in the ```build``` folder works fine (may need to add to bashrc though, for example in my case:

```bash
export PYTHONPATH="${PYTHONPATH}:/home/jrmet/Desktop/Git/pangeo-pytide/build/lib.linux-x86_64-3.8"
```

## Testing

```bash
python -m unittest
```

## Quick Tutorial

See the ```example``` folder.

The recommended way to use the code is to use the ```PyTideAnalyzer``` class, which is a thin wraper around older pytide methods with the aim to make the API clear and robust, and to avoid that the user shoots himself in the foot in particular when it regards time zone issues.

```python
import pytide
import os
import netCDF4
import datetime
import pytz
import time
from pytide.helpers import show_tides

os.environ["TZ"] = "UTC"
time.tzset()

path_to_data = os.path.join(os.getcwd(),
                            "tests", "dataset",
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
```
