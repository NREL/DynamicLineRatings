# Dynamic Line Ratings (DLR)

This package provides tools for estimating dynamic transmission line ratings (DLR) using weather data from the [WIND Toolkit (WTK)](https://www.nrel.gov/grid/wind-toolkit.html) and [National Solar Radiation Database (NSRDB)](https://nsrdb.nrel.gov/). Transmission line routes can be pulled from the [Transmission Lines Homeland Infrastructure Foundation-Level Dataset (HIFLD)](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines) or provided by the user.

Methodological details and important caveats are described at https://www.nrel.gov/docs/fy25osti/91599.pdf. Example outputs for ~84,000 HIFLD lines are available at https://data.openei.org/submissions/6231.

## Installation
1. Clone this repo: `git clone git@github.com:NREL/DynamicLineRatings.git`
2. Navigate to the repository directory, then set up the conda environment:
    1. `conda env create -f environment.yml`
    2. Each time you use code from this repo, run `conda activate dlr` first.
3. To access WTK and NSRDB data remotely, set up your `~/.hscfg` file following the directions at https://github.com/NREL/hsds-examples:
    1. Request an NREL API key from https://developer.nrel.gov/signup/
    2. Create a `~/.hscfg` file with the following information:
        ```
        hs_endpoint = https://developer.nrel.gov/api/hsds
        hs_username = None
        hs_password = None
        hs_api_key = your API key
        ```

## Usage
### CLI
```console
$ python -m dlr --help
usage: __main__.py [-h] [-s [START]] [-e [END]] [-y YEARS [YEARS ...]] [-w [WINDSPEED]] [-d [WIND_CONDUCTOR_ANGLE]] [-t [TEMPERATURE]]
                   [-p [PRESSURE]] [-i [IRRADIANCE]] [-c [CONDUCTOR_PARAMS ...]] [-m [FORECAST_MARGIN ...]]

options:
  -h, --help            show this help message and exit
  -s [START], --start [START]
  -e [END], --end [END]
  -y YEARS [YEARS ...], --years YEARS [YEARS ...]
                        Years to derive weather data for (options: 2007 through 2013)
  -w [WINDSPEED], --windspeed [WINDSPEED]
                        Static windspeed (in m/s) or data source for variable windspeed (options: ['wtk'])
  -d [WIND_CONDUCTOR_ANGLE], --wind_conductor_angle [WIND_CONDUCTOR_ANGLE]
                        Static angle between wind and conductor (in degrees) or data source for variable wind direction (options: ['wtk'])
  -t [TEMPERATURE], --temperature [TEMPERATURE]
                        Static ambient air temperature (in Celsius) or data source for variable air temp (options: ['wtk'])
  -p [PRESSURE], --pressure [PRESSURE]
                        Static air pressure (in Pa) or data source for variable pressure (options: ['wtk'])
  -i [IRRADIANCE], --irradiance [IRRADIANCE]
                        Static solar irradiance (in W/m**2) or '-'-delimited pair of data source for variable irradiance and irradiance type
                        (options: ['nsrdb-ghi', 'nsrdb-clearsky_ghi'])
  -c [CONDUCTOR_PARAMS ...], --conductor_params [CONDUCTOR_PARAMS ...]
                        Accepted parameters are 'temperature' (in C), 'emissivity', and 'absorptivity'
  -m [FORECAST_MARGIN ...], --forecast_margin [FORECAST_MARGIN ...]
                        Accepted parameters are 'windspeed' (m/s), 'wind_conductor_angle' (degrees), 'temperature' (C), 'pressure' (Pa), and
                        'irradiance' (W/m**2)
```

Example:
```console
python -m dlr -s 0 -e 10 --windspeed wtk --wind_conductor_angle 90 --irradiance nsrdb-ghi --conductor_params temperature=100 emissivity=0.9
```

### API
See [analysis/example_calc.ipynb](https://github.com/NREL/DynamicLineRatings/blob/main/analysis/example_calc.ipynb) for an example line rating calculation.

See [analysis/example_oedi.ipynb](https://github.com/NREL/DynamicLineRatings/blob/main/analysis/example_oedi.ipynb) for an example of how to interact with the precalculated datasets available at https://data.openei.org/submissions/6231.
