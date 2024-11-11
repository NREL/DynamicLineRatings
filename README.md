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

## Example usage
See [analysis/example.py](https://github.com/NREL/DynamicLineRatings/blob/main/analysis/example.py) for an example line rating calculation.

See [analysis/example_oedi.py](https://github.com/NREL/DynamicLineRatings/blob/main/analysis/example_oedi.py) for an example of how to interact with the precalculated datasets available at https://data.openei.org/submissions/6231.
