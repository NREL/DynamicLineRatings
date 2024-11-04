# Dynamic Line Ratings (DLR)

## Installation
1. Clone this repo: `git clone git@github.com:NREL/DynamicLineRatings.git`
2. Navigate to the repository directory, then set up the conda environment:
    1. `conda env create -f environment.yml`
    2. Each time you use code from this repo, run `conda activate dlr` first.
3. To access WIND Toolkit (WTK) and National Solar Radiation Database (NSRDB) data remotely, set up your `~/.hscfg` file following the directions at https://github.com/NREL/hsds-examples:
    1. Request an NREL API key from https://developer.nrel.gov/signup/
    2. Create a `~/.hscfg` file with the following information:
        ```
        hs_endpoint = https://developer.nrel.gov/api/hsds
        hs_username = None
        hs_password = None
        hs_api_key = your API key
        ```

## Example usage
See analysis/example.py for a complete example.
