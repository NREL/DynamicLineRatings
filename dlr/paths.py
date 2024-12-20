## Imports
import datetime
import os

## This repository
repo = os.path.realpath(os.path.join(os.path.dirname(__file__),'..'))
data = os.path.join(repo, 'data')
outputs = os.path.join(repo, 'outputs')
io = os.path.join(data, 'io')
## Data files
### In the repo
regional_assumptions = os.path.join(data, 'regional_assumptions.csv')
### Downloaded from external sources
downloads = os.path.join(data, 'downloads')
#### https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines
lines = os.path.join(data, 'downloads', 'Electric__Power_Transmission_Lines')
### Generated by the user
meta_nsrdb = os.path.join(io, 'meta_nsrdb.gpkg')
meta_wtk = os.path.join(io, 'meta_wtk.gpkg')
### External sources
nsrdb = '/nrel/nsrdb/current/nsrdb_{year}.h5'
wtk = '/nrel/wtk/conus/wtk_conus_{year}.h5'
## Output files
ratings = os.path.join(outputs, 'ratings')
### Organize figures by today's date
figures = os.path.join(outputs, 'figures', datetime.datetime.now().strftime('%Y%m%d'))

## Make untracked folders
os.makedirs(io, exist_ok=True)
os.makedirs(downloads, exist_ok=True)
os.makedirs(ratings, exist_ok=True)
