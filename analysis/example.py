#%% Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
## Local
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlr import helpers
from dlr import linerating
from dlr import paths
from dlr import plots
from dlr import physics

## Plot formatting
plots.plotparams()
pd.options.display.max_columns = 200
pd.options.display.max_rows = 30

#%% Shared settings
## Line height in meters
height = 10
years = range(2007,2014)

#%% Get test lines
fiona.supported_drivers['KML'] = 'rw'
dflines = (
    gpd.read_file(os.path.join(paths.data,'test_lines.kml'), driver='KML')
    .rename(columns={'Name':'ID'})
    .to_crs('ESRI:102008')
    .assign(VOLTAGE=230)
)
dflines = helpers.lookup_diameter_resistance(dflines).set_index('ID')
## Get SLR for comparison
dflines['SLR'] = physics.ampacity(
    diameter_conductor=dflines.diameter,
    resistance_conductor=dflines.resistance,
)

# #%% Or get some HIFLD lines from a given state
# _dflines = helpers.get_hifld()
# dfstates =  helpers.get_reeds_zones()['st']
# state = 'CO'
# statebounds = dfstates.loc[[state]].statebounds.squeeze()
# dflines = _dflines.loc[
#     (_dflines.length_miles >= 5)
#     & (_dflines.rep_voltage == 230)
#     & (_dflines.statebounds.maxy <= statebounds.maxy)
#     & (_dflines.statebounds.maxx <= statebounds.maxx)
#     & (_dflines.statebounds.minx >= statebounds.minx)
#     & (_dflines.statebounds.miny >= statebounds.miny)
# ].sample(5, random_state=1)

# #%% Or get a specific HIFLD line
# dflines = helpers.get_hifld().loc[['202132']].copy()


#%% Get WTK and NSRDB cells
meta = helpers.get_grids()

## Output container
ratings = {}

#%%### Full dynamic line rating, including wind speed and direction
method = 'DLR'
ratings[method] = {}
for iline, line in dflines.iterrows():
    ### Get grid cells
    keep_cells = linerating.get_cells(line=line, meta=meta, buffer_km=10)
    cell_combinations = linerating.get_cell_overlaps(keep_cells=keep_cells)

    ### Get segment angles from North
    line_segments = linerating.get_segment_azimuths(
        line=line, cell_combinations=cell_combinations)

    ### Get weather data
    dfweather = linerating.get_weather_h5py(
        line=line,
        meta=meta,
        weatherlist=['temperature','windspeed','winddirection','pressure','ghi'],
        years=years,
        verbose=1,
    )

    ### Calculate ratings for all segments
    segment_ampacity = {}
    for segment, (i_wtk, i_nsrdb, azimuth) in line_segments.iterrows():
        segment_ampacity[segment] = physics.ampacity(
            windspeed=dfweather['windspeed'][i_wtk],
            wind_conductor_angle=(dfweather['winddirection'][i_wtk] - azimuth),
            temp_ambient_air=(dfweather['temperature'][i_wtk] + physics.C2K),
            pressure=dfweather['pressure'][i_wtk],
            solar_ghi=dfweather['ghi'][i_nsrdb],
            diameter_conductor=line.diameter,
            resistance_conductor=line.resistance,
        )
    segment_ampacity = pd.concat(segment_ampacity, axis=1)
    ### Line rating is the minimum across all segments
    ratings[method][line.name] = segment_ampacity.min(axis=1)


#%%### Clear-sky-irradiance adjusted ratings: No wind speed or direction
method = 'CLR'
defaults = {'windspeed': 0, 'wind_conductor_angle': 90}
ratings[method] = {}
for iline, line in dflines.iterrows():
    ### Get grid cells
    keep_cells = linerating.get_cells(line=line, meta=meta, buffer_km=10)
    cell_combinations = linerating.get_cell_overlaps(keep_cells=keep_cells)

    ### Get weather data
    dfweather = linerating.get_weather_h5py(
        line=line,
        meta=meta,
        weatherlist=['temperature','pressure','clearsky_ghi'],
        years=years,
        verbose=1,
    )

    ### Because we're not using wind direction we only need cell combinations, not segments
    cell_ampacity = {}
    for (i_wtk, i_nsrdb) in cell_combinations.index:
        cell_ampacity[(i_wtk, i_nsrdb)] = physics.ampacity(
            windspeed=defaults['windspeed'],
            wind_conductor_angle=defaults['wind_conductor_angle'],
            temp_ambient_air=(dfweather['temperature'][i_wtk] + physics.C2K),
            pressure=dfweather['pressure'][i_wtk],
            solar_ghi=dfweather['clearsky_ghi'][i_nsrdb],
            diameter_conductor=line.diameter,
            resistance_conductor=line.resistance,
        )
    cell_ampacity = pd.concat(cell_ampacity, axis=1)
    ratings[method][line.name] = cell_ampacity.min(axis=1)


#%%### DLR but with clear-sky irradiance, always-parallel wind (worst case), and
###### forecast margin adjustments for temperature and wind speed
method = 'FLR'
defaults = {'wind_conductor_angle': 0}
forecast_margin = {
    'windspeed': -2, # m/s
    'temp_ambient_air': +2, # °C or K
}
ratings[method] = {}
for iline, line in dflines.iterrows():
    ### Get grid cells
    keep_cells = linerating.get_cells(line=line, meta=meta, buffer_km=10)
    cell_combinations = linerating.get_cell_overlaps(keep_cells=keep_cells)

    ### Get weather data
    dfweather = linerating.get_weather_h5py(
        line=line,
        meta=meta,
        weatherlist=['temperature','windspeed','pressure','clearsky_ghi'],
        years=years,
        verbose=1,
    )

    ### Because we're not using wind direction we only need cell combinations, not segments
    cell_ampacity = {}
    for (i_wtk, i_nsrdb) in cell_combinations.index:
        cell_ampacity[(i_wtk, i_nsrdb)] = physics.ampacity(
            windspeed=dfweather['windspeed'][i_wtk],
            wind_conductor_angle=defaults['wind_conductor_angle'],
            temp_ambient_air=(dfweather['temperature'][i_wtk] + physics.C2K),
            pressure=dfweather['pressure'][i_wtk],
            solar_ghi=dfweather['clearsky_ghi'][i_nsrdb],
            diameter_conductor=line.diameter,
            resistance_conductor=line.resistance,
            forecast_margin=forecast_margin,
        )
    cell_ampacity = pd.concat(cell_ampacity, axis=1)
    ratings[method][line.name] = cell_ampacity.min(axis=1)


#%%### Take a look
### Distribution of ratings
methods = ['CLR','DLR','FLR']
colors = plots.rainbowmapper(dflines.index)
ncols = len(methods)
plt.close()
f, ax = plt.subplots(1, ncols, figsize=(2*ncols, 3.75), sharex=True, sharey=True)
for col, method in enumerate(methods):
    df = pd.concat(ratings[method], axis=1)
    for line in colors:
        ax[col].plot(
            np.linspace(0,100,len(df)),
            (df[line].sort_values(ascending=False) / dflines.loc[line,'SLR'] - 1) * 100,
            c=colors[line], label=line,
        )
    ax[col].axhline(0, c='k', ls='--', lw=0.75)
    ax[col].set_title(f"SLR → {method}")
    if col == 0:
        ax[col].legend(loc='upper right', frameon=False)
ax[0].set_ylabel('Rating change [%]')
ax[0].set_xlabel('Percent of hours above y-axis value [%]', x=0, ha='left')
plots.despine(ax)
plt.show()

#%% Maps
state = helpers.get_reeds_zones()['st'].loc[['CO']]
plt.close()
f, ax = plt.subplots(1, 2, figsize=(10, 5))
## Zoomed out
state.plot(ax=ax[0], facecolor='none', edgecolor='k', lw=0.5)
dflines.plot(ax=ax[0], color='C3', lw=1)
## Zoomed in, with weather grid cells
colors = {'wtk':'C0', 'nsrdb':'C1'}
alpha = 0.3
for iline, line in dflines.iterrows():
    keep_cells = linerating.get_cells(line=line, meta=meta, buffer_km=10)
    cell_combinations = linerating.get_cell_overlaps(keep_cells=keep_cells)
    for data, c in colors.items():
        keep_cells[data].plot(ax=ax[1], alpha=alpha, facecolor=c, edgecolor=c, lw=0.5)
    cell_combinations.plot(ax=ax[1], facecolor='none', edgecolor='k', lw=0.5)
dflines.plot(ax=ax[1], color='C3', lw=1.0)

handles = [
    mpl.patches.Patch(
        facecolor=colors[data], edgecolor=colors[data], alpha=alpha,
        label=f'{data.upper()} grid cells',
    ) for data in colors
] + [
    mpl.lines.Line2D([], [], color='k', label='Weather cell combinations'),
    mpl.lines.Line2D([], [], color='C3', label='Test lines'),
]
ax[1].legend(
    handles=handles, loc='lower left', frameon=False, bbox_to_anchor=(-0.1,-0.25),
    handlelength=0.75, handletextpad=0.5,
)

for col in range(2):
    ax[col].axis('off')
plt.show()
