#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cmocean
from tqdm import tqdm
import traceback
import geopandas as gpd
import shapely
import rasterio
import rasterio.features
import rasterio.warp
## Local
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dlr import helpers
from dlr import paths
import plots
plots.plotparams()

#%% Download data
url = 'https://www.nrel.gov/gis/assets/images/us-wind-data.zip'
datapath = os.path.join(paths.downloads)
os.makedirs(datapath, exist_ok=True)
helpers.download(url, datapath)

#%% Read data
altitude = '10m'
fpath = os.path.join(datapath, 'us-wind-data', f'wtk_conus_{altitude}_mean_masked.tif')
dataset = rasterio.open(fpath, 'r')

#%%
shape = dataset.shape
data = dataset.read(1)
mask = dataset.dataset_mask()
crs = dataset.crs
print(crs)

#%% Take a look
plt.imshow(data, cmap=cmocean.cm.rain)

#%%
xy = [
    dataset.xy(row, col)
    for row in range(shape[0])
    for col in range(shape[1])
    if mask[row,col]
]

#%%
vals = [
    data[row,col]
    for row in range(shape[0])
    for col in range(shape[1])
    if mask[row,col]
]

#%%
df = gpd.GeoSeries(
    pd.Series(xy).map(shapely.geometry.Point).rename('geometry'),
    crs=crs.to_string(),
).to_crs('ESRI:102008')

#%%
dfout = gpd.GeoDataFrame(df)
dfout[f'windspeed_{altitude}'] = vals

#%%
plt.close()
f,ax = plt.subplots()
dfout.sample(100000).plot(
    ax=ax, column=f'windspeed_{altitude}', cmap=cmocean.cm.rain,
    lw=0, marker='s', markersize=1,
)
plt.show()

#%% Filter out offshore
dfcountry = helpers.get_reeds_zones()['country']
country = dfcountry.squeeze().geometry

in_country = dfout.intersection(country)
dfwrite = dfout.loc[~in_country.is_empty].copy()

#%%
plt.close()
f,ax = plt.subplots()
dfcountry.plot(ax=ax, facecolor='none', edgecolor='k', zorder=1e8)
dfwrite.sample(100000).plot(
    ax=ax, column=f'windspeed_{altitude}', cmap=cmocean.cm.rain,
    lw=0, marker='s', markersize=1,
)
plt.show()

print(dfout.shape)
print(dfwrite.shape)


#%% Write it
dfwrite.astype({f'windspeed_{altitude}':np.float32}).to_file(
    os.path.join(paths.io, f'average_windspeed_{altitude}.gpkg')
)


#%%### Plot it ######
figpath = paths.figures
print(figpath)
os.makedirs(figpath, exist_ok=True)
tz = 'Etc/GMT+6'


#%% ReEDS zones
dfmap = helpers.get_reeds_zones()

#%% HIFLD
dfhifld = helpers.get_hifld(
    calc_slr=True, calc_zlr=True,
    regional_air_temp={'summer':40, 'winter':20},
)

#%% Average windspeed
dfwindspeed = gpd.read_file(
    os.path.join(paths.io, f'average_windspeed_{altitude}.gpkg')
).rename(columns={'windspeed_10m':'windspeed'})
#%%
dfwindspeed['x'] = dfwindspeed.centroid.x
dfwindspeed['y'] = dfwindspeed.centroid.y
dfwindspeed['i'] = dfwindspeed.index

#%%### Get windspeed stats along each line
### Settings
buffer_km = 10
thresholds = np.arange(0.5,1.00001,0.05)
percents = [f'{x*100:.0f}%' for x in thresholds]
index = np.around(thresholds, 2)


dictout = {}
for ID in tqdm(dfhifld.index):
    buffer = dfhifld.loc[[str(ID)]].buffer(buffer_km*1000)
    line = dfhifld.loc[str(ID), 'geometry']
    bounds = buffer.bounds.squeeze(0)

    df = dfwindspeed.loc[
        (bounds['miny'] <= dfwindspeed.y)
        & (dfwindspeed.y <= bounds['maxy'])
        & (bounds['minx'] <= dfwindspeed.x)
        & (dfwindspeed.x <= bounds['maxx'])
    ]
    if not len(df):
        print(f'No overlap for {ID}')
        dictout[ID] = pd.Series(index=percents, name='windspeed')
        continue
    try:
        voronois = helpers.voronoi_polygons(df[['x','y','windspeed','i']])
    except Exception:
        print(f'Voronois filed for {ID}')
        print(traceback.format_exc())
        dictout[ID] = pd.Series(index=percents, name='windspeed')
        continue

    voronois['i'] = df.iloc[
        helpers.closestpoint(
            voronois,
            df,
            dfquerylabel=None, 
            dfqueryx='centroid_x',
            dfqueryy='centroid_y', 
            dfdatax='x',
            dfdatay='y',
            method='cartesian',
            return_distance=False,
            verbose=False,
        )
    ]['i'].values

    voronois['overlap_km'] = voronois.intersection(line).length / 1000
    overlap = voronois.loc[voronois.overlap_km > 0].copy()
    overlap['windspeed'] = overlap.i.map(dfwindspeed.windspeed)
    line_km = line.length / 1000
    sum_overlap = overlap['overlap_km'].sum()
    assert sum_overlap - line_km < 1

    # #%% Take a look
    # plt.close()
    # f,ax = plt.subplots()
    # dfmap['country'].plot(ax=ax, facecolor='none', edgecolor='k', lw=0.2)
    # buffer.buffer(100000).plot(ax=ax, facecolor='C3')
    # dfhifld.loc[[str(ID)]].plot(ax=ax, color='k')
    # buffer.plot(ax=ax, facecolor='C1', alpha=0.5)
    # voronois.plot(ax=ax, edgecolor='C0', facecolor='none', lw=0.1)
    # overlap.plot(ax=ax, edgecolor='none', facecolor='C0', alpha=0.3)
    # plt.show()
    # #%%

    ### Get the cumulative windspeed distribution
    dfout = overlap.sort_values('windspeed', ascending=False)[['overlap_km','windspeed']]
    dfout['length_fraction'] = dfout.overlap_km / sum_overlap
    dfout['length_cumfrac'] = dfout['length_fraction'].cumsum()


    ### Interpolate some thresholds
    dfstats = (
        pd.concat([
            dfout.set_index('length_cumfrac').windspeed,
            pd.Series(index=thresholds)
        ])
        .sort_index().interpolate('linear')
        # .round(2).drop_duplicates()
        .rename('windspeed').rename_axis('fraction').reset_index()
    )
    dfstats['pct'] = dfstats.fraction.map(lambda x: f'{x*100:.0f}%')
    dfwrite = (
        dfstats.loc[dfstats.pct.isin(percents)]
        .drop_duplicates('pct')
        .set_index('pct').windspeed
        .bfill()
    )
    dictout[ID] = dfwrite


    # #%% Take a look
    # plt.close()
    # f, ax = plt.subplots()
    # # ax.plot(dfout.length_fraction.cumsum(), dfout.windspeed)
    # # ax.set_xlabel('Fraction of line')
    # ax.plot(dfout.overlap_km.cumsum(), dfout.windspeed)
    # ax.set_xlabel('Kilometers of line')
    # ax.set_ylabel('Windspeed')
    # ax.set_ylim(0)
    # plt.show()

#%%
dfresults = pd.concat(dictout, axis=1).T
numlines = dfresults.shape[0]
savename = f'length_percentiles-windspeed_10m-{numlines}lines.h5'
dfresults.to_hdf(
    os.path.join(figpath, savename),
    key='data', complevel=4, format='table',
)

#%% Take a look
keep = ['50%','75%','100%']
dfresults[keep].stack().describe(percentiles=[
    0.05,0.02,0.01,0.005,0.001,
    0.95,0.98,0.99,0.995,0.999
])
vmin, vmax = 1, 5
lw = 0.75
ncols = len(keep)
nrows = 1
cmap = cmocean.cm.rain
cmap = cmocean.cm.curl
cmap = plt.cm.turbo
savename = f"windspeed_10m-{','.join(keep)}-{numlines}lines.png"

plt.close()
f,ax = plt.subplots(
    nrows, ncols, figsize=(3.5*ncols, 3.5*0.8*nrows), sharex=True, sharey=True,
    gridspec_kw={'wspace':-0.05},
)
for col, pct in enumerate(keep):
    dfplot = dfhifld.copy()
    dfplot['windspeed'] = dfresults[pct]
    print(dfplot.windspeed.describe())

    title = f"Mean 10m windspeed [m/s];\n{pct} of line\nabove color value"

    ax[col].axis('off')
    dfmap['country'].plot(ax=ax[col], facecolor='none', edgecolor='k', lw=0.75, zorder=1e6)
    dfmap['st'].plot(ax=ax[col], facecolor='none', edgecolor='k', lw=0.25, zorder=1e6)
    dfplot.plot(
        ax=ax[col], column='windspeed', cmap=cmap, lw=lw,
        vmin=vmin, vmax=vmax,
    )

    plots.addcolorbarhist(
        f=f, ax0=ax[col], data=dfplot['windspeed'],
        cmap=cmap, vmin=vmin, vmax=vmax, nbins=101,
        title=title,
        title_fontsize='large',
        ticklabel_fontsize='large',
        orientation='horizontal', cbarbottom=-0.1, histratio=1.5,
        cbarheight=0.8, cbarwidth=0.05, labelpad=2.5,
    )

plt.savefig(os.path.join(figpath, savename))
plt.show()
