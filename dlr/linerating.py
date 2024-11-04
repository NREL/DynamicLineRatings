#%% Imports
import pandas as pd
import math
import os
import sys
from tqdm import tqdm
import geopandas as gpd
import shapely
import h5pyd
## Local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import helpers
import paths

os.environ['USE_PYGEOS'] = '0'

#%% Functions
def get_cells(line, meta=None, buffer_km=10):
    """Get Voronoi polygons for NSRDB and WTK pixels that overlap transmission line
    Args:
        line: gpd.GeoSeries
        meta: dict (['nsrdb','wtk'] keys)
        buffer_km: float (km)

    Returns:
        keep_cells (dict): dictionary (['nsrdb','wtk'] keys) of gpd.GeoDataFrame's
    """
    ## Get raster of weather points if necessary
    if meta is None:
        meta = helpers.get_grids()

    ## Add a buffer around the line to avoid edge effects
    linebuffer = line.geometry.buffer(buffer_km * 1e3)
    linebounds = dict(zip(['minx','miny','maxx','maxy'], linebuffer.bounds))

    ## Get Voronoi polygons for cells
    voronois = {}
    keep_cells = {}
    for data in ['nsrdb','wtk']:
        df = meta[data].loc[
            (linebounds['miny'] <= meta[data].y)
            & (meta[data].y <= linebounds['maxy'])
            & (linebounds['minx'] <= meta[data].x)
            & (meta[data].x <= linebounds['maxx'])
        ]
        voronois[data] = helpers.voronoi_polygons(df[['x','y','i']])

        voronois[data]['i'] = df.iloc[
            helpers.closestpoint(
                voronois[data],
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

        overlap_length = voronois[data].intersection(line.geometry).length
        keep_cells[data] = voronois[data].loc[overlap_length.astype(bool)].copy()
        keep_cells[data]['km'] = overlap_length / 1000
        keep_cells[data].index = keep_cells[data].i

    return keep_cells


def get_cell_overlaps(keep_cells):
    """
    Args:
        keep_cells (dict): dictionary (['nsrdb','wtk'] keys) of gpd.GeoDataFrame's

    Returns:
        cell_combinations (gpd.GeoSeries): intersection of NSRDB and WTK cells

    """
    ### Get combinations of NSRDB and WTK cells
    cell_pairs = set()
    for _, wtkrow in keep_cells['wtk'].iterrows():
        overlap = keep_cells['nsrdb'].intersection(wtkrow.geometry)
        nsrdb_cells = keep_cells['nsrdb'].loc[~overlap.is_empty].i.values
        cell_pairs.update([(wtkrow.i, nsrdb_i) for nsrdb_i in nsrdb_cells])

    cell_combinations = gpd.GeoSeries({
        (i_wtk, i_nsrdb): (
            keep_cells['wtk'].loc[i_wtk,'geometry']
            .intersection(keep_cells['nsrdb'].loc[i_nsrdb,'geometry'])
        )
        for (i_wtk, i_nsrdb) in cell_pairs
    }).rename_axis(['i_wtk','i_nsrdb'])

    return cell_combinations


def get_points_list(geom):
    if isinstance(geom, shapely.geometry.linestring.LineString):
        return [list(geom.coords)]
    elif isinstance(geom, shapely.geometry.multilinestring.MultiLineString):
        return [list(i.coords) for i in geom.geoms]
    else:
        raise Exception(f'Unsupported segment geometry: {type(geom)}')


def get_azimuths_list(points_lists):
    azimuths = []
    for sublist in points_lists:
        for i in range(len(sublist) - 1):
            start = sublist[i]
            end = sublist[i + 1]
            lon_diff = start[0] - end[0]
            lat_diff = start[1] - end[1]
            azimuth = math.degrees(math.atan2(lon_diff, lat_diff))
            azimuths.append(azimuth)
    return azimuths


def get_segment_azimuths(line, cell_combinations, full_output=False):
    """
    """
    ### Section line by cells
    cell_segments = (
        cell_combinations.intersection(line.geometry)
    )
    cell_segments = cell_segments.loc[~cell_segments.is_empty].copy()
    assert cell_segments.length.sum() - line.geometry.length <= 1000, (
        "WTK/NSRDB cells don't fully contain line: "
        "Line length = {} km but only {} km lies within cells".format(
            line.geometry.length / 1e3, cell_segments.length.sum() / 1e3
        )
    )
    ## Convert to equirectangular since we'll calculate angle with wind direction
    cell_segments_latlon = (
        cell_segments.set_crs('ESRI:102008').to_crs('EPSG:4326')
        .rename('geometry').to_frame()
    )

    ### Segments within cells
    cell_segments_latlon['points_lists'] = cell_segments_latlon.geometry.map(get_points_list)

    ### Segment angles from North
    cell_segments_latlon['azimuth'] = cell_segments_latlon.points_lists.map(get_azimuths_list)
    line_segments = cell_segments_latlon.azimuth.explode().reset_index()

    if full_output:
        return {
            'cell_segments': cell_segments,
            'cell_segments_latlon': cell_segments_latlon,
            'line_segments': line_segments,
        }
    else:
        return line_segments


def get_weather_h5py(
    line,
    meta=None,
    weatherlist=['temperature','windspeed','winddirection','pressure','ghi'],
    height=10,
    years=range(2007,2014),
    verbose=0,
    buffer_km=10,
):
    """
    Args:
        points: dict with ['nsrdb','wtk'] keys
        weatherlist: list containing elements from
            ['temperature','windspeed','winddirection','pressure','clearsky_ghi','ghi']
        height: meters
        years: int or list
    """
    ### Check inputs
    allowed_weatherlist = [
        'temperature',
        'windspeed',
        'winddirection',
        'pressure',
        'clearsky_ghi',
        'ghi',
    ]
    for i in weatherlist:
        assert i in allowed_weatherlist, (
            f"Provided {i} in weatherlist but only the following are allowed:\n"
            + '\n> '.join(allowed_weatherlist)
        )
    ### Sites to query
    keep_cells = get_cells(line=line, meta=meta, buffer_km=buffer_km)

    ### Derived inputs
    ## Pressure is available at [0m, 100m, 200m] so round to nearest 100
    height_pressure = round(height, -2)
    if isinstance(years, (int, float)):
        years = [int(years)]

    ## Convenience dicts
    weather2data = {
        'temperature': 'wtk',
        'windspeed': 'wtk',
        'winddirection': 'wtk',
        'pressure': 'wtk',
        'clearsky_ghi': 'nsrdb',
        'ghi': 'nsrdb',
    }
    weather2datum = {
        'temperature': f'temperature_{height}m',
        'windspeed': f'windspeed_{height}m',
        'winddirection': f'winddirection_{height}m',
        'pressure': f'pressure_{height_pressure}m',
        'clearsky_ghi': 'clearsky_ghi',
        'ghi': 'ghi',
    }
    datum2weather = {v:k for k,v in weather2datum.items()}
    datums = {
        data: sorted(set([
            datum for weather,datum in weather2datum.items()
            if ((weather in weatherlist) and (weather2data[weather] == data))
        ]))
        for data in ['wtk', 'nsrdb']
    }
    datum2data = {e: k for k,v in datums.items() for e in v}
    fpaths = {'nsrdb':paths.nsrdb, 'wtk':paths.wtk}
    scale = {'wtk':0.01, 'nsrdb':1}
    queries = [(v,k,y) for k,v in datum2data.items() for y in years]

    ## Loop through queries and download data
    dictweather = {}
    iterator = tqdm(queries, desc=str(line.name)) if verbose else queries
    for (data, datum, year) in iterator:
        weather = datum2weather[datum]
        indices = keep_cells[data].i.sort_values().values
        with h5pyd.File(fpaths[data].format(year=year), 'r') as f:
            time_index = pd.to_datetime(f['time_index'][...].astype(str))
            dictweather[weather,year] = pd.DataFrame(
                f[datum][:,indices],
                index=time_index,
                columns=indices,
            ) * scale[data]
            ## NSRDB has 30-minute resolution, so downsample to 60-minute to match WTK
            if data == 'nsrdb':
                dictweather[weather,year] = dictweather[weather,year].iloc[::2]
            ## Pressure is in kPa but needs to be in Pa
            if datum.startswith('pressure'):
                dictweather[weather,year] *= 1e3

    dfweather = {
        weather: pd.concat([dictweather[weather,year] for year in years])
        for weather in weatherlist
    }

    return dfweather
