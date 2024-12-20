#%% Imports
import pandas as pd
import math
import os
import sys
from tqdm import tqdm
import geopandas as gpd
import shapely
import h5py
import h5pyd
## Local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import helpers
import paths
from dlr import physics

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
        ## Add a buffer around the line to avoid edge effects
        linebuffer = line.geometry.buffer(buffer_km * 1e3)
        meta = helpers.get_grids(bounds=linebuffer.bounds)

    ## Get Voronoi polygons for cells
    voronois = {}
    keep_cells = {}
    for data in ['nsrdb','wtk']:
        voronois[data] = helpers.voronoi_polygons(meta[data][['x','y','i']])
        voronois[data]['i'] = meta[data].iloc[
            helpers.closestpoint(
                voronois[data],
                meta[data],
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
    weather2data,
    meta=None,
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
    for i in weather2data.keys():
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

    ## Convenience dict
    weather2datum = {
        'temperature': f'temperature_{height}m',
        'windspeed': f'windspeed_{height}m',
        'winddirection': f'winddirection_{height}m',
        'pressure': f'pressure_{height_pressure}m',
        'clearsky_ghi': 'clearsky_ghi',
        'ghi': 'ghi',
    }
    weather2datum = {k:v for k,v in weather2datum.items() if k in weather2data.keys()}
    datum2weather = {v:k for k,v in weather2datum.items()}
    datums = {
        data: sorted(set([
            datum for weather,datum in weather2datum.items()
            if weather2data[weather] == data
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
        fpath = fpaths[data].format(year=year)
        
        hdf5_file = helpers.get_hdf5_file(fpath)
        with hdf5_file as f:
            time_index = pd.to_datetime(f['time_index'][...].astype(str))
            dictweather[weather,year] = pd.DataFrame(
                f[datum][:,indices],
                index=time_index,
                columns=indices,
            ) * scale[data]
            ## NSRDB has localized timestamps at 30-minute resolution, so
            # delocalize and downsample to 60-minute to match WTK
            if data == 'nsrdb':
                dictweather[weather,year] = (
                    dictweather[weather,year]
                    .set_index([idx.tz_localize(None) for idx in dictweather[weather,year].index])
                    .iloc[::2]
                )
            ## Pressure is in kPa but needs to be in Pa
            if datum.startswith('pressure'):
                dictweather[weather,year] *= 1e3

    dfweather = {
        weather: pd.concat([dictweather[weather,year] for year in years])
        for weather in weather2data.keys()
    }

    return dfweather

def calc_ratings(
    line: gpd.GeoSeries,
    years: int | list[int] = list(range(2007, 2014)),
    windspeed: float | str = 0.61,
    pressure: float | str = 101325,
    temp_ambient_air: float | str = 40 + physics.C2K,
    wind_conductor_angle: float | str = 90,
    solar_ghi: float | str = 1000,
    temp_conductor: float | str = 75 + physics.C2K,
    emissivity_conductor: float | str = 0.8,
    absorptivity_conductor: float | str = 0.8,
    forecast_margin: dict = {},
    check_units: bool = True,
):
    """Calculate hourly ratings for a given line as a function of weather and conductor parameters.

    Args:
        line: gpd.GeoSeries
        years: int or list representing year(s) of historic weather data
        windspeed (numeric, str): Static windspeed [m/s] or data source for variable windspeed 
            (e.g., 'wtk')
        pressure (numeric, str): Static air pressure [Pa] or data source for variable pressure 
            (e.g., 'wtk')
        temp_ambient_air (numeric, str): Static air temperature [K] or data source for variable temp
            (e.g., 'wtk')
        wind_conductor_angle (numeric): Static angle between wind direction and line segment [°] or
            data source for variable wind direction (e.g., 'wtk')
        solar_ghi (numeric): Static solar global horizontal irradiance [W m^-2] or '-'-delimited
            pair of variable irradiance data source and irradiance type (e.g., 'nsrdb-ghi')
        temp_conductor (float): Maximum allowable temperature of conductor [K].
            Default of 75°C + C2K = 348.15 K is a rule of thumb for ACSR conductors.
        diameter_conductor (float): Diameter of conductor [m]
        resistance_conductor (float): Resistance of conductor [Ω/m]
        absorptivity_conductor (float): Absorptivity of conductor. Defaults to 0.8.
        emissivity_conductor (float): Emissivity of conductor. Defaults to 0.8.
        forecast_margin (dict[str, numeric]): Additive adjustments to apply to each
            weather parameter
        check_units (bool): Check that provided temperature and pressure values are within
            reasonable ranges (assuming units of K and Pa respectively)

    Returns:
        current (numeric): Rated ampacity [A]
    """
    ### Get grid cells
    keep_cells = get_cells(line=line, buffer_km=10)
    cell_combinations = get_cell_overlaps(keep_cells=keep_cells)

    ### Get weather data
    weather2data = {
        'temperature': temp_ambient_air,
        'windspeed': windspeed,
        'winddirection': wind_conductor_angle,
        'pressure': pressure
    }
    weather2data = {k:v for k,v in weather2data.items() if isinstance(v, str)}
    if isinstance(solar_ghi, str):
        ghi_source, ghi_type = solar_ghi.split('-')
        weather2data[ghi_type] = ghi_source

    dfweather = get_weather_h5py(
        line=line,
        weather2data=weather2data,
        years=years,
        verbose=1,
    )
    windspeed_data = dfweather.get('windspeed', {})
    temp_ambient_air_data = dfweather.get('temperature', {})
    if isinstance(temp_ambient_air_data, pd.DataFrame):
        temp_ambient_air_data += physics.C2K
    pressure_data = dfweather.get('pressure', {})
    solar_ghi_data = dfweather.get(ghi_type, {})

    if 'winddirection' in dfweather:
        ### Get segment angles from North
        line_segments = get_segment_azimuths(
            line=line, cell_combinations=cell_combinations)

        ### Calculate ratings for all segments        
        segment_ampacity = {}
        for segment, (i_wtk, i_nsrdb, azimuth) in line_segments.iterrows():
            segment_ampacity[segment] = physics.ampacity(
                windspeed=windspeed_data.get(i_wtk, windspeed),
                wind_conductor_angle=(dfweather['winddirection'][i_wtk] - azimuth),
                temp_ambient_air=temp_ambient_air_data.get(i_wtk, temp_ambient_air),
                pressure=pressure_data.get(i_wtk, pressure),
                solar_ghi=solar_ghi_data.get(i_nsrdb, solar_ghi),
                temp_conductor=temp_conductor,
                diameter_conductor=line.diameter,
                resistance_conductor=line.resistance,
                emissivity_conductor=emissivity_conductor,
                absorptivity_conductor=absorptivity_conductor,
                forecast_margin=forecast_margin,
                check_units=check_units
            )
        min_ratings = pd.concat(segment_ampacity, axis=1).min(axis=1)
    else:
        ### Because we're not using wind direction we only need cell combinations, not segments
        cell_ampacity = {}
        for (i_wtk, i_nsrdb) in cell_combinations.index:
            cell_ampacity[(i_wtk, i_nsrdb)] = physics.ampacity(
                windspeed=windspeed_data.get(i_wtk, windspeed),
                wind_conductor_angle=wind_conductor_angle,
                temp_ambient_air=temp_ambient_air_data.get(i_wtk, temp_ambient_air),
                pressure=pressure_data.get(i_wtk, pressure),
                solar_ghi=solar_ghi_data.get(i_nsrdb, solar_ghi),
                temp_conductor=temp_conductor,
                diameter_conductor=line.diameter,
                resistance_conductor=line.resistance,
                emissivity_conductor=emissivity_conductor,
                absorptivity_conductor=absorptivity_conductor,
                forecast_margin=forecast_margin,
                check_units=check_units
            )
        min_ratings = pd.concat(cell_ampacity, axis=1).min(axis=1)

    return min_ratings