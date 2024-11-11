#%% Imports
import pandas as pd
import numpy as np
import io
import os
import sys
import requests
import zipfile
import h5py
from tqdm import tqdm
import geopandas as gpd
import h5pyd
import pyproj
## Local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import physics
import paths
import plots

pyproj.network.set_network_enabled(False)
os.environ['USE_PYGEOS'] = '0'

#%% Constants
FOOT_PER_METER = 3.28084
DEFAULTS = {
    'wind_speed': 0.61, # m/s
    'wind_direction': 90, # degrees from parallel to conductor
    'air_temperature': 40 + physics.C2K, # K
    'conductor_temperature': 75 + physics.C2K, # K
    'ghi': 1000, # W/m^2
    'emissivity': 0.8,
    'absorptivity': 0.8,
}

volt_class_rep_voltage_map = {
    'UNDER 100': 69,
    '100-161': 115,
    '220-287': 230,
    '345': 345,
    '500': 500,
    '735 AND ABOVE': 765,
    'NOT AVAILABLE': np.nan,
    'DC': np.nan
}


#%% Functions
### File handling
def download(url, path, unzip=True):
    r = requests.get(url)
    if url.endswith('.zip') and unzip:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=path)
    else:
        with open(path, 'wb') as f:
            f.write(r.content)


### Geospatial
def closestpoint(
        dfquery,
        dfdata,
        dfquerylabel=None, 
        dfqueryx='longitude',
        dfqueryy='latitude', 
        dfdatax='longitude',
        dfdatay='latitude',
        method='cartesian',
        return_distance=False,
        verbose=True,
    ):
    """
    For each row of dfquery, returns closest label from dfdata
    """
    closestindexes = []
    closest_distances = []
    if method in ['cartesian', 'xy', None, 'equirectangular', 'latlon']:
        lons, lats = dfdata[dfdatax].values, dfdata[dfdatay].values
        if verbose:
            iterator = tqdm(dfquery.iterrows(), total=len(dfquery))
        else:
            iterator = dfquery.iterrows()
        for i, row in iterator:
            lon, lat = row[dfqueryx], row[dfqueryy]
            sqdistances = (lons - lon)**2 + (lats - lat)**2
            closestindex = np.argmin(sqdistances)
            closestindexes.append(closestindex)
            
            if return_distance is True:
                import geopy.distance
                closest_distances.append(
                    geopy.distance.distance(
                        (lat, lon), 
                        (dfdata[dfdatay].values[closestindex], 
                         dfdata[dfdatax].values[closestindex])).km
                )
            
    elif method in ['geopy', 'geodesic']:
        import geopy.distance
        lons, lats = dfdata[dfdatax].values, dfdata[dfdatay].values
        for i in tqdm(dfquery.index):
            lon, lat = dfquery.loc[i, dfqueryx], dfquery.loc[i, dfqueryy]
            distances = dfdata.apply(
                lambda row: geopy.distance.distance((lat, lon), (row[dfdatay], row[dfdatax])).km,
                axis=1).values
            closestindex = np.argmin(distances)
            closestindexes.append(closestindex)
            
            if return_distance is True:
                closest_distances.append(
                    geopy.distance.distance(
                        (lat, lon), 
                        (dfdata[dfdatay].values[closestindex], 
                         dfdata[dfdatax].values[closestindex])).km
                )
            
    if return_distance is True:
        return closestindexes, closest_distances
    else:
        return closestindexes


def voronoi_polygons(dfpoints, calc_areas=False):
    """
    Inputs
    ------
    dfpoints: pd.DataFrame with latitude and longitude columns
    
    Ouputs
    ------
    dfpoly: dataframe with Voronoi polygons and descriptive parameters
    
    Sources
    -------
    ('https://stackoverflow.com/questions/27548363/'
     'from-voronoi-tessellation-to-shapely-polygons)
    
    """
    import shapely
    import scipy
    import scipy.spatial
    import pyproj
    import geopandas as gpd

    ### Get latitude and longitude column names
    latlabel, lonlabel = plots.get_latlonlabels(dfpoints)
    if (latlabel is None) and (lonlabel is None):
        lonlabel, latlabel = 'x', 'y'
    
    ### Get polygons
    points = dfpoints[[lonlabel,latlabel]].values
    vor = scipy.spatial.Voronoi(points)

    ### Make shapely linestrings 
    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]

    ### Make shapely polygons, coords, and bounds
    ### Intersect each poly with region bounds
    regionhull = (
        shapely.geometry.MultiPoint(dfpoints[[lonlabel,latlabel]].values)
        .convex_hull)
    polys = [poly.intersection(regionhull) for poly in shapely.ops.polygonize(lines)]
    coords = [list(poly.exterior.coords) for poly in polys]
    bounds = [list(poly.bounds) for poly in polys] ### (minx, miny, maxx, maxy)
    centroid_x = [poly.centroid.x for poly in polys]
    centroid_y = [poly.centroid.y for poly in polys]
    centroids = [[poly.centroid.x, poly.centroid.y] for poly in polys]

    ### Make and return output dataframe
    dfpoly = gpd.GeoDataFrame(pd.DataFrame(
        {'coords':coords, 'bounds':bounds,
         'centroid':centroids,'centroid_x':centroid_x,'centroid_y':centroid_y,
         'geometry':polys,}))
    
    ### Calculate areas in square kilometers
    if calc_areas:
        areas = []
        for i, (poly, coord, bound) in enumerate(list(zip(polys, coords, bounds))):
            pa = pyproj.Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
                bound[1], bound[3], (bound[1]+bound[3])/2, (bound[0]+bound[2])/2))
            lon,lat = zip(*coord)
            x,y = pa(lon,lat)
            cop = {'type':'Polygon','coordinates':[zip(x,y)]}
            areas.append(shapely.geometry.shape(cop).area/1000000)
        dfpoly['area'] = areas

    return dfpoly


def get_reeds_zones(
    path_map='https://github.com/NREL/ReEDS-2.0/raw/refs/heads/main/inputs/shapefiles/US_PCA/US_PCA.shp',
    path_hierarchy='https://raw.githubusercontent.com/NREL/ReEDS-2.0/refs/heads/main/inputs/hierarchy.csv',
):
    """Get geodataframe of ReEDS zones (https://github.com/NREL/ReEDS-2.0).
    If you have a local copy of the ReEDS repository, you can use:
        path_map = {ReEDS repo path}/inputs/shapefiles/US_PCA
        path_hierarchy = {ReEDS repo path}/inputs/hierarchy.csv

    Returns a dictionary of maps of the hierarchy levels used in ReEDS
    """
    ### Get hierarchy of zones
    hierarchy = (
        pd.read_csv(path_hierarchy)
        .rename(columns={'*r':'r','ba':'r'})
        .set_index('r')
        .drop(columns=['st_interconnect','offshore'], errors='ignore')
    )
    hierarchy = hierarchy.loc[hierarchy.country.str.lower() == 'usa'].copy()

    ### Model zones
    dfba = gpd.read_file(path_map).set_index('rb')
    dfba['centroid_x'] = dfba.geometry.centroid.x
    dfba['centroid_y'] = dfba.geometry.centroid.y
    ## Include all hierarchy levels
    for col in hierarchy:
        dfba[col] = dfba.index.map(hierarchy[col])

    ### Aggregate zones to hierarchy levels
    dfmap = {'r': dfba.dropna(subset='country').copy()}
    dfmap['r']['centroid_x'] = dfmap['r'].centroid.x
    dfmap['r']['centroid_y'] = dfmap['r'].centroid.y
    for col in hierarchy:
        dfmap[col] = dfba.copy()
        dfmap[col]['geometry'] = dfmap[col].buffer(0.)
        dfmap[col] = dfmap[col].dissolve(col)
        for prefix in ['','centroid_']:
            dfmap[col][prefix+'x'] = dfmap[col].centroid.x
            dfmap[col][prefix+'y'] = dfmap[col].centroid.y

    return dfmap


def get_grids(verbose=False, buffer=40, offshore=True):
    """
    Inputs
    - buffer (float): km buffer to include around the contiguous U.S.
    - offshore (bool): whether to keep offshore wind sites

    Background:
    - https://github.com/NREL/hsds-examples/blob/master/notebooks/01_WTK_introduction.ipynb
    - https://github.com/NREL/hsds-examples/blob/master/notebooks/03_NSRDB_introduction.ipynb

    Returns:
    - dict of pd.DataFrame's with ['nsrdb','wtk'] keys
    """
    meta = {}
    ### Download and generate if they don't exist
    if (
        (not os.path.exists(paths.meta_nsrdb))
        or (not os.path.exists(paths.meta_wtk))
    ):
        print('Downloading and caching NSRDB/WTK points (subsequent calls will be faster)')
        nsrdb_fpath = "/nrel/nsrdb/v3/nsrdb_2012.h5"
        with h5pyd.File(nsrdb_fpath, 'r') as f:
            if verbose:
                nsrdb_list = list(f)
                print(nsrdb_list)
            meta['nsrdb'] = pd.DataFrame(f['meta'][...])

        wtk_fpath = "/nrel/wtk/conus/wtk_conus_2012.h5"
        with h5pyd.File(wtk_fpath, 'r') as f:
            if verbose:
                wtk_list = list(f)
                print(wtk_list)
            meta['wtk'] = pd.DataFrame(f['meta'][...])
            if not offshore:
                meta['wtk'] = meta['wtk'].loc[meta['wtk'].offshore == 0].copy()

        for data in meta:
            meta[data] = plots.df2gdf(meta[data])
            meta[data]['x'] = meta[data].centroid.x
            meta[data]['y'] = meta[data].centroid.y
            meta[data]['i'] = meta[data].index

            ## Downselect
            if buffer not in [None, np.inf]:
                country = get_reeds_zones()['country'].geometry.squeeze().buffer(buffer*1e3)
                if verbose:
                    print(f'Before downselection: {len(meta[data])}')
                meta[data] = meta[data].loc[~meta[data].intersection(country).is_empty].copy()
                if verbose:
                    print(f'After downselection: {len(meta[data])}')

            ## Write it
            meta[data][['i','latitude','longitude','x','y','geometry']].to_file(
                os.path.join(paths.io, f'meta_{data}.gpkg')
            )

    ### Read them
    for data in ['nsrdb', 'wtk']:
        meta[data] = gpd.read_file(os.path.join(paths.io, f'meta_{data}.gpkg'))

    return meta


### Ratings
def round_voltage(voltage):
    if np.isnan(voltage):
        return np.nan
    else:
        return min(
            volt_class_rep_voltage_map.values(),
            key=lambda x: abs(x - voltage)
        )


def read_lines(
    fpath: str = paths.hifld,
    line_idx_range: slice | None = None,
    crs: str = 'ESRI:102008',
):
    if line_idx_range is None:
        dflines = gpd.read_file(fpath)
    else:
        dflines = gpd.read_file(fpath, rows=line_idx_range)

    if str(dflines.crs) != crs:
        dflines = dflines.to_crs(crs)

    return dflines


def lookup_diameter_resistance(
    dflines,
    conductor_temp_kelvin: float = 75 + physics.C2K,
):
    assert (
        all(col in dflines.columns for col in ["ID", "geometry"])
    ), "The provided transmission line dataset must contain 'ID' and 'geometry' columns."

    assert (
        ("VOLTAGE" in dflines.columns)
        or (all(col in dflines.columns for col in ['diameter', 'resistance']))
    ), (
        "The provided transmission line dataset must contain either a 'VOLTAGE' column "
        "or 'diameter' and 'resistance' columns."
    )

    _dflines = dflines.copy()

    if not all(col in _dflines.columns for col in ['diameter', 'resistance']):
        # Read in voltage-to-diameter/resistance mapping and recalculate
        # resistance values if a new conductor temperature is provided
        kv_to_conductor_props = pd.read_csv(
            os.path.join(paths.data, "kv_to_conductor_props.csv"),
            index_col="voltage",
        )
        if conductor_temp_kelvin == 75 + physics.C2K:
            kv_to_conductor_props = (
                kv_to_conductor_props.rename(columns={"AC_R_75C": "resistance"})
                [["diameter", "resistance"]]
            )
        else:
            conductor_temp_celsius = conductor_temp_kelvin - physics.C2K
            R_25C = kv_to_conductor_props["AC_R_25C"]
            R_75C = kv_to_conductor_props["AC_R_75C"]
            kv_to_conductor_props["resistance"] = (
                R_25C + ((R_75C - R_25C) / (75-25)) * (conductor_temp_celsius-25)
            )
            kv_to_conductor_props = (
                kv_to_conductor_props[["diameter", "resistance"]]
            )

        # Replace any negative voltages with an inferred value based on the volt class,
        # or NaN if volt class doesn't specify a kV range
        if "VOLT_CLASS" in _dflines.columns:
            _dflines.loc[_dflines.VOLTAGE < 0, "VOLTAGE"] = (
                _dflines.loc[_dflines.VOLTAGE < 0, "VOLT_CLASS"].map(
                    volt_class_rep_voltage_map)
            )

        # Add a column representing the representative voltage
        # closest to the line's real voltage
        _dflines["rep_voltage"] = _dflines["VOLTAGE"].map(round_voltage)
        _dflines["diameter"] = _dflines["rep_voltage"].map(kv_to_conductor_props["diameter"])
        _dflines["resistance"] = _dflines["rep_voltage"].map(kv_to_conductor_props["resistance"])

    return _dflines


def assign_line_to_region(dflines, dfregions, label='region'):
    """
    """
    ## If a line overlaps with at least one region, assign it to the most overlapping region
    regions = dfregions.index
    _overlaps = {}
    for region in regions:
        _overlaps[region] = dflines.intersection(dfregions[region]).length
    overlaps = pd.concat(_overlaps, axis=1, names=label)
    main_region = (
        overlaps.stack().rename('overlap')
        .sort_values().reset_index().drop_duplicates('ID', keep='last')
        .set_index('ID')
    )
    main_region.loc[main_region.overlap == 0, label] = '_none'
    _dflines = dflines.merge(main_region[[label]], left_index=True, right_index=True)
    ## Also record lines that cross between regions
    _dflines[f'multi_{label}'] = overlaps.replace(0,np.nan).apply(
        lambda row: ','.join(row.dropna().index.tolist()),
        axis=1,
    )
    ## For unmapped lines, map them to the closest region
    ids_unmapped = _dflines.loc[_dflines[label] == '_none'].index
    for ID in ids_unmapped:
        _dflines.loc[ID,label] = (
            dfregions.distance(_dflines.loc[ID, 'geometry']).nsmallest(1).index[0])

    return _dflines


def calculate_zlr(
    dfhifld=None,
    regional_air_temp=True,
    regional_wind=False,
    regional_irradiance=False,
    regional_conductor_temp=False,
    regional_emissivity=False,
    regional_absorptivity=False,
    aggfunc='representative',
    minimal=False,
):
    """Calculate seasonal line ratings using regional_assumptions.csv
    Inputs
    ------
    regional_{}:
        If True, use value from regional_assumptions.csv in all seasons and regions
        If False, use default value from helpers.DEFAULTS in all seasons and regions
        If numeric, use the provided numeric value in all seasons and regions
        If dictionary with ('summer', 'winter') as keys, use those in all regions
    regional_air_temp: °Celsius
    regional_conductor_temp: °Celsius
    regional_wind: m/s

    Outputs
    -------
    pd.DataFrame: Copy of dfhifld with ZLR_summer and ZLR_winter added
    """
    ## Spatial resolution: ReEDS transmission regions (transreg)
    regional_assumptions = pd.read_csv(paths.regional_assumptions)
    level = 'transreg'
    ## Fill missing winter values with summer values
    fill_columns = [
        'ambient_temp_{}_celsius',
        'windspeed_{}_fps',
        'solar_radiation_{}_watts_per_square_meter',
    ]
    for col in fill_columns:
        regional_assumptions.loc[
            regional_assumptions[col.format('winter')].isnull(),
            col.format('winter')
        ] = regional_assumptions.loc[
            regional_assumptions[col.format('winter')].isnull(),
            col.format('summer')
        ]
    ## Convert feet per second to meters per second
    for col in ['windspeed_summer_fps','windspeed_winter_fps']:
        regional_assumptions[col.replace('fps','mps')] = regional_assumptions[col] / FOOT_PER_METER
    ## Aggregate
    if aggfunc == 'representative':
        regional_assumptions = regional_assumptions.loc[
            regional_assumptions.representative == 1
        ].set_index(level)
    else:
        regional_assumptions = regional_assumptions.groupby(level).agg(aggfunc)
    assert (regional_assumptions.index.value_counts() == 1).all()

    ### Assign lines to regions
    dfregions = get_reeds_zones()[level].geometry
    _dfhifld = assign_line_to_region(dflines=dfhifld, dfregions=dfregions, label=level)

    ### Get the appropriate regional data for each season
    for season in ['summer', 'winter']:
        if regional_air_temp is True:
            _dfhifld[f'temperature_{season}'] = _dfhifld[level].map(
                regional_assumptions[f'ambient_temp_{season}_celsius'] + physics.C2K)
        elif regional_air_temp is False:
            _dfhifld[f'temperature_{season}'] = DEFAULTS['air_temperature']
        elif isinstance(regional_air_temp, dict):
            _dfhifld[f'temperature_{season}'] = regional_air_temp[season] + physics.C2K
        else:
            _dfhifld[f'temperature_{season}'] = regional_air_temp + physics.C2K

        if regional_wind is True:
            _dfhifld[f'windspeed_{season}'] = _dfhifld[level].map(
                regional_assumptions[f'windspeed_{season}_mps'])
            _dfhifld['windangle'] = _dfhifld[level].map(
                regional_assumptions['windangle_deg'])
        elif regional_wind is False:
            _dfhifld[f'windspeed_{season}'] = DEFAULTS['wind_speed']
            _dfhifld['windangle'] = DEFAULTS['wind_direction']
        else:
            _dfhifld[f'windspeed_{season}'] = regional_wind
            _dfhifld['windangle'] = DEFAULTS['wind_direction']

        if regional_irradiance is True:
            _dfhifld[f'ghi_{season}'] = _dfhifld[level].map(
                regional_assumptions[f'solar_radiation_{season}_watts_per_square_meter'])
        elif regional_irradiance is False:
            _dfhifld[f'ghi_{season}'] = DEFAULTS['ghi']
        else:
            _dfhifld[f'ghi_{season}'] = regional_irradiance

        if regional_conductor_temp is True:
            _dfhifld['conductor_temp'] = _dfhifld[level].map(
                regional_assumptions['conductor_acsr_temp_celsius'] + physics.C2K)
        elif regional_conductor_temp is False:
            _dfhifld['conductor_temp'] = DEFAULTS['conductor_temperature']
        else:
            _dfhifld['conductor_temp'] = regional_conductor_temp

        if regional_emissivity is True:
            _dfhifld['emissivity'] = _dfhifld[level].map(
                regional_assumptions['emissivity'])
        elif regional_emissivity is False:
            _dfhifld['emissivity'] = DEFAULTS['emissivity']
        else:
            _dfhifld['emissivity'] = regional_emissivity

        if regional_absorptivity is True:
            _dfhifld['absorptivity'] = _dfhifld[level].map(
                regional_assumptions['absorptivity'])
        elif regional_absorptivity is False:
            _dfhifld['absorptivity'] = DEFAULTS['absorptivity']
        else:
            _dfhifld['absorptivity'] = regional_absorptivity

        ### Calculate the rating
        _dfhifld[f'ZLR_{season}'] = physics.ampacity(
            windspeed=_dfhifld[f'windspeed_{season}'],
            wind_conductor_angle=_dfhifld['windangle'],
            temp_ambient_air=_dfhifld[f'temperature_{season}'],
            solar_ghi=_dfhifld[f'ghi_{season}'],
            temp_conductor=_dfhifld['conductor_temp'],
            diameter_conductor=_dfhifld.diameter,
            resistance_conductor=_dfhifld.resistance,
            emissivity_conductor=_dfhifld.emissivity,
            absorptivity_conductor=_dfhifld.absorptivity,
        )

    if minimal:
        outcols = [level, 'ZLR_summer', 'ZLR_winter']
    else:
        outcols = (
            [level, 'conductor_temp', 'emissivity', 'absorptivity', 'windangle']
            + [
                i+f'_{season}' for i in ['temperature', 'windspeed', 'ghi', 'ZLR']
                for season in ['summer','winter']
            ]
        )

    return _dfhifld[outcols]


def get_hifld(
    fpath=paths.hifld,
    min_kv=115,
    max_miles=50,
    calc_slr=True,
    calc_zlr=False,
    within_poly=None,
    regional_air_temp=True,
    regional_wind=False,
    regional_irradiance=False,
    regional_conductor_temp=False,
    regional_emissivity=False,
    regional_absorptivity=False,
    aggfunc='representative',
    hifld_ids=slice(None),
    slr_kwargs={},
):
    """
    Inputs
    ------
    min_kv: Minimum voltage to include in results.
        ≥115 kV lines = HV/EHV transmission system (ANSI C84.1-2020)
        (see chat with Jarrad + Greg on 20240909)
    """
    ### Get HIFLD
    if not os.path.exists(fpath):
        err = (
            "Download HIFLD shapefile from "
            "https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines "
            f"and unzip to {fpath}"
        )
        raise Exception(err)

    dflines = read_lines(fpath, crs='ESRI:102008')
    _dfhifld = (
        lookup_diameter_resistance(dflines=dflines)
        .astype({'ID':int})
        .set_index('ID')
        .loc[hifld_ids]
    )
    ## Downselect to US
    dfhifld = _dfhifld.loc[
        (_dfhifld.bounds.maxy <= 1.4e6)
        & (_dfhifld.bounds.miny >= -1.8e6)
        & (_dfhifld.bounds.minx >= -2.5e6)
        & (_dfhifld.bounds.maxx <= 2.5e6)
        ## Remove DC
        & (_dfhifld.VOLT_CLASS != 'DC')
        ## Remove underground
        & (_dfhifld.TYPE.map(lambda x: 'UNDERGROUND' not in x))
    ].copy()
    ## Clip to provided polygon
    if within_poly is not None:
        dfhifld.geometry = dfhifld.intersection(within_poly)
    ## Remove lines below cutoff voltage and above maximum length
    dfhifld['length_miles'] = dfhifld.length / 1609.344
    dfhifld = dfhifld.loc[
        (dfhifld.VOLTAGE >= min_kv)
        & (dfhifld.length_miles <= max_miles)
    ]
    ### Calculate SLR and ZLR if desired
    if calc_slr:
        dfhifld['SLR'] = physics.ampacity(
            diameter_conductor=dfhifld.diameter,
            resistance_conductor=dfhifld.resistance,
            **slr_kwargs,
        )

    if calc_zlr:
        dfhifld = dfhifld.merge(
            calculate_zlr(
                dfhifld=dfhifld,
                regional_air_temp=regional_air_temp,
                regional_wind=regional_wind,
                regional_irradiance=regional_irradiance,
                regional_conductor_temp=regional_conductor_temp,
                regional_emissivity=regional_emissivity,
                regional_absorptivity=regional_absorptivity,
                aggfunc=aggfunc,
            ),
            left_index=True, right_index=True, how='left',
        )

    return dfhifld


def get_lines_and_ratings(
    data_meas='DLR',
    data_base='ALR',
    path_to_meas=None,
    path_to_base=None,
    min_kv=115,
    max_miles=50,
    within_poly=None,
    years=range(2007,2014),
    tz='Etc/GMT+6',
    verbose=1,
    output='percent_diff',
    errors='warn',
    dropna=True,
    hifld_ids=slice(None),
    slr_kwargs={},
):
    """
    Inputs
    ------
    min_kv: Minimum voltage to include in results.
        ≥115 kV lines = HV/EHV transmission system (ANSI C84.1-2020)
        (see chat with Jarrad + Greg on 20240909)
    """
    ### Get HIFLD
    dfhifld = get_hifld(
        min_kv=min_kv, max_miles=max_miles,
        calc_slr=True, within_poly=within_poly,
        calc_zlr=(True if data_base == 'ZLR' else False),
        slr_kwargs=slr_kwargs,
    )
    ids_hifld = dfhifld.index.values

    ### Get results
    if output != 'percent_diff':
        raise NotImplementedError("only output='percent_diff' is currently supported")
    _years = [years] if isinstance(years, int) else years
    _years = tqdm(_years) if verbose else _years
    if data_base is None:
        dfin = pd.concat(
            {
                year: (
                    pd.read_hdf(path_to_meas, key=str(year))
                    .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                )
                for year in _years
            }, names=('year',),
        ### Drop year and switch to output timezone
        ).reset_index(level='year', drop=True).tz_convert(tz)
    else:
        dfin = pd.concat(
            {
                year: (
                    (
                        pd.read_hdf(path_to_meas, key=str(year))
                        .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                    ) / (
                        pd.read_hdf(path_to_base, key=str(year))
                        .reindex(columns=ids_hifld).dropna(axis=1, how='all')[hifld_ids]
                    )
                    - 1
                ) * 100
                for year in _years
            }, names=('year',),
        ### Drop year and switch to output timezone
        ).reset_index(level='year', drop=True).tz_convert(tz)

    ### Subsets
    results_ids = dfin.columns
    unused_from_hifld = [c for c in results_ids if c not in dfhifld.index]
    missing_from_results = [c for c in dfhifld.index if c not in results_ids]
    print('results_ids:', len(results_ids))
    print('missing_from_results:', len(missing_from_results))
    print('unused_from_hifld:', len(unused_from_hifld))
    ## Drop lines missing from results
    dfhifld.drop(missing_from_results, inplace=True, errors='ignore')
    ## Drop lines missing or filtered from hifld
    dfin.drop(columns=unused_from_hifld, inplace=True, errors='ignore')
    print('after dropping lines filtered from hifld:', dfin.shape[1])
    ## Drop lines with missing data
    if dropna:
        dfin.dropna(axis=1, how='any', inplace=True)
    print('after dropping lines with missing data:', dfin.shape[1])

    if dfin.shape[1] != dfhifld.shape[0]:
        err = f"{dfin.shape[1]} results but {dfhifld.shape[0]} lines"
        print('WARNING:', err)
    if errors in ['warn','ignore']:
        return dfhifld, dfin
    else:
        raise IndexError(err)


def get_ratings(
    fpath: str,
    timestamps: list | str | pd.Timestamp | None = None,
    ids: list | int | None = None,
    report_missing: bool = False,
):
    """
    Data file specified by fpath is expected to be in the format used at
    https://data.openei.org/submissions/6231.
    Returns float if a single value, pd.Series if a single timestamp or line, or
    pd.DataFrame if more than one timestamp and line.
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"{fpath} not found. Download data from "
            "https://data.openei.org/submissions/6231 and direct fpath to one of the "
            "{}_SLR_ratio-{}.h5 files."
        )
    all_ids = False
    all_timestamps = False
    if timestamps is None:
        _timestamps = pd.date_range(
            '2007-01-01', '2014-01-01', inclusive='left', freq='h', tz='UTC',
        )
        all_timestamps = True
    elif isinstance(timestamps, str):
        _timestamps = [pd.Timestamp(timestamps).tz_convert('UTC')]
    elif isinstance(timestamps, pd.Timestamp):
        _timestamps = [timestamps.tz_convert('UTC')]
    else:
        _timestamps = [
            pd.Timestamp(t).tz_convert('UTC') for t in timestamps
        ]

    min_t = pd.Timestamp('2007-01-01 00:00+00')
    max_t = pd.Timestamp('2013-12-31 23:00+00')
    assert all([(t >= min_t) and (max_t >= t) for t in _timestamps]), (
        "timestamps must be between 2007-01-01 00:00+00 and 2013-12-31 23:00+00"
    )

    with h5py.File(fpath, 'r') as h:
        columns = pd.Series(h['columns'])
        column_ids = columns.values
        if report_missing and (ids is not None):
            missing = [i for i in ids if i not in column_ids]
            if len(missing):
                print(f"missing {len(missing)} from results")
        id2index = pd.Series(columns.index, index=column_ids)
        if ids is None:
            all_ids = True
            keepcols = id2index
        elif isinstance(ids, int):
            keepcols = id2index.loc[[ids]]
        else:
            keepcols = id2index.reindex(ids).dropna().astype(int)
        timeindex2index = pd.Series(
            index=pd.to_datetime(pd.Series(h['index']).map(lambda x: x.decode())),
            data=range(len(h['index'])),
        ).tz_localize('UTC')
        keepindex = timeindex2index.loc[_timestamps]
        ### Special cases for all_ids and all_timestamps are for faster indexing
        if (len(_timestamps) > 1) and (len(keepcols) > 1):
            if all_ids and all_timestamps:
                dfout = pd.DataFrame(
                    columns=keepcols.index,
                    index=_timestamps,
                    data=h['data'][...],
                ).rename_axis(index='time_index', columns='ID')
            elif all_ids:
                dfout = pd.DataFrame(
                    columns=keepcols.index,
                    index=_timestamps,
                    data=h['data'][keepindex.values, :],
                )
            elif all_timestamps:
                dfout = pd.DataFrame(
                    columns=keepcols.index,
                    index=_timestamps,
                    data=h['data'][:, keepcols.values],
                )
            else:
                dfout = pd.DataFrame(
                    columns=keepcols.index,
                    index=_timestamps,
                    data=h['data'][keepindex.values, keepcols.values],
                )
        elif len(keepcols) > 1:
            if all_ids:
                dfout = pd.Series(
                    h['data'][keepindex.values[0], :],
                    index=keepcols.index,
                    name=_timestamps[0],
                ).rename_axis('ID')
            else:
                dfout = pd.Series(
                    h['data'][keepindex.values[0], keepcols.values],
                    index=keepcols.index,
                    name=_timestamps[0],
                ).rename_axis('ID')
        elif len(_timestamps) > 1:
            if all_timestamps:
                dfout = pd.Series(
                    h['data'][:, keepcols.values[0]],
                    index=_timestamps,
                    name=keepcols.index[0],
                ).rename_axis('time_index')
            else:
                dfout = pd.Series(
                    h['data'][keepindex.values, keepcols.values[0]],
                    index=_timestamps,
                    name=keepcols.index[0],
                ).rename_axis('time_index')
        else:
            dfout = h['data'][keepindex.values[0], keepcols.values[0]]

    return dfout


def make_zlr_timeseries(
    dfhifld,
    time_index,
    winter_months=['Dec','Jan','Feb'],
):
    """Make timeseries of ratings using ZLR from dfhifld
    """
    months = range(1,13)
    monthabbrevs = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthnames = [
        'January',' February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
    ]
    month2num = {
        **dict(zip(monthabbrevs, months)),
        **dict(zip(monthnames, months)),
    }
    winter = [month2num.get(m,m) for m in winter_months]
    summer_times = time_index[~time_index.month.isin(winter)]
    winter_times = time_index[time_index.month.isin(winter)]

    dfout = pd.concat({
        **{t: dfhifld.ZLR_summer for t in summer_times},
        **{t: dfhifld.ZLR_winter for t in winter_times},
    }, axis=1).T.sort_index(axis=0)

    return dfout
